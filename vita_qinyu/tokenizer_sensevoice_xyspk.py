import glob
import io
import logging
import math
import os
import tarfile
import uuid
import numpy as np
import safetensors
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

import torchaudio

from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from flow_inference import AudioDecoder

from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.sense_voice.model import SenseVoiceSmall

from .constants import (
    AUD_CONTEXT_TOKEN,
    AUD_END_TOKEN,
    AUD_START_TOKEN,
    AUD_TAG_TOKEN,
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    IMG_TAG_TOKEN,
    PATCH_CONTEXT_TOKEN,
    PATCH_END_TOKEN,
    PATCH_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    VID_CONTEXT_TOKEN,
    VID_END_TOKEN,
    VID_START_TOKEN,
    VID_TAG_TOKEN,
    CONV_END_TOKEN,
    FUNC_START_TOKEN,
    FUNC_END_TOKEN,
    SPEAKER_TAG_TOKEN,
    USER_SPEAKER_TAG_TOKEN,
    IM_END_TOKEN,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_tokenizer_for_sensevoice_xyspktokenizer(tokenizer):
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        VID_START_TOKEN,
        VID_END_TOKEN,
        VID_CONTEXT_TOKEN,
        PATCH_START_TOKEN,
        PATCH_END_TOKEN,
        PATCH_CONTEXT_TOKEN,
        AUD_START_TOKEN,
        AUD_END_TOKEN,
        AUD_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        IMG_TAG_TOKEN,
        VID_TAG_TOKEN,
        AUD_TAG_TOKEN,
        CONV_END_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    codebook_size = 1024
    pad_size = 8
    # token_list = [f"<|audio_{i}|>" for i in range(16384)]
    token_list = [
        f"<|audio_{i}_{j}|>" if j < codebook_size else f"<|audio_{i}_{j-codebook_size}_pad|>" for i in range(8) for j in range(codebook_size+pad_size)
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=False)

    tokenizer.add_tokens([
        '<|audio_token_start|>',
        '<|audio_token_end|>',
        FUNC_START_TOKEN,
        FUNC_END_TOKEN,
        SPEAKER_TAG_TOKEN,
        USER_SPEAKER_TAG_TOKEN,
        IM_END_TOKEN,
    ], special_tokens=True)
    # logger.info(f"tokenizer {tokenizer}")
    return tokenizer


class SenseVoiceXYSpkTokenizer:
    def __init__(self, model_name_or_path, rank=None):

        if rank is None and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            self.rank = rank % 8
        else:
            self.rank = rank
        logger.info(f"{self.rank=}")

        self.model_name_or_path = model_name_or_path
        self.sample_rate = 16000
        self.is_discrete = True
        self.is_contiguous = True
        import faulthandler
        faulthandler.enable()
        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}
        for sample_rate in [8000, 22050, 24000, 32000, 44100, 48000]:
            self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )

    def load_model(self):
        if hasattr(self, "xy_tokenizer"):
            return

        if self.rank is not None:
            self.device = f"cuda:{self.rank}"
            torch.cuda.set_device(self.rank)
        else:
            self.device = "cpu"

        model_path, speaker_embedding_model_path, model_dir= self.model_name_or_path # xy_tokenizer.ckpt
        config_path = model_path.replace('xy_tokenizer.ckpt', 'xy_tokenizer_config.yaml')
        logger.info(f"{self.device=} Loading XYTokenizer")
        _, self.kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=self.device)
        logger.info(f"{self.device=} Loading SenseVoiceSmall Done")
        from xy_tokenizer.model import XY_Tokenizer
        generator = XY_Tokenizer.load_from_checkpoint(
            config_path=config_path,
            ckpt_path=model_path
        ).to(self.device).eval()

        self.xy_tokenizer = generator
        from speakerlab.process.processor import FBank
        from speakerlab.models.campplus.DTDNN import CAMPPlus
        embedding_model = CAMPPlus(feat_dim=80, embedding_size=192)
        embedding_model.load_state_dict(torch.load(speaker_embedding_model_path, map_location='cpu'))
        embedding_model.to(self.device)
        self.speaker_embedding_model = embedding_model.eval()
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        logger.info(f"{self.device=} Loading speaker embedding model Done")


    def encode(self, audio_path, is_discrete=False, is_contiguous=True, **kwargs):

        if not hasattr(self, "whisper_model"):
            self.load_model()

        assert not (is_discrete and is_contiguous)
        assert is_discrete or is_contiguous

        if is_discrete:
            audio_tokens = self.extract_speech_token(
                self.xy_tokenizer, [audio_path], device=self.device
            )[0] # 8 x T
            audio_tokens_ = []
            _, T = audio_tokens.shape
            layer_indices = torch.arange(self.xy_tokenizer.nq).repeat(T)
            audio_tokens_ = [tuple([int(i), int(j)]) for i, j in zip(layer_indices, audio_tokens.T.reshape(-1))]
            return audio_tokens_

        if is_contiguous:

            # audio, sample_rate = torchaudio.load(audio_path)
            audio, sample_rate = load_audio(audio_path)
            audio = audio.mean(0)
            duration = len(audio) / sample_rate # in second
            if duration > 60 * 5: # 10 min
                raise ValueError(f'{audio_path=} is too long {audio.shape=} {sample_rate=}')
            if sample_rate != self.sample_rate:
                if sample_rate not in self._resample_buffer:
                    self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=self.sample_rate
                    ).to(self.device)
                audio = audio.to(self.device)
                self._resample_buffer[sample_rate].to(self.device)
                audio = self._resample_buffer[sample_rate](audio[None, :])[0, :]
                audio = audio.cpu()
            # resampler = torchaudio.transforms.Resample(
            #     orig_freq=sample_rate, new_freq=self.sample_rate
            # )
            # audio = resampler(audio[None, :])[0, :]
            # audio = audio.to(self.device)

            frontend = self.kwargs["frontend"]
            speech, speech_lengths = extract_fbank(audio, data_type="sound", frontend=frontend)
            speech = speech[0]
            # print(f"{speech_lengths=}")
            # print(f"{speech.size()=}")

            return speech

    def decode(self, audio_tokens, option_steps=10, **kwargs):
        if not hasattr(self, "xy_tokenizer"):
            self.load_model()

        tts_token = torch.tensor(audio_tokens, device=self.device)
        tts_token = [tts_token.view(-1, 8).T]

        decode_result = self.xy_tokenizer.decode(tts_token, overlap_seconds=10)
        syn_wav_list = decode_result["syn_wav_list"][0]
        return syn_wav_list

    def apply_to_role(self, role, **kwargs):
        is_discrete = kwargs.get("is_discrete", False)
        if is_discrete and role in ["assistant", "gpt"]:
            return True

        is_contiguous = kwargs.get("is_contiguous", False)
        if is_contiguous and role in ["user", "human"]:
            return True

        return False

    def extract_speech_token(self, model, utts, device="cuda"):
        with torch.no_grad():
            codes_lists = []
            for idx, utt in enumerate(utts):
                if isinstance(utt, tuple):
                    audio, sample_rate = utt
                # elif utt.count(':') == 2 and 'raw' in utt:
                else:
                    audio, sample_rate = load_audio(utt)
                    # audio, sample_rate = torchaudio.load(utt)
                audio = audio.to(device)
                if sample_rate != 16000:
                    if sample_rate not in self._resample_buffer:
                        self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                            orig_freq=sample_rate, new_freq=16000
                        ).to(device)
                    self._resample_buffer[sample_rate].to(device)
                    audio = self._resample_buffer[sample_rate](audio)
                audio = audio[0]
                # audio = audio.cpu().numpy()
                wav_list = [audio]
                encode_result = self.xy_tokenizer.encode(wav_list)
                codes_list = encode_result["codes_list"][0]
                codes_lists.append(codes_list)
        return codes_lists

    def extract_speaker_embedding(self, utts, obj_fs=16000, chunk_size=10, max_load_len=90, device="cuda"):
        with torch.no_grad():
            speaker_embeddings = []
            for idx, utt in enumerate(utts):
                if isinstance(utt, tuple):
                    audio, sample_rate = utt
                else:
                    # audio, sample_rate = torchaudio.load(utt)
                    audio, sample_rate = load_audio(utt)
                audio = audio.to(device)
                if sample_rate != 16000:
                    if sample_rate not in self._resample_buffer:
                        self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                            orig_freq=sample_rate, new_freq=16000
                        ).to(device)
                    self._resample_buffer[sample_rate].to(device)
                    audio = self._resample_buffer[sample_rate](audio)
                # audio = audio[0]
                audio = audio.mean(dim=0) # .cpu()
                audio = audio[:int(max_load_len*obj_fs)]
                audios = self.chunk_wav(audio, int(chunk_size*obj_fs))
                feats = []
                for audio in audios:
                    feats.append(self.feature_extractor(audio))
                feats = torch.stack(feats)
                embedding = self.speaker_embedding_model(feats).mean(0).detach().cpu()
                speaker_embeddings.append(embedding)
        return speaker_embeddings

    def process_audio_for_speaker_embedding(self, wav_path):
        return self.extract_speaker_embedding([wav_path], device=self.device)[0]

    def chunk_wav(self, wav, chunk_sample_size):
        def circle_pad(wav, object_len):
            wav_len = wav.shape[0]
            n = int(np.ceil(object_len/wav_len))
            wav = [wav for i in range(n)]
            wav = torch.cat(wav)
            return wav[:object_len]

        n = int(np.ceil(wav.shape[0] / chunk_sample_size))
        wav = circle_pad(wav, n*chunk_sample_size)
        wavs = [wav[i*chunk_sample_size:(i+1)*chunk_sample_size] for i in range(n)]

        return wavs

    def load_wav(self, wav_path, obj_fs=16000, chunk_size=10, max_load_len=90):
        #start_time = time.time()
        # wav, fs = torchaudio.load(wav_path)
        wav, fs = load_audio(wav_path)
        #print(f'[INFO]: Load wav {wav_path} cost {time.time() - start_time} seconds.')

        #if fs != obj_fs:
        #    print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
        #    wav, fs = torchaudio.sox_effects.apply_effects_tensor(
        #        wav, fs, effects=[['rate', str(obj_fs)]]
        #    )
        wav = wav[0, :].unsqueeze(0)
        if fs != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(wav)
        wav = wav.mean(dim=0)
        wav = wav[:int(max_load_len*obj_fs)]
        wavs = self.chunk_wav(wav, int(chunk_size*obj_fs))
        return wavs


    def text_audio_interval(self,
        input_ids, AUD_START_ID, AUD_END_ID, text_audio_interval_ratio,
        use_audio_special_token=True):
        num_codebook = self.xy_tokenizer.nq
        assert num_codebook == 8, f'{num_codebook=}!=8'
        if text_audio_interval_ratio is None:
            #                            T   A
            text_audio_interval_ratio = [13, 26]
            #                            T  A  T  A  T  A
            text_audio_interval_ratio = [1, 4, 3, 8, 4, 10]
            #                            T  A   T  A
            text_audio_interval_ratio = [1, 10, 4, 10]

        text_nums = text_audio_interval_ratio[::2]
        # +2 for AUD_START and AUD_END
        if use_audio_special_token:
            audio_nums = [x-2 for x in text_audio_interval_ratio[1::2]]
        else:
            audio_nums = [x for x in text_audio_interval_ratio[1::2]]
        # exclude AUD_START and AUD_END
        # audio_nums = [x - 2 for x in audio_nums]

        st = [i for i, x in enumerate(input_ids) if x == AUD_START_ID]
        ed = [i for i, x in enumerate(input_ids) if x == AUD_END_ID]

        # import pdb; pdb.set_trace()
        # only text
        if len(st) == 0 and len(ed) == 0:
            return input_ids

        assert len(st) == 1
        assert len(ed) == 1

        st = st[0]
        ed = ed[0]

        assert st < ed

        # only audio
        if st == 0 and ed == len(input_ids) - 1:
            return input_ids

        audio_tokens = input_ids[st + 1 : ed]
        text_tokens = input_ids[:st] + input_ids[ed + 1 :]

        audio_tokens_chunks = []
        while len(audio_tokens) > 0:
            if len(audio_nums) > 1:
                audio_num = audio_nums.pop(0)
            else:
                audio_num = audio_nums[0]

            audio_tokens_chunks.append(audio_tokens[:audio_num])
            audio_tokens = audio_tokens[audio_num:]

        text_tokens_chunks = []
        while len(text_tokens) > 0:
            if len(text_nums) > 1:
                text_num = text_nums.pop(0)
            else:
                text_num = text_nums[0]

            text_tokens_chunks.append(text_tokens[:text_num])
            text_tokens = text_tokens[text_num:]

        chunk_num = min(len(audio_tokens_chunks), len(text_tokens_chunks))
        audio_tokens_chunks = audio_tokens_chunks[: chunk_num - 1] + [
            sum(audio_tokens_chunks[chunk_num - 1 :], [])
        ]
        text_tokens_chunks = text_tokens_chunks[: chunk_num - 1] + [
            sum(text_tokens_chunks[chunk_num - 1 :], [])
        ]

        interval_input_ids = []
        for text_tokens, audio_tokens in zip(text_tokens_chunks, audio_tokens_chunks):
            if use_audio_special_token:
                interval_input_ids += text_tokens + [AUD_START_ID] + audio_tokens + [AUD_END_ID]
            else:
                interval_input_ids += text_tokens + audio_tokens

        return interval_input_ids

def load_audio(file_path):
    def load(file_path, start, end):
        with open(file_path, 'rb') as f:
            f.seek(start)
            content = f.read(end - start)
            return io.BytesIO(content)
    if type(file_path) is tuple:
        return file_path
    elif file_path.count(':') == 2 and 'raw' in file_path:
        fp, s, e = file_path.split(':')
        s, e = int(s), int(e)
        audio_file = load(fp, s, e)
    elif os.path.isfile(file_path):
        audio_file = file_path
    else:
        raise OSError(f'{file_path} does not exist')
    audio, sample_rate = torchaudio.load(audio_file)
    if file_path.count(':') == 2 and 'raw' in file_path:
        audio_file.close()
    return audio, sample_rate
