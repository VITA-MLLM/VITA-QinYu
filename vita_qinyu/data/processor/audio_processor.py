import json
import math
import os

import numpy as np
import torch

import natsort
from vita_qinyu.tokenizer import get_audio_tokenizer

from ...constants import IGNORE_TOKEN_ID


class AudioProcessor:
    def __init__(
        self,
        audio_tokenizer_path=None,
        audio_tokenizer_type=None,
        flow_path=None,
        rank=None,
        text_audio_interval_ratio=None,
    ):

        self.audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_path,
            audio_tokenizer_type,
            flow_path=flow_path,
            rank=rank,
        )

        self.audio_tokenizer_type = audio_tokenizer_type

        self.text_audio_interval_ratio = text_audio_interval_ratio

        # self.load_model()

    def load_model(self):
        if self.audio_tokenizer is not None:
            self.audio_tokenizer.load_model()

    def process_audio_for_speaker_embedding(self, audio_path):
        if type(audio_path) is tuple:
            speaker_embedding = self.audio_tokenizer.process_audio_for_speaker_embedding(audio_path)
            return speaker_embedding
        AUDIO_CACHE_DIR = os.environ.get("AUDIO_CACHE_DIR", None)
        _, _, speaker_embedding_type = self.audio_tokenizer_type.split("_")
        # if audio_path.count(':') == 0:
        #     cache_path = os.path.splitext(audio_path)[0] + f"_{speaker_embedding_type}.pt"
        # elif audio_path.count(':') == 2:
        #     cache_path = audio_path + f"_{speaker_embedding_type}.pt"
        # else:
        #     raise NotImplementedError(f"{audio_path} has unkown format.")
        cache_path = get_cache_path(audio_path, speaker_embedding_type, 'pt')
        if AUDIO_CACHE_DIR is not None and AUDIO_CACHE_DIR != "":
            cache_path = os.path.join(AUDIO_CACHE_DIR, cache_path.lstrip('/'))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        try:
            if os.path.isfile(cache_path):
                speaker_embedding = torch.load(cache_path)
                return speaker_embedding
                # with open(cache_path, "r") as f:
                #     speaker_embedding = json.load(f)
                # return audio_data
        except Exception as e:
            print(f"Failed to load speaker embedding from file {cache_path=}: {e=}")
            pass

        speaker_embedding = self.audio_tokenizer.process_audio_for_speaker_embedding(audio_path)
        try:
            if isinstance(speaker_embedding, torch.Tensor):
                torch.save(speaker_embedding, cache_path)
        except OSError as e:
            pass
        except Exception as e:
            print(f"Failed to save audio token to file {cache_path=}: {e=}")
            pass

        return speaker_embedding

    def process_audio(self, audio_path, is_discrete=False, is_contiguous=False, **kwargs):

        assert not (is_discrete and is_contiguous)
        assert is_discrete or is_contiguous

        AUDIO_CACHE_DIR = os.environ.get("AUDIO_CACHE_DIR", None)

        if is_discrete:
            _, audio_tokenizer_type, *_ = self.audio_tokenizer_type.split("_")

            # cache_path = os.path.splitext(audio_path)[0] + f"_{audio_tokenizer_type}.json"
            cache_path = get_cache_path(audio_path, audio_tokenizer_type, 'json')
            if AUDIO_CACHE_DIR is not None and AUDIO_CACHE_DIR != "":
                cache_path = os.path.join(AUDIO_CACHE_DIR, cache_path.lstrip('/'))
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            try:
                if os.path.isfile(cache_path):
                    with open(cache_path, "r") as f:
                        audio_data = json.load(f)
                    return audio_data
            except Exception as e:
                print(f"Failed to load audio token from file {cache_path=}: {e=}")
                pass

        audio_data = self.audio_tokenizer.encode(
            audio_path, is_discrete=is_discrete, is_contiguous=is_contiguous, **kwargs
        )
        # print(f"{len(audio_data)=}")

        if is_discrete:
            try:
                if isinstance(audio_data, list):
                    with open(cache_path, "w") as f:
                        json.dump(audio_data, f)
            except OSError as e:
                pass
            except Exception as e:
                print(f"Failed to save audio token to file {cache_path=}: {e=}")
                pass

        return audio_data

    @property
    def is_discrete(self):
        return self.audio_tokenizer.is_discrete

    @property
    def is_contiguous(self):
        return self.audio_tokenizer.is_contiguous

    def apply_to_role(self, role, **kwargs):
        return self.audio_tokenizer.apply_to_role(role, **kwargs)

    def text_audio_interval(
        self, content_input_id, AUD_START_ID, AUD_END_ID,
        use_audio_special_token=True
    ):
        if hasattr(self.audio_tokenizer, "text_audio_interval"):
            return self.audio_tokenizer.text_audio_interval(
                content_input_id,
                AUD_START_ID,
                AUD_END_ID,
                self.text_audio_interval_ratio,
                use_audio_special_token=use_audio_special_token
            )
        return text_audio_interval(
            content_input_id,
            AUD_START_ID,
            AUD_END_ID,
            self.text_audio_interval_ratio,
            use_audio_special_token=use_audio_special_token
        )

def get_cache_path(audio_path, cache_type, cache_ext):
    if audio_path.count(':') == 0:
        cache_path = os.path.splitext(audio_path)[0] + f"_{cache_type}.{cache_ext}"
    elif audio_path.count(':') == 2:
        cache_path = audio_path + f"_{cache_type}.{cache_ext}"
    else:
        raise NotImplementedError(f"{audio_path} has unkown format.")
    return cache_path

def add_audio_input_contiguous(input_ids, audio_paths, tokenizer, audio_processor, targets=None, extra_sensevoice_token=True):

    from ...constants import (
        AUD_START_TOKEN,
        AUD_END_TOKEN,
        AUD_TAG_TOKEN,
        AUD_CONTEXT_TOKEN,
    )

    AUD_CONTEXT_ID = tokenizer(AUD_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids
    AUD_START_ID = tokenizer(AUD_START_TOKEN, add_special_tokens=False).input_ids
    AUD_END_ID = tokenizer(AUD_END_TOKEN, add_special_tokens=False).input_ids

    assert len(AUD_CONTEXT_ID) == 1
    assert len(AUD_START_ID) == 1
    assert len(AUD_END_ID) == 1

    AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
    AUD_TAG_ID = AUD_TAG_ID[0]
    AUD_START_ID = AUD_START_ID[0]
    AUD_END_ID = AUD_END_ID[0]

    aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
    assert len(aud_positions) == len(audio_paths), f"{input_ids=} {audio_paths=} {AUD_TAG_ID=}"

    audios = []
    audio_indices = []
    new_input_ids = []
    new_targets = []
    st = 0
    for aud_idx, aud_pos in enumerate(aud_positions):
        # audio = audio_tokenizer.encode(audio_paths[aud_idx], is_contiguous=True)
        audio = audio_processor.process_audio(audio_paths[aud_idx], is_contiguous=True)
        audios.append(audio)
        if extra_sensevoice_token:
            audio_token_length = audio.size(0) + 4
        else:
            audio_token_length = audio.size(0)

        new_input_ids += input_ids[st:aud_pos]
        if targets is not None:
            new_targets += targets[st:aud_pos]

        new_input_ids += [AUD_START_ID]
        if targets is not None:
            new_targets += [IGNORE_TOKEN_ID]

        audio_indice_b = torch.zeros(
            1, audio_token_length, dtype=torch.int64
        )  # This will change in collate_fn
        audio_indice_s = (
            torch.arange(len(new_input_ids), len(new_input_ids) + audio_token_length)
            .unsqueeze(0)
            .repeat(1, 1)
        )
        audio_indice_b_s = torch.stack(
            [audio_indice_b, audio_indice_s], dim=0
        )  # 2, num_image, image_length
        audio_indices.append(audio_indice_b_s)

        new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
        if targets is not None:
            new_targets += [IGNORE_TOKEN_ID] * audio_token_length

        new_input_ids += [AUD_END_ID]
        if targets is not None:
            new_targets += [IGNORE_TOKEN_ID]

        st = aud_pos + 1

        # if max(audio_token_length) > 512:
        #     raise Exception(f"Audio is to long {speech_lengths}")

    new_input_ids += input_ids[st:]
    if targets is not None:
        new_targets += targets[st:]

    input_ids = new_input_ids
    if targets is not None:
        targets = new_targets

    if targets is not None:
        return input_ids, audios, audio_indices, targets

    return input_ids, audios, audio_indices


def text_audio_interval(input_ids, AUD_START_ID, AUD_END_ID, text_audio_interval_ratio):

    if text_audio_interval_ratio is None:
        #                            T   A
        text_audio_interval_ratio = [13, 26]
        #                            T  A  T  A  T  A
        text_audio_interval_ratio = [1, 4, 3, 8, 4, 10]
        #                            T  A   T  A
        text_audio_interval_ratio = [1, 10, 4, 10]

    text_nums = text_audio_interval_ratio[::2]
    audio_nums = text_audio_interval_ratio[1::2]

    # exclude AUD_START and AUD_END
    audio_nums = [x - 2 for x in audio_nums]

    st = [i for i, x in enumerate(input_ids) if x == AUD_START_ID]
    ed = [i for i, x in enumerate(input_ids) if x == AUD_END_ID]

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
        interval_input_ids += text_tokens + [AUD_START_ID] + audio_tokens + [AUD_END_ID]
        # interval_input_ids += text_tokens + audio_tokens

    return interval_input_ids
