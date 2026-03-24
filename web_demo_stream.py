import argparse
import datetime
import json
import os
import sys
import threading
import time
from threading import Thread, Timer
from queue import Queue
import numpy as np
import torch
import yaml
from utils.xy_inference_utils import generate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.generation.streamers import BaseStreamer

# 新增：Whisper相关import
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from flask import Flask, render_template, request
from flask_socketio import SocketIO, disconnect, emit
from loguru import logger
from web.parms import GlobalParams
from web.pem import generate_self_signed_cert
from transformers import StoppingCriteria

import random
from VoiceDesign.infer_Qwen3 import VoiceDesignInfer


# ================================================================================
# Configuration Loading
# ================================================================================
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_args():
    """Parse command line arguments with YAML config override"""
    parser = argparse.ArgumentParser(description="VITA-Audio")
    parser.add_argument("--config", default='configs/config.yaml', help="path to config file")
    parser.add_argument("--ip", default=None, help="ip of server (overrides config)")
    parser.add_argument("--port", default=None, type=int, help="port of server (overrides config)")
    parser.add_argument("--max_users", type=int, default=None, help="max users (overrides config)")
    parser.add_argument("--timeout", type=int, default=None, help="timeout (overrides config)")
    parser.add_argument("--mode", default="default", type=str, help="mode (overrides config)")
    parser.add_argument("--model", default="default", type=str, help="model (overrides config)")
    parser.add_argument("--role_description", default=None, type=str, help="role description (overrides config)")
    args = parser.parse_args()

    # Load config from YAML
    config = load_config(args.config)

    # Override config with command line arguments if provided
    if args.ip is not None:
        config['server']['ip'] = args.ip
    if args.port is not None:
        config['server']['port'] = args.port
    if args.max_users is not None:
        config['server']['max_users'] = args.max_users
    if args.timeout is not None:
        config['server']['timeout'] = args.timeout
    if args.mode is not None:
        config['mode']['type'] = args.mode
    if args.role_description is not None:
        config['mode']['role_description'] = args.role_description
    if args.model is not None:
        config['mode']['model_name_or_path'] = args.model

    logger.info(f"Loaded config from {args.config}")
    return config


# ================================================================================
# Initialize Configuration
# ================================================================================
config = get_args()

# Set random seeds
random.seed(config['random']['seed'])
torch.manual_seed(config['random']['torch_manual_seed'])

# Add system paths
for path in config['system_paths']:
    sys.path.append(path)

from vita_qinyu.data.processor.audio_processor import add_audio_input_contiguous, AudioProcessor
from vita_qinyu.models.qwen3_sensevoice_xy_v4_51_3.modeling_qwen3 import Qwen3MTPSenseVoiceForCausalLM
from vita_qinyu.models.vita_utu_mtp_sensevoice_xy_v4_51_3.modeling_utu_v1 import VITAUTUV1ForCausalLM


# Extract frequently used config values
target_sample_rate = config['audio']['target_sample_rate']
mode = config['mode']['type']
model_name_or_path = config['model']['model_name_or_path']
device_map = config['model']['device_map']
print(mode, model_name_or_path)

# ================================================================================
# Load Speaker Embeddings
# ================================================================================
spk2emb = torch.load(config['model']['spk2emb_path'])

# Special tokens
SPEAKER_TAG_TOKEN = config['tokens']['speaker_tag']
USER_SPEAKER_TAG_TOKEN = config['tokens']['user_speaker_tag']

# Convert torch_dtype string to torch dtype
torch_dtype_map = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32
}
torch_dtype = torch_dtype_map[config['model']['torch_dtype']]


# ================================================================================
# Load Audio Tokenizer
# ================================================================================
audio_processor = AudioProcessor(
    config['model']['audio_tokenizer_path'],
    config['model']['audio_tokenizer_type'],
    rank=config['model']['audio_tokenizer_rank'],
    text_audio_interval_ratio=config['model']['text_audio_interval_ratio']
)

audio_tokenizer = audio_processor.audio_tokenizer


# ================================================================================
# Load LLM Model
# ================================================================================
llm_config = AutoConfig.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
)
llm_config.audio_model_name_or_path = config['model']['audio_tokenizer_path'][-1]


# Determine chat template and system message based on model type
if "qwen2" in llm_config.model_type.lower() or "qwen3" in llm_config.model_type.lower() or "utu" in llm_config.model_type.lower():
    from utils.get_chat_template import qwen2_chat_template as chat_template
    add_generation_prompt = True

    sys_msg_cfg = config['mode']['default_system_message']
    default_system_message = [
        {
            "role": "system",
            "content": f"Your Name: {sys_msg_cfg['name']}\nYour Origin: {sys_msg_cfg['origin']}.\nRespond in {sys_msg_cfg['response_mode']} manner.\nYour Voice Embedding: {sys_msg_cfg['voice_embedding_tag']}.\nYour Role Description: {sys_msg_cfg['role_description']}",
        },
    ]

if "hunyuan" in llm_config.model_type.lower():
    from utils.get_chat_template import hunyuan_chat_template as chat_template
    add_generation_prompt = False
    default_system_message = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        }
    ]


def get_roleplay_system_message(description):
    """Generate system message for roleplay mode"""
    sys_msg_cfg = config['mode']['default_system_message']
    content = (
        f"Your Name: {sys_msg_cfg['name']}\nYour Origin: {sys_msg_cfg['origin']}.\n"
        f"Respond in {sys_msg_cfg['response_mode']} manner.\n"
        f"Your Voice Embedding: {sys_msg_cfg['voice_embedding_tag']}.\n"
        f"Your Role Description: {description}"
    )

    roleplay_system_message = [
        {
            "role": "system",
            "content": content,
        },
    ]
    return roleplay_system_message

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    chat_template=chat_template,
)
print(f"{tokenizer.get_chat_template()=}")

# Determine model class
if 'utu' in llm_config.model_type.lower():
    ModelForCausalLM = VITAUTUV1ForCausalLM
    extra_sensevoice_token = False
else:
    ModelForCausalLM = Qwen3MTPSenseVoiceForCausalLM
    extra_sensevoice_token = config['model']['extra_sensevoice_token']

# Load model
model = ModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    config=llm_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    attn_implementation=config['model']['attn_implementation'],
).eval()

print(f"{model.config.model_type=}")
print(f"{model.hf_device_map=}")

# Configure generation settings
model.generation_config = GenerationConfig.from_pretrained(
    model_name_or_path, trust_remote_code=True
)
gen_config = config['generation']
model.generation_config.max_new_tokens = gen_config['max_new_tokens']
model.generation_config.chat_format = gen_config['chat_format']
model.generation_config.max_window_size = gen_config['max_window_size']
model.generation_config.use_cache = gen_config['use_cache']
model.generation_config.do_sample = gen_config['do_sample']
model.generation_config.temperature = gen_config['temperature']
model.generation_config.top_k = gen_config['top_k']
model.generation_config.top_p = gen_config['top_p']
model.generation_config.num_beams = gen_config['num_beams']
model.generation_config.pad_token_id = tokenizer.pad_token_id

if model.config.model_type == "hunyuan":
    model.generation_config.eos_token_id = tokenizer.eos_id

print(f"{model.generation_config=}")


# ================================================================================
# Initialize System Message and Speaker Embedding based on Mode
# ================================================================================
message = ""
prompt_audio_path = None

if prompt_audio_path is not None:
    system_message = [
        {
            "role": "system",
            "content": f"Your Voice: <|audio|>\n",
        },
    ]
elif mode == "default":
    speaker_embedding = spk2emb[config['mode']['default_speaker']]
    system_message = default_system_message
elif mode == "roleplay":
    # Initialize timbre model for roleplay mode
    vd_config = config['voice_design']
    timbre_model = VoiceDesignInfer(
        flow_ckpt=vd_config['text_spk_ckpt'],
        config_path=vd_config['text_spk_config'],
        Qwen_ckpt=vd_config['text_encoder_ckpt'],
        device=device_map,
        n_timesteps=vd_config['n_timesteps'],
        temperature=vd_config['temperature']
    )
    speaker_embedding = timbre_model.infer([config['mode']['role_description']], None)
    speaker_embedding = torch.tensor(speaker_embedding[0])
    system_message = get_roleplay_system_message(config['mode']['role_description'])
else:
    raise NotImplementedError(f'{mode} not implemented')
    system_message = default_system_message


# ================================================================================
# Load Turn Detection Model
# ================================================================================
print("load turn models")
turn_config = config['turn_detection']
turn_model = AutoModelForCausalLM.from_pretrained(
    turn_config['model_id'],
    trust_remote_code=True,
    torch_dtype=torch_dtype_map[turn_config['torch_dtype']]
)
turn_tokenizer = AutoTokenizer.from_pretrained(turn_config['model_id'], trust_remote_code=True)

# Move model to GPU
turn_model = turn_model.cuda()
turn_model.eval()


# ================================================================================
# Load ASR Model (Whisper)
# ================================================================================
asr_config = config['asr']
try:
    whisper_processor = WhisperProcessor.from_pretrained(asr_config['model_id'])

    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        asr_config['model_id'],
        torch_dtype=torch_dtype_map[asr_config['torch_dtype']],
        device_map=asr_config['device_map'],
        use_safetensors=asr_config['use_safetensors'],
    ).eval()

    # Configure for Chinese recognition
    whisper_model.config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
        language=asr_config['language'],
        task=asr_config['task']
    )
    logger.info(f"Whisper ASR model loaded successfully: {asr_config['model_id']}")

except Exception as e:
    logger.error(f"Failed to load Whisper ASR: {e}, falling back to base model")
    # Fallback to Whisper-base
    whisper_processor = WhisperProcessor.from_pretrained(asr_config['fallback_model_id'])
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        asr_config['fallback_model_id'],
        torch_dtype=torch_dtype_map[asr_config['torch_dtype']]
    ).cuda().eval()
    whisper_model.config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
        language=asr_config['language'],
        task=asr_config['task']
    )
    logger.info(f"Whisper-base loaded as fallback")


def turn_detect(turn_model, turn_tokenizer, system_prompt, user_input):
    """
    TEN Turn Detector对话状态检测函数
    """
    inf_messages = [{"role": "system", "content": system_prompt}] + [{"role": "user", "content": user_input}]
    input_ids = turn_tokenizer.apply_chat_template(
        inf_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).cuda()

    with torch.no_grad():
        outputs = turn_model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=True,
            top_p=0.1,
            temperature=0.1,
            pad_token_id=turn_tokenizer.eos_token_id
        )

    response = outputs[0][input_ids.shape[-1]:]
    output = turn_tokenizer.decode(response, skip_special_tokens=True)
    return output


# ================================================================================
# Flask Web Service and SocketIO Setup
# ================================================================================
MAX_USERS = config['server']['max_users']
TIMEOUT = config['server']['timeout']

app = Flask(__name__, template_folder=config['paths']['resources_dir'])
socketio = SocketIO(
    app,
    cors_allowed_origins='*',
)
connected_users = {}

# ================================================================================
# Special Tokens Configuration
# ================================================================================
token_config = config['tokens']
AUD_TAG_TOKEN = token_config['audio_tag']
AUD_CONTEXT_TOKEN = token_config['audio_context']
AUD_START_TOKEN = token_config['audio_start']
AUD_END_TOKEN = token_config['audio_end']
CONV_END_TOKEN = token_config['conv_end']
first_audio_token_id = tokenizer.convert_tokens_to_ids(token_config['first_audio_token'])
last_audio_token_id = tokenizer.convert_tokens_to_ids(token_config['last_audio_token'])
AUD_START_ID = tokenizer.convert_tokens_to_ids(AUD_START_TOKEN)
AUD_END_ID = tokenizer.convert_tokens_to_ids(AUD_END_TOKEN)
IM_END_ID = tokenizer.convert_tokens_to_ids(token_config['im_end'])
IM_END_IDS = tokenizer(token_config['im_end'], add_special_tokens=False).input_ids
CONV_END_ID = tokenizer.convert_tokens_to_ids(CONV_END_TOKEN)
MAX_TOKEN_LENGTH = 1024
channels = model.config.num_codebook


# ================================================================================
# Streaming Classes and Parameters
# ================================================================================
class XYTextTokenStreamer(BaseStreamer):

    def __init__(self):
        self.token_queue = Queue()
        self.stop_signal = None

    def put(self, next_t, next_ua, api_result):
        self.token_queue.put((next_t, next_ua, api_result))

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value  = self.token_queue.get()
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class InterruptCriteria(StoppingCriteria):
    """LLM推理中断条件类"""
    def __init__(self, llm_stop_event):
        super().__init__()
        self.llm_stop_event = llm_stop_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.llm_stop_event.is_set():
            logger.info(f"<debug>InterruptCriteria检查: interrupt_event状态={self.llm_stop_event.is_set()}")
            return True
        return False


# Streaming parameters from config
stream_config = config['streaming']
max_code_length = stream_config['max_code_length']
first_code_length = stream_config['first_code_length']
fade_code_length = stream_config['fade_code_length']
cache_wav_len = stream_config['cache_wav_len']
speech_window = np.hamming(2 * cache_wav_len)
num_history = stream_config['num_history']


# ================================================================================
# Audio Decoding Functions
# ================================================================================
def decode_stream(duration_code_length, token_cache_length,
                  decoded_idx, token_cache, past_wav, past_chunk,
                  last_chunk=False):
    """Audio token streaming decode function"""
    if past_chunk is not None:
        past_token_len = past_chunk.shape[2]
        past_overlap_wav_len = past_token_len * audio_tokenizer.xy_tokenizer.decoder_upsample_rate
    else:
        past_token_len = 0
        past_overlap_wav_len = 0

    duration_to_decode = min(duration_code_length, token_cache_length - decoded_idx - channels + 1)
    speech_ids = torch.full((duration_to_decode, channels), 0).to(device_map)

    for j in range(channels):
        speech_ids[..., j] = token_cache[decoded_idx + j : decoded_idx + j + duration_to_decode, j]

    start_time = time.time()
    with torch.no_grad():
        chunk_codes = speech_ids.permute(1, 0).unsqueeze(1)
        # 将超出1024的code设置为0
        chunk_codes = torch.where(chunk_codes > 1024, torch.zeros_like(chunk_codes), chunk_codes)

        if past_chunk is not None:
            input_codes = torch.cat([past_chunk, chunk_codes], dim=-1)
        else:
            input_codes = chunk_codes
        decode_result = audio_tokenizer.xy_tokenizer.inference_detokenize(
            input_codes, torch.tensor([duration_to_decode + past_token_len], device=device_map)
        )

        audio_result = decode_result["y"][0].cpu().detach()

        if past_chunk is not None:
            audio_result = audio_result[:, past_overlap_wav_len - cache_wav_len:]
        past_chunk = chunk_codes

        if not last_chunk:
            if past_wav is not None:
                audio_result = fade_in_out(audio_result, past_wav, speech_window)
            past_wav = audio_result[:, -cache_wav_len :]
            audio_result = audio_result[:, : -cache_wav_len]
        else:
            audio_result = fade_in_out(audio_result, past_wav, speech_window)
            past_wav = None
            past_chunk = None

    logger.info(f"<debug>decode_stream: {time.time() - start_time:.2f}s")

    return audio_result, past_wav, past_chunk


def fade_in_out(fade_in_mel, fade_out_mel, window):
    """Apply fade in/out to audio"""
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


# ================================================================================
# Core Inference Functions
# ================================================================================
def __run_infer_stream(audio_tensor, user_speaker_embedding, llm_stop_event, observation, sid, ishistory):
    """Core streaming inference function"""
    logger.info("=" * 100)
    start_time = time.time()

    if not ishistory:
        if observation is not None:
            if audio_tensor is not None:
                messages = system_message + [
                    {
                        "role": "user",
                        "content": USER_SPEAKER_TAG_TOKEN + message + AUD_TAG_TOKEN,
                    }
                ] + observation
            else:
                messages = system_message + [
                    {
                        "role": "user",
                        "content": message,
                    }
                ] + observation
        else:
            if audio_tensor is not None:
                messages = system_message + [
                    {
                        "role": "user",
                        "content": USER_SPEAKER_TAG_TOKEN + message + AUD_TAG_TOKEN,
                    },
                ]
            else:
                messages = system_message + [
                    {
                        "role": "user",
                        "content": message,
                    },
                ]
    else:
        if audio_tensor is not None:
            messages = system_message + connected_users[sid][2][-2 * num_history:] + [
                {
                    "role": "user",
                    "content": USER_SPEAKER_TAG_TOKEN + message + AUD_TAG_TOKEN,
                },
            ]
        else:
            messages = system_message + connected_users[sid][2][-2 * num_history:] + [
                {
                    "role": "user",
                    "content": message,
                },
            ]
    #  print("*******************************  messages  ***************************************")
    #  print(messages)

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )

    SPEAKER_TAG_ID = tokenizer.convert_tokens_to_ids(SPEAKER_TAG_TOKEN)
    USER_SPEAKER_TAG_ID = tokenizer.convert_tokens_to_ids(USER_SPEAKER_TAG_TOKEN)

    if not ishistory:
        speaker_embedding_list = [speaker_embedding]
        if audio_tensor is not None and audio_tokenizer.apply_to_role("user", is_contiguous=True):
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, [audio_tensor], tokenizer, audio_processor, extra_sensevoice_token=extra_sensevoice_token
            )
            speaker_embedding_list.append(user_speaker_embedding)
        else:
            audios = None
            audio_indices = None
    else:
        history_speaker = connected_users[sid][4]
        num_speaker = len(history_speaker)
        speaker_embedding_list = history_speaker[:1] + history_speaker[max(num_speaker-num_history, 1):]
        if audio_tensor is not None and audio_tokenizer.apply_to_role("user", is_contiguous=True):
            audio_tensor_list = connected_users[sid][3][-1 * num_history:] + [audio_tensor]
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, audio_tensor_list, tokenizer, audio_processor, extra_sensevoice_token=extra_sensevoice_token
            )
            speaker_embedding_list.append(user_speaker_embedding)
        else:
            audios = None
            audio_indices = None

    input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")
    speaker_mask = (input_ids == SPEAKER_TAG_ID) | (input_ids == USER_SPEAKER_TAG_ID)
    speaker_embeddings = torch.stack(speaker_embedding_list)

    if input_ids.shape[1] == 0:
        raise ValueError("ERROR: 输入序列长度不能为0！")
    logger.info("input:", tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)

    local_streamer = XYTextTokenStreamer()

    generation_kwargs = dict(
        tokenizer=tokenizer,
        model=model,
        input_ids=input_ids,
        audios=audios,
        audio_indices=audio_indices,
        streamer=local_streamer,
        speaker_embeddings=speaker_embeddings.cuda(),
        speaker_mask=speaker_mask.cuda(),
        stop_event=llm_stop_event,
    )

    thread = Thread(target=generate, kwargs=generation_kwargs, daemon=True)
    thread.start()
    logger.info(f"LLM 线程名字：{thread.name}")

    token_cache = torch.zeros(MAX_TOKEN_LENGTH, channels, dtype=torch.long, device=device_map)
    token_cache_length = 0
    text_tokens = []
    num_audio_chunk = 0
    past_wav = None
    past_chunk = None
    decoded_idx = 0

    api_result = None
    accum_t = []
    for token in local_streamer:
        next_t, next_ua, api_result = token
        if next_t is None:
            break
        accum_t.append(next_t.item())

        if next_t >= first_audio_token_id and next_t <= last_audio_token_id:
            token_cache[token_cache_length] = next_ua
            token_cache_length += 1
        else:
            if next_t[0] != AUD_START_ID and next_t[0] != AUD_END_ID and next_t[0] != CONV_END_ID:
                text_tokens.append(next_t[0])

        if num_audio_chunk < 1:
            code_length = 20
        elif num_audio_chunk < 3:
            code_length = 16
        elif num_audio_chunk < 5:
            code_length = 24
        elif num_audio_chunk < 7:
            code_length = 36
        else:
            code_length = 48

        if token_cache_length >= decoded_idx + code_length + channels - 1 or next_t == IM_END_ID or tokenizer.decode(accum_t[-len(IM_END_IDS):]) == token_config['im_end']:
            pass
        else:
            continue

        # print('------------------------------------')
        # print(len(token_cache), token_cache_length, len(text_tokens), decoded_idx, code_length)

        if next_t == IM_END_ID or tokenizer.decode(accum_t[-len(IM_END_IDS):]) == token_config['im_end']:
            last_chunk = True
        else:
            last_chunk = False

        speech, past_wav, past_chunk = decode_stream(code_length, token_cache_length, decoded_idx,
                                         token_cache, past_wav, past_chunk, last_chunk=last_chunk)

        decoded_idx += code_length
        speech = speech.squeeze() * 32678.0
        speech = speech.clamp(-32768, 32767).short()
        tts_np = speech.cpu().numpy()
        output_data = tts_np.astype(np.int16)

        first_audio_time = time.time() - start_time
        dt = datetime.datetime.fromtimestamp(first_audio_time)
        formatted_time = dt.strftime("%S.%f")[:-3] + " seconds"
        logger.info(f"audio generation time: {formatted_time}")

        if sid is not None:
            connected_users[sid][1].tts_data.put(output_data)

        first_audio_time = time.time() - start_time
        dt = datetime.datetime.fromtimestamp(first_audio_time)
        formatted_time = dt.strftime("%S.%f")[:-3] + " seconds"
        logger.info(f"audio generation time: {formatted_time}")


        if sid is not None and llm_stop_event.is_set():
            logger.debug(f"<run_infer_stream 函数内部> 打断点, {thread.name} ")
            llm_stop_event.set()

            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(f"线程 {thread.name} 未能在2秒内结束，强制跳过")

            logger.debug(f"llm推理线程已结束, {thread.name}")
            local_streamer.end()
            generated_text = tokenizer.decode(torch.stack(text_tokens))
            logger.info(f"Generated text: {generated_text}")
            return generated_text, None

        if num_audio_chunk == 0:
            first_audio_time = time.time() - start_time
            dt = datetime.datetime.fromtimestamp(first_audio_time)
            formatted_time = dt.strftime("%S.%f")[:-3] + " seconds"
            if sid is not None:
                socketio.emit("first_audio_time", {"time": formatted_time}, to=sid)
            logger.info(f"First audio generation time: {formatted_time}")

        generated_text = tokenizer.decode(torch.stack(text_tokens))
        logger.info(f"Generated text: {generated_text}")

        first_audio_time = time.time() - start_time
        dt = datetime.datetime.fromtimestamp(first_audio_time)
        formatted_time = dt.strftime("%S.%f")[:-3] + " seconds"

        logger.info(f"audio generation time: {formatted_time}")
        num_audio_chunk += 1

    if thread.is_alive():
        thread.join(timeout=1.0)

    return generated_text, api_result


def run_infer_stream(audio_tensor, sid):
    """Main inference streaming function wrapper"""
    logger.info("=" * 100)

    if sid is not None and sid in connected_users:
        llm_stop_event = connected_users[sid][1].interrupt_event
    else:
        llm_stop_event = threading.Event()

    llm_stop_event.clear()

    user_speaker_embedding = audio_processor.process_audio_for_speaker_embedding(audio_tensor)

    generated_text, api_result = __run_infer_stream(audio_tensor, user_speaker_embedding, llm_stop_event, None, sid, True and sid is not None)
    if api_result is not None:
        observation = [{"role": "assistant", "content": generated_text.replace(token_config['im_end'], '')}, {"role": "system", "content": f"<google_search response>{api_result}</google_search response>" }]
        generated_text, _ = __run_infer_stream(audio_tensor, user_speaker_embedding, llm_stop_event, observation, sid, False and sid is not None)

    if sid is not None:
        connected_users[sid][2] = connected_users[sid][2] + [
                {
                    "role": "user",
                    "content": USER_SPEAKER_TAG_TOKEN + message + AUD_TAG_TOKEN,
                },
            ] +  \
            [
                {
                    "role": "assistant",
                    "content": generated_text.replace(token_config['im_end'], ''),
                },
            ]
        connected_users[sid][3] = connected_users[sid][3] + [audio_tensor]
        connected_users[sid][4] = connected_users[sid][4] + [user_speaker_embedding]

    if sid is not None and sid in connected_users:
        llm_stop_event.set()


def reset_model_state(model):
    """Reset model state for next inference"""
    if hasattr(model, 'reset'):
        model.reset()
    elif hasattr(model, 'clean_cache'):
        model.clean_cache()
    model.generation_config.use_cache = True


# ================================================================================
# Real-time Audio Stream Processing, VAD Detection and ASR
# ================================================================================
def send_pcm(sid):
    """Real-time audio stream processing function"""
    chunk_szie = connected_users[sid][1].wakeup_and_vad.get_chunk_size()

    logger.info(f"Sid: {sid} Start listening")
    count_flag = 0
    is_first_time_to_work = True
    asr_text = ''
    while True:
        count_flag += 1
        if connected_users[sid][1].stop_pcm:
            logger.info(f"Sid: {sid} Stop pcm")
            connected_users[sid][1].stop_generate = True
            connected_users[sid][1].stop_tts = True
            break
        time.sleep(0.01)
        e = connected_users[sid][1].pcm_fifo_queue.get(chunk_szie)
        if e is None:
            continue
        if len(e) == config['audio']['chunk_size']:
            pass
        else:
            logger.info("Sid: ", sid, " Received PCM data: ", len(e))

        res = connected_users[sid][1].wakeup_and_vad.predict(e)
        if res is not None:
            if "start" in res:
                logger.info(f"Sid: {sid} Vad start")
                connected_users[sid][1].tts_data.clear()

            elif "cache_dialog" in res:
                logger.info(f"Sid: {sid} Vad end")
                audio_duration = len(res["cache_dialog"]) / target_sample_rate
                if audio_duration < config['audio']['min_audio_duration']:
                    logger.info("The duration of the audio is less than 1s, skipping...")
                    continue

                # ASR + Turn detection
                try:
                    if whisper_model is None or whisper_processor is None:
                        logger.warning("Whisper ASR not available, skipping semantic check")
                    else:
                        # ASR transcription
                        asr_start_time = time.time()
                        wav = res["cache_dialog"].unsqueeze(0).cuda()

                        wav_float32 = wav.squeeze(0).cpu().to(torch.float32)

                        whisper_inputs = whisper_processor(
                            wav_float32,
                            sampling_rate=target_sample_rate,
                            return_tensors="pt"
                        ).input_features

                        whisper_inputs = whisper_inputs.to(whisper_model.device)

                        pred_ids = whisper_model.generate(
                            whisper_inputs,
                            max_new_tokens=asr_config['max_new_tokens'],
                            language=asr_config['language']
                        )
                        asr_text_ = whisper_processor.batch_decode(
                            pred_ids,
                            skip_special_tokens=True
                        )[0].strip()

                        asr_text += asr_text_

                        asr_end_time = time.time()
                        logger.info(f"[ASR] {asr_text} (ASR time: {asr_end_time - asr_start_time:.2f}s)")

                        # Turn detection
                        if len(asr_text.strip()) > 1:
                            socketio.emit("stop_tts", to=sid)
                            turn_start_time = time.time()
                            system_prompt = turn_config['system_prompt']
                            turn_result = turn_detect(turn_model, turn_tokenizer, system_prompt, asr_text)
                            turn_end_time = time.time()

                            should_reply = turn_result.strip().lower() == "finished"
                            should_wait = turn_result.strip().lower() == "wait"
                            logger.info(f"[Tenturn] should_reply={should_reply}, should_wait={should_wait}, result='{turn_result}', time: {turn_end_time - turn_start_time:.2f}s")

                            if should_wait:
                                logger.info("Tenturn result is 'wait', terminating current LLM-TTS without starting new inference")
                                asr_text = ''
                                if connected_users[sid][1].current_infer_thread is not None:
                                    if connected_users[sid][1].current_infer_thread.is_alive():
                                        connected_users[sid][1].interrupt_event.set()
                                        connected_users[sid][1].tts_data.clear()
                                        socketio.emit("stop_tts", to=sid)
                                        time.sleep(0.01)
                                        connected_users[sid][1].current_infer_thread.join(timeout=3.0)
                                    connected_users[sid][1].current_infer_thread = None
                                    connected_users[sid][1].interrupt_event.clear()
                                    connected_users[sid][1].tts_data.clear()
                                continue
                            elif not should_reply:
                                logger.info("Skipping LLM-TTS due to semantic filter")
                                continue
                        else:
                            asr_text = ''
                            logger.info("ASR result empty, skipping")
                            continue

                except Exception as e:
                    logger.error(f"ASR/Tenturn error: {e}, proceeding with original flow")

                if is_first_time_to_work:
                    logger.info(f"第一次工作，不打断")
                    asr_text = ' '
                    connected_users[sid][1].current_infer_thread = threading.Thread(
                        target=run_infer_stream,
                        args=((res["cache_dialog"].unsqueeze(0), target_sample_rate), sid)
                    )
                    connected_users[sid][1].current_infer_thread.start()
                    is_first_time_to_work = False
                else:
                    if connected_users[sid][1].current_infer_thread is not None:
                        logger.debug(f"pcm主线程存在，id是：{connected_users[sid][1].current_infer_thread}")
                        if connected_users[sid][1].current_infer_thread.is_alive():
                            logger.debug(f"pcm主线程是活的, 进行打断")
                            connected_users[sid][1].interrupt_event.set()

                            connected_users[sid][1].tts_data.clear()
                            socketio.emit("stop_tts", to=sid)

                            time.sleep(0.1)
                            connected_users[sid][1].current_infer_thread.join(timeout=3.0)
                            logger.debug(f"pcm主线程正常打断并join")

                        connected_users[sid][1].current_infer_thread = None
                        connected_users[sid][1].interrupt_event.clear()
                        connected_users[sid][1].tts_data.clear()
                        time.sleep(0.1)

                    connected_users[sid][1].interrupt_event.clear()

                    asr_text = ' '
                    connected_users[sid][1].current_infer_thread = threading.Thread(
                        target=run_infer_stream,
                        args=((res["cache_dialog"].unsqueeze(0), target_sample_rate), sid)
                    )
                    connected_users[sid][1].current_infer_thread.start()


def disconnect_user(sid):
    """Disconnect user and cleanup resources"""
    if sid in connected_users:
        logger.info(f"Disconnecting user {sid} due to time out")
        socketio.emit("out_time", to=sid)
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(2)
        del connected_users[sid]


# ================================================================================
# Flask Routes and SocketIO Handlers
# ================================================================================
@app.route("/")
#  def index():
#      return render_template("index_v2.html")
def index():
  return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    try:
        if len(connected_users) >= MAX_USERS:
            logger.info("Too many users connected, disconnecting new user")
            emit("too_many_users")
            return

        sid = request.sid
        connected_users[sid] = []
        connected_users[sid].append(Timer(TIMEOUT, disconnect_user, [sid]))
        connected_users[sid].append(GlobalParams())

        # Multi-turn related
        connected_users[sid].append([])  # history messages
        connected_users[sid].append([])  # history audios
        connected_users[sid].append([speaker_embedding])  # history speaker

        connected_users[sid][0].start()
        pcm_thread = threading.Thread(target=send_pcm, args=(sid,))
        pcm_thread.start()
        logger.info(f"User {sid} connected")
    except Exception as e:
        logger.error(f"Error in handle_connect: {e}")
        disconnect()


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]
    logger.info(f"User {sid} disconnected")


@socketio.on("recording-started")
def handle_recording_started():
    sid = request.sid
    if sid in connected_users:
        socketio.emit("stop_tts", to=sid)
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit("stop_tts", to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    logger.info("Recording started")


@socketio.on("recording-stopped")
def handle_recording_stopped():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit("stop_tts", to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    logger.info("Recording stopped")


@socketio.on("tts_playing")
def handle_tts_playing():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][1].tts_end_lock = True


@socketio.on("tts_stopped")
def handle_tts_stopped():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][1].tts_end_lock = False


@socketio.on("audio")
def handle_audio(data):
    sid = request.sid
    if sid in connected_users:
        if not connected_users[sid][1].tts_data.is_empty():
            connected_users[sid][0].cancel()
            connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
            connected_users[sid][0].start()
            output_data = connected_users[sid][1].tts_data.get()

            if output_data is not None:
                emit("audio", output_data.tobytes())

        if connected_users[sid][1].tts_over_time > 0:
            socketio.emit("stop_tts", to=sid)
            connected_users[sid][1].tts_over_time = 0

        data = json.loads(data)

        audio_data = np.frombuffer(bytes(data["audio"]), dtype=np.int16)
        connected_users[sid][1].pcm_fifo_queue.put(audio_data.astype(np.float32) / 32768.0)

    else:
        disconnect()


# ================================================================================
# Main Entry Point
# ================================================================================
if __name__ == "__main__":
    try:
        logger.info("Start VITA-Audio sever")

        # Ensure necessary directories exist
        os.makedirs(config['paths']['resources_dir'], exist_ok=True)
        os.makedirs(config['paths']['output_dir'], exist_ok=True)

        cert_file = config['server']['cert_file']
        key_file = config['server']['key_file']
        if config['server']['ssl_enabled'] and (not os.path.exists(cert_file) or not os.path.exists(key_file)):
            generate_self_signed_cert(cert_file, key_file)

        logger.info("=" * 100)
        logger.info("Warmup...")
        # Check if warmup audio file exists
        warmup_audio = config['paths']['warmup_audio']
        if os.path.exists(warmup_audio):
            run_infer_stream(warmup_audio, None)
        else:
            logger.warning(f"Warmup audio file not found: {warmup_audio}")
        logger.info("Warmup Done.")
        logger.info("=" * 100)

        # Start server
        if config['server']['ssl_enabled']:
            socketio.run(app, host=config['server']['ip'], port=config['server']['port'],
                        ssl_context=(cert_file, key_file))
        else:
            socketio.run(app, host=config['server']['ip'], port=config['server']['port'],
                        allow_unsafe_werkzeug=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
