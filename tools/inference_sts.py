import argparse
import os
import re
import sys
import time
import json
import tqdm
import random
import logging

import torch
import torchaudio
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation import GenerationConfig

from vita_qinyu.data.processor.audio_processor import add_audio_input_contiguous, AudioProcessor
from vita_qinyu.tokenizer import get_audio_tokenizer
from utils.xy_inference_utils import generate


parser = argparse.ArgumentParser(description="VITA-QinYu")
parser.add_argument("--model",  type=str, help="model")
parser.add_argument("--output_dir", type=str, help="output dir")
args = parser.parse_args()
output_dir = args.output_dir


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.manual_seed(1234)

device_map = "cuda:0"
audio_tokenizer_rank = 0
torch_dtype = torch.bfloat16

AUDIO_TAG_TOKEN = "<|audio|>"
IM_END_TOKEN = "<|im_end|>"
CONV_END_TOKEN = "<|end_of_conv|>"
FIRST_AUDIO_TOKEN = "<|audio_0_0|>"
LAST_AUDIO_TOKEN = "<|audio_7_7_pad|>"
SPEAKER_TAG_TOKEN = "<|speaker|>"
USER_SPEAKER_TAG_TOKEN = "<|user_speaker|>"
CODEBOOK_SIZE = 1024
chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""

sys.path.append("third_party/MOSS-TTSD-v0.7/XY_Tokenizer")
sys.path.append("third_party/GLM-4-Voice")
sys.path.append("third_party/GLM-4-Voice/third_party/Matcha-TTS/")
sys.path.append("third_party/3D-Speaker")

audio_tokenizer_path = [
    "/vita-qinyu-models/VITA-QinYu-Models/xy_tokenizer.ckpt",
    "/vita-qinyu-models/VITA-QinYu-Models/campplus_cn_common.bin",
    "/vita-qinyu-models/FunAudioLLM/SenseVoiceSmall",
]

spk2emb_path = "/vita-qinyu-models/VITA-QinYu-Models/spk2embeds_roleplay.pt"
audio_tokenizer_type = "sensevoice_xytokenizer_speaker"
model_name_or_path = None

os.makedirs(args.output_dir, exist_ok=True)

def is_audio_token(x):
    first_audio_token_id = self.tokenizer.convert_tokens_to_ids(FIRST_AUDIO_TOKEN)
    last_audio_token_id = self.tokenizer.convert_tokens_to_ids(LAST_AUDIO_TOKEN)
    return (x >= first_audio_token_id) & (x <= last_audio_token_id)

class S2SInference:
    def __init__(
        self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, spk2emb_path,
    ):

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        config.audio_model_name_or_path = audio_tokenizer_path[-1]

        add_generation_prompt = True
        default_system_message = []
        qinyu4o_system_message = [{
            "role": "system",
            "content": "Your Name: QinYu-4o\nYour Origin: TME and Youtu Collaboration.\nRespond in text-audio interleaved manner.\nYour Voice Embedding: <|speaker|>.\nYour Role Description: 一个声音可爱甜美的女性。",
        }]

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            chat_template=chat_template,
        )
        print(f"{tokenizer.get_chat_template()=}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).eval()

        if model.config.model_type == "vita_utu_v1":
            self.extra_sensevoice_token = False
        elif model.config.model_type == "qwen3_sensevoice_xy":
            self.extra_sensevoice_token = True
        else:
            raise NotImplementedError(f"Model type not implemented: {model.config.model_type=}")

        print(f"{model.config.model_type=}")
        print(f"{model.hf_device_map=}")

        audio_processor = AudioProcessor(
            audio_tokenizer_path,
            audio_tokenizer_type,
            rank=audio_tokenizer_rank,
            text_audio_interval_ratio=[4, 8]
        )
        audio_tokenizer = audio_processor.audio_tokenizer
        audio_tokenizer.load_model()

        self.model = model
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.audio_processor = audio_processor
        self.add_generation_prompt = add_generation_prompt
        self.default_system_message = default_system_message
        self.qinyu4o_system_message = qinyu4o_system_message

        self.spk2emb = torch.load(spk2emb_path)


    def run_infer_tts(self, message):
        system_message = []
        messages = messages + [{
            "role": "user",
            "content": message,
        }]


    def run_infer_multiturn(
        self, input_messages, audio_paths=None, mode="qinyu4o", assistant_speaker="tian", system_message=None
    ):
        SPEAKER_TAG_ID = self.tokenizer.convert_tokens_to_ids(SPEAKER_TAG_TOKEN)
        USER_SPEAKER_TAG_ID = self.tokenizer.convert_tokens_to_ids(USER_SPEAKER_TAG_TOKEN)

        if system_message is not None:
            pass
        elif mode == "qinyu4o":
            system_message = self.qinyu4o_system_message
        else:
            system_message = self.default_system_message

        num_codebook = self.model.config.num_codebook
        messages = system_message
        texts, speeches = [], []
        speaker_embedding_list = []
        if mode == "qinyu4o":
            assistant_speaker_embedding = self.spk2emb[assistant_speaker]
            speaker_embedding_list.append(assistant_speaker_embedding.to(device_map))

        for i, input_msg in enumerate(input_messages):
            messages = messages + [{
                "role": "user",
                "content": input_msg
            }]

            if USER_SPEAKER_TAG_TOKEN in input_msg:
                user_speaker_embedding = self.audio_processor.process_audio_for_speaker_embedding(audio_paths[i])
                speaker_embedding_list.append(user_speaker_embedding.to(device_map))
                speaker_embeddings = torch.stack(speaker_embedding_list).unsqueeze(0)
            else:
                speaker_embedding_list = None
                speaker_embeddings = None

            if AUDIO_TAG_TOKEN in input_msg:
                audios = audio_paths[:i+1]
            else:
                audios = None

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=self.add_generation_prompt,
            )

            if audios is not None and self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
                input_ids, audios, audio_indices = add_audio_input_contiguous(
                    input_ids, audios, self.tokenizer, self.audio_processor, extra_sensevoice_token=self.extra_sensevoice_token
                )
            else:
                audios = None
                audio_indices = None

            input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")
            speaker_mask = (input_ids == SPEAKER_TAG_ID) | (input_ids == USER_SPEAKER_TAG_ID)

            print("input", self.tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)

            all_tokens, text_tokens, audio_tokens, *_ = generate(
                self.tokenizer, self.model, input_ids, audios, audio_indices, None, [], [],
                speaker_embeddings=speaker_embeddings, speaker_mask=speaker_mask
            )

            text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)

            if len(audio_tokens) > 0:
                audio_tokens = audio_tokens.T.clone()
                audio_tokens = torch.stack([torch.roll(at, -i) for i, at in enumerate(audio_tokens)]).T
                audio_tokens = audio_tokens[:-(num_codebook-1)]
                audio_tokens = audio_tokens.reshape(-1)
                audio_tokens[audio_tokens>=CODEBOOK_SIZE] = 0
                tts_speech = self.audio_tokenizer.decode(audio_tokens).cpu()
            else:
                tts_speech = None

            output = text = text.strip().replace(IM_END_TOKEN, "").replace(CONV_END_TOKEN, "")
            texts.append(text)
            speeches.append(tts_speech)
            messages += [{
                "role": "assistant",
                "content": output.strip()
            }]

        return output, speeches, texts

def sts_conv_task():
    audio_paths = [
        "asset/介绍一下上海.wav",
        "asset/发表一个悲伤的演讲.wav",
        "asset/发表一个振奋人心的演讲.wav",
    ]
    output_audio_paths = [
        os.path.join(output_dir, f"conv_{i}.wav") for i in range(len(audio_paths))
    ]
    output_text_paths = [
        os.path.join(output_dir, f"conv_{i}.txt") for i in range(len(audio_paths))
    ]
    messages = [USER_SPEAKER_TAG_TOKEN + AUDIO_TAG_TOKEN] * len(audio_paths)
    output, speeches, texts = s2s_inference.run_infer_multiturn(messages, audio_paths=audio_paths, mode="qinyu4o")
    for text, speech, output_audio_path, output_text_path in zip(texts, speeches, output_audio_paths, output_text_paths):
        torchaudio.save(output_audio_path, speech.unsqueeze(0), 32000, format="wav")
        with open(output_text_path, "w") as f:
            f.write(f"{text}\n")

def sts_sing_task():
    audio_paths = [
        # "asset/唱一首晴天.wav",
        # "asset/唱一首遇见.wav",
        # "asset/唱一首演员.wav",
        "asset/唱一首热带雨林.wav",
    ]
    output_audio_paths = [
        os.path.join(output_dir, f"sing_{i}.wav") for i in range(len(audio_paths))
    ]
    output_text_paths = [
        os.path.join(output_dir, f"sing_{i}.txt") for i in range(len(audio_paths))
    ]
    messages = [USER_SPEAKER_TAG_TOKEN + AUDIO_TAG_TOKEN] * len(audio_paths)
    output, speeches, texts = s2s_inference.run_infer_multiturn(messages, audio_paths=audio_paths, mode="qinyu4o")
    for text, speech, output_audio_path, output_text_path in zip(texts, speeches, output_audio_paths, output_text_paths):
        torchaudio.save(output_audio_path, speech.unsqueeze(0), 32000, format="wav")
        with open(output_text_path, "w") as f:
            f.write(f"{text}\n")

def sts_roleplay_task():
    audio_paths = [
        "asset/SQS-roleplay/e1c5b71a-5e95-4a7b-ac2b-cb3bca8af446_user_0.wav",
        "asset/SQS-roleplay/e1c5b71a-5e95-4a7b-ac2b-cb3bca8af446_user_1.wav",
        "asset/SQS-roleplay/e1c5b71a-5e95-4a7b-ac2b-cb3bca8af446_user_2.wav",
    ]
    output_audio_paths = [
        os.path.join(output_dir, f"roleplay_{i}.wav") for i in range(len(audio_paths))
    ]
    output_text_paths = [
        os.path.join(output_dir, f"roleplay_{i}.txt") for i in range(len(audio_paths))
    ]

    system_message = [{
        "role": "system",
        "content": (
            "Your Name: QinYu-4o\nYour Origin: TME and Youtu Collaboration.\n"
            "Respond in text-audio interleaved manner.\nYour Voice Embedding: <|speaker|>.\n"
            "Your Role Description: 该角色是一个中年男性， 身份是皇帝近侍, 该角色是太监总管，执行命令迅速果断，忠诚可靠。"
        )
    }]
    assistant_speaker = "2619-王承恩"
    messages = [USER_SPEAKER_TAG_TOKEN + AUDIO_TAG_TOKEN] * len(audio_paths)
    output, speeches, texts = s2s_inference.run_infer_multiturn(
        messages, audio_paths=audio_paths, mode="qinyu4o", assistant_speaker=assistant_speaker, system_message=system_message
    )
    for text, speech, output_audio_path, output_text_path in zip(texts, speeches, output_audio_paths, output_text_paths):
        torchaudio.save(output_audio_path, speech.unsqueeze(0), 32000, format="wav")
        with open(output_text_path, "w") as f:
            f.write(f"{text}\n")

def tts_task():
    texts = [
        "我们将为全球城市的可持续发展贡献力量。",
        "通天河 灵感大王",
        "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",
        "一二三四五六七八九十",
        "One Two Tree Four Five Six Seven Eight Night Ten",
        "123456789",
        "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
        "坡上立着一只鹅，坡下就是一条河。宽宽的河，肥肥的鹅，鹅要过河，河要渡鹅不知是鹅过河，还是河渡鹅?",
        "扁担长，板凳宽，扁担没有板凳宽，板凳没有扁担长。扁担绑在板凳上，板凳不让扁担绑在板凳上。",
        "化肥会挥发，黑化肥发灰，灰化肥发黑。黑化肥发灰会挥发；灰化肥挥发会发黑。黑化肥挥发发灰会花飞；灰化肥挥发发黑会飞花，黑灰化肥会挥发发灰黑讳为花飞；灰黑化肥会挥发发黑灰为讳飞花。",
        "圆桌儿、方桌儿没有腿儿，墨水瓶儿里没有水儿，花瓶里有花儿没有叶儿，练习本儿上写字儿没有准儿，甘蔗好吃净是节儿。西瓜挺大没有味儿，坛儿里的小米儿长了虫儿，鸡毛掸子成了棍儿，水缸沿儿上系围裙儿，耗子打更猫打盹儿，新买的小褂儿没钉扣儿，奶奶想说没有劲儿。",
    ]
    output_audio_paths = [
        os.path.join(output_dir, f"tts_{i}.wav") for i in range(len(texts))
    ]
    output_text_paths = [
        os.path.join(output_dir, f"tts_{i}.txt") for i in range(len(texts))
    ]

    for text, output_audio_path, output_text_path in zip(texts, output_audio_paths, output_text_paths):
        print("=" * 100)
        print("tts_task")
        print(f"{text=}")
        messages = ["Convert the text to speech.\n" + text]
        output, speeches, texts = s2s_inference.run_infer_multiturn(messages, mode=None)
        text, speech = texts[0], speeches[0]
        torchaudio.save(output_audio_path, speech.unsqueeze(0), 32000, format="wav")
        with open(output_text_path, "w") as f:
            f.write(f"{text}\n")


def asr_task():
    audio_paths = [
        "asset/ASR-en/1455-134435-0012.wav",
        "asset/ASR-en/2416-152139-0009.wav",
        "asset/ASR-zh/X0000025373_214273075_S00098.wav",
        "asset/ASR-zh/Y0000018846_m_wTh-GHKoA_S00399.wav",
    ]
    output_text_paths = [
        os.path.join(output_dir, f"asr_{i}.txt") for i in range(len(audio_paths))
    ]
    for audio_path, output_text_path in zip(audio_paths, output_text_paths):
        print("=" * 100)
        print("asr_task")
        print(f"{audio_path=}")
        messages = ["Convert the speech to text.\n<|audio|>"]
        output, speeches, texts = s2s_inference.run_infer_multiturn(
            messages,
            audio_paths=[audio_path],
            mode=None,
        )
        text = texts[0]
        with open(output_text_path, "w") as f:
            f.write(f"{text}\n")

        print(f"{output=}", flush=True)


s2s_inference = S2SInference(
    args.model, audio_tokenizer_path, audio_tokenizer_type, spk2emb_path,
)
tts_task()
asr_task()
sts_conv_task()
#sts_sing_task()
sts_roleplay_task()
