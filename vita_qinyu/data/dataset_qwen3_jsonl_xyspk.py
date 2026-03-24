import json
import logging
import math
import os
import pdb
import random
import re
import sys
import time
import copy
import traceback
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother


from .dataset_base_jsonl import BaseDataset
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from ..constants import (
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
    IMG_TAG_TOKEN,
    VID_TAG_TOKEN,
    AUD_TAG_TOKEN,
    AUD_CONTEXT_TOKEN,
    CONV_END_TOKEN,
    SPEAKER_TAG_TOKEN,
    USER_SPEAKER_TAG_TOKEN,
)


class Qwen3JsonlXYSpkDataset(BaseDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.default_system_message = "You are a helpful AI assistant."
        self.default_system_message = None

        spk2emb = kwargs.get('spk2emb', None)
        self.spk2emb_path = spk2emb

        self.spk2emb = torch.load(spk2emb) if spk2emb is not None else None
        self.ret = defaultdict(dict)
        self.is_cat = True

        if self.cross_dataset_joint:
            for i in range(2):
                self.maybe_init_ret(f"default_{i}")

        self.offsets = {}
        self.num_samples_list = None
        self.source2speaker = {}
        self.load_data()


    def __len__(self):
        return sum([len(v) for k, v in self.offsets.items()])

    def get_source_index(self, idx):
        if self.num_samples_list is None:
            self.num_samples_list = [len(v) for k, v in self.offsets.items()]
        for source_idx, ns in enumerate(self.num_samples_list):
            residual = idx - ns
            if residual < 0:
                break
            idx = residual
        return source_idx, idx

    def load_data(self):
        source_idx = 0
        jsonl_idx = 0
        for data_name, data_info in self.cfg["dataset"].items():
            speaker = data_info.get("speaker", None)
            data_ratio = data_info.get("ratio", 1)
            data_num = data_info.get("num", 999999999)
            prefix_path = data_info.get("prefix_path", "")
            if data_ratio == 0:
                continue

            if data_num == 0:
                continue
            for data_idx, data_path in enumerate(data_info["json_paths"]):

                if not os.path.isfile(data_path) and not os.path.isdir(data_path):
                    logger.warning(f"Data file no found {data_path}")
                    continue

                if '_tmp.jsonl' in data_path:
                    ele_path = data_path.replace('_tmp.jsonl', '.ele')
                    if not os.path.isfile(ele_path):
                        ele_path = data_path.replace('.jsonl', '.offset')
                else:
                    ele_path = data_path.replace('.jsonl', '.offset')

                if not (ele_path.endswith('.ele') or ele_path.endswith('.offset')):
                    logger.warning(f"Data file no found {ele_path}")
                    continue

                if not os.path.isfile(ele_path):
                    logger.warning(f"Data file no found {ele_path}")
                    continue
                try:
                    offsets = self.load_ele(data_name, source_idx, ele_path, data_ratio, data_num)
                except:
                    raise ValueError(f"error loading {data_name} {source_idx} {ele_path}")
                self.offsets[source_idx] = offsets
                self.source2jsonpath[source_idx] = data_path
                self.source2prefixpath[source_idx] = prefix_path
                self.source2speaker[source_idx] = speaker

                source_idx += 1


    def load_ele(self, data_name, source_idx, ele_path, data_ratio, data_num):
        offsets = []
        with open(ele_path) as f:
            line_iterator = tqdm(f, desc=f'loading {data_name} source_idx {source_idx}') if self.is_main_process() else f
            for line in line_iterator:
                offset = int(line.strip())
                offsets.append(offset)

        total_num = len(offsets)
        used_num = min(int(total_num * data_ratio), data_num)

        # self.sample_num = int(len(self.offsets) * self.repeat)
        sample_num = len(offsets)
        # self.instance_offset_ = np.random.choice(self.offsets, self.sample_num)
        instance_offset_ = np.random.choice(offsets, used_num)
        return instance_offset_

    def is_main_process(self, local=True):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            if local:
                rank = int(os.environ["LOCAL_RANK"])
            else:
                rank = torch.distributed.get_rank()
            _is_main_process = rank == 0
            return _is_main_process
        return True





    def maybe_init_ret(self, source, force=False):
        if source not in self.ret or force:
            self.ret[source] = {}

            self.ret[source]["tokens"] = []
            self.ret[source]["labels"] = []
            self.ret[source]["actual_seq_len"] = []

            if self.create_position_ids:
                self.ret[source]["position_ids"] = []

            if self.create_attention_mask:
                self.ret[source]["attention_mask"] = []

            if self.create_attention_mask_2d:
                self.ret[source]["attention_mask_2d"] = torch.tril(
                    torch.ones(
                        (1, self.max_padding_length, self.max_padding_length), dtype=torch.bool
                    )
                )

            self.ret[source]["audio_labels"] = []
            self.ret[source]["audio_tokens"] = []
            self.ret[source]["audio_token_starts"] = []
            self.ret[source]["audio_token_ends"] = []


        return len(self.ret[source]["tokens"]) == 0

    def get_max_min_ret_length(self):
        max_ret_lengh = 0
        min_ret_lengh = self.max_padding_length + 1

        max_ret_key = None
        min_ret_key = None

        for k, v in self.ret.items():
            cur_length = len(v["tokens"])

            if cur_length > max_ret_lengh:
                max_ret_lengh = cur_length
                max_ret_key = k

            if cur_length < min_ret_lengh:
                min_ret_lengh = cur_length
                min_ret_key = k

        return max_ret_lengh, max_ret_key, min_ret_lengh, min_ret_key

    def add_ret(self, ret, source):
        # import pdb; pdb.set_trace()
        cur_length = len(ret["input_ids"])
        cur_image_length = len(ret["images"])
        cur_audio_length = len(ret["audios"])

        all_length = len(self.ret[source]["tokens"])

        if "images" in self.ret[source]:
            all_image_length = len(self.ret[source]["images"])
        else:
            all_image_length = 0

        if cur_image_length > 0:
            if all_image_length > 0:
                self.ret[source]["images"] = torch.cat(
                    [self.ret[source]["images"], ret["images"]], dim=0
                )
                ret["image_indices"][1, :, :] += all_length
                self.ret[source]["image_indices"] = torch.cat(
                    [self.ret[source]["image_indices"], ret["image_indices"]], dim=1
                )
            else:
                self.ret[source]["images"] = ret["images"]
                self.ret[source]["image_indices"] = ret["image_indices"]

        if "audios" in self.ret[source]:
            all_audio_length = len(self.ret[source]["audios"])
        else:
            all_audio_length = 0

        if cur_audio_length > 0:

            if all_audio_length > 0:
                self.ret[source]["audios"].extend(ret["audios"])
                for audio_indice in ret["audio_indices"]:
                    audio_indice[1, :, :] += all_length
                self.ret[source]["audio_indices"].extend(ret["audio_indices"])
            else:
                self.ret[source]["audios"] = ret["audios"]
                for audio_indice in ret["audio_indices"]:
                    audio_indice[1, :, :] += all_length
                self.ret[source]["audio_indices"] = ret["audio_indices"]

            # print(self.ret[source]["audios"])

        if self.create_attention_mask:
            self.ret[source]["attention_mask"] += ret["attention_mask"]

        if self.create_attention_mask_2d:
            self.ret[source]["attention_mask_2d"][:, all_length:, :all_length] = 0

        if self.create_position_ids:
            self.ret[source]["position_ids"] += list(range(cur_length))

        self.ret[source]["tokens"] += ret["input_ids"]
        self.ret[source]["labels"] += ret["labels"]
        self.ret[source]["actual_seq_len"] += [all_length + cur_length]


        # import pdb; pdb.set_trace()
        if "audio_tokens" in self.ret[source] and len(self.ret[source]["audio_tokens"]) > 0:
            all_audio_token_length = len(self.ret[source]["audio_tokens"][0])
        else:
            all_audio_token_length = 0

        cur_audio_token_length = len(ret["audio_tokens"][0]) if len(ret["audio_tokens"]) > 0 else 0

        if cur_audio_token_length > 0:
            if all_audio_token_length > 0:
                # self.ret[source]["audio_tokens"].extend(ret["audio_tokens"])
                # self.ret[source]["audio_labels"].extend(ret["audio_labels"])
                self.ret[source]["audio_tokens"] = concat([self.ret[source]["audio_tokens"], ret["audio_tokens"]])
                self.ret[source]["audio_labels"] = concat([self.ret[source]["audio_labels"], ret["audio_labels"]])

                # for audio_indice in ret["audio_token_starts"]:
                #     audio_indice[1, :, :] += all_length
                # for audio_indice in ret["audio_token_ends"]:
                #     audio_indice[1, :, :] += all_length
                ret["audio_token_starts"] = [x + all_length for x in ret["audio_token_starts"]]
                ret["audio_token_ends"] = [x + all_length for x in ret["audio_token_ends"]]

                self.ret[source]["audio_token_starts"].extend(ret["audio_token_starts"])
                self.ret[source]["audio_token_ends"].extend(ret["audio_token_ends"])
            else:
                self.ret[source]["audio_tokens"] = ret["audio_tokens"]
                self.ret[source]["audio_labels"] = ret["audio_labels"]
                ret["audio_token_starts"] = [x + all_length for x in ret["audio_token_starts"]]
                ret["audio_token_ends"] = [x + all_length for x in ret["audio_token_ends"]]
                # for audio_indice in ret["audio_token_starts"]:
                #     audio_indice[1, :, :] += all_length
                # for audio_indice in ret["audio_token_ends"]:
                #     audio_indice[1, :, :] += all_length
                # self.ret[source]["audio_indices"] = ret["audio_indices"]
                self.ret[source]["audio_token_starts"] = ret["audio_token_starts"]
                self.ret[source]["audio_token_ends"] = ret["audio_token_ends"]

        if "speaker_embeddings" in self.ret[source]:
            all_num_speaker = len(self.ret[source]["speaker_embeddings"])
        else:
            all_num_speaker = 0
        curr_num_speaker = len(ret["speaker_embeddings"])
        if curr_num_speaker > 0:
            if all_num_speaker > 0:
                self.ret[source]["speaker_embeddings"] = torch.cat([
                    self.ret[source]["speaker_embeddings"], ret["speaker_embeddings"]
                ])
            else:
                self.ret[source]["speaker_embeddings"] = ret["speaker_embeddings"]

        if "speaker_mask" in self.ret[source]:
            self.ret[source]["speaker_mask"] += ret["speaker_mask"]
        else:
            self.ret[source]["speaker_mask"] = ret["speaker_mask"]

    def process_ret(self, to_ret):
        if "tokens" in to_ret and len(to_ret["tokens"]) > 0:
            pass
        else:
            return to_ret

        if self.create_position_ids:
            if self.reset_position_ids:
                pass
            else:
                to_ret["position_ids"] = list(range(len(to_ret["tokens"])))

        if self.create_attention_mask_2d:
            if self.reset_attention_mask:
                pass
            else:
                to_ret["attention_mask_2d"] = torch.tril(
                    torch.ones(
                        (1, self.max_padding_length, self.max_padding_length), dtype=torch.bool
                    )
                )

        if self.shift_token:
            to_ret["tokens"] = to_ret["tokens"][:-1]
            to_ret["labels"] = to_ret["labels"][1:]
            to_ret["actual_seq_len"][-1] -= 1
            if self.create_position_ids:
                to_ret["position_ids"] = to_ret["position_ids"][:-1]
            if self.create_attention_mask:
                to_ret["attention_mask"] = to_ret["attention_mask"][:-1]

            if self.create_attention_mask_2d:
                to_ret["attention_mask_2d"][:, :, -1] = 0
                to_ret["attention_mask_2d"][:, -1, :] = 0

        assert len(to_ret["tokens"]) == len(
            to_ret["labels"]
        ), f"{len(to_ret['tokens'])} {len(to_ret['labels'])}"
        if not self.variable_length and self.max_padding_length > len(to_ret["tokens"]):
            to_ret["tokens"] += [self.tokenizer.pad_token_id] * (
                self.max_padding_length - len(to_ret["tokens"])
            )
            to_ret["labels"] += [IGNORE_TOKEN_ID] * (
                self.max_padding_length - len(to_ret["labels"])
            )
            to_ret["actual_seq_len"][-1] = self.max_padding_length
            if self.create_position_ids:
                # to_ret["position_ids"] += to_ret["position_ids"][-1:] * (
                #     self.max_padding_length - len(to_ret["position_ids"])
                # )
                to_ret["position_ids"] += list(
                    range(to_ret["position_ids"][-1] + 1, self.max_padding_length)
                )
            if self.create_attention_mask:
                to_ret["attention_mask"] += [0] * (
                    self.max_padding_length - len(to_ret["attention_mask"])
                )

        to_ret["tokens"] = to_ret["tokens"][: self.max_padding_length]
        to_ret["labels"] = to_ret["labels"][: self.max_padding_length]
        to_ret["actual_seq_len"][-1] = self.max_padding_length
        if self.create_position_ids:
            to_ret["position_ids"] = to_ret["position_ids"][: self.max_padding_length]
        if self.create_attention_mask:
            to_ret["attention_mask"] = to_ret["attention_mask"][: self.max_padding_length]

        to_ret["tokens"] = torch.tensor(to_ret["tokens"], dtype=torch.int64)
        to_ret["labels"] = torch.tensor(to_ret["labels"], dtype=torch.int64)
        to_ret["actual_seq_len"] = torch.tensor(to_ret["actual_seq_len"], dtype=torch.int64)
        if self.create_position_ids:
            to_ret["position_ids"] = torch.tensor(to_ret["position_ids"], dtype=torch.int64)
        if self.create_attention_mask:
            to_ret["attention_mask"] = torch.tensor(to_ret["attention_mask"], dtype=torch.int64)

        if self.create_attention_mask_2d:
            attention_mask_2d = to_ret.pop("attention_mask_2d")
            attention_mask_2d = attention_mask_2d.masked_fill(
                (to_ret["attention_mask"] < 0.5).view(1, 1, self.max_padding_length), value=0
            )
            attention_mask_2d = attention_mask_2d < 0.5

            to_ret["attention_mask"] = attention_mask_2d

        if self.create_loss_mask:
            loss_mask = torch.where(to_ret["labels"] == IGNORE_TOKEN_ID, 0, 1)
            to_ret["loss_mask"] = loss_mask.to(torch.float32)

        if not self.reset_position_ids and not self.reset_attention_mask:
            to_ret.pop("actual_seq_len")

        to_ret["input_ids"] = to_ret["tokens"]

        to_ret["audio_labels"] = torch.tensor(to_ret["audio_labels"], dtype=torch.int64)
        to_ret["audio_tokens"] = torch.tensor(to_ret["audio_tokens"], dtype=torch.int64)

        to_ret["audio_token_starts"] = torch.tensor(to_ret["audio_token_starts"], dtype=torch.int64)
        to_ret["audio_token_ends"] = torch.tensor(to_ret["audio_token_ends"], dtype=torch.int64)

        if "speaker_embeddings" in to_ret:
            # to_ret["speaker_embeddings"] = to_ret["speaker_embeddings"]

            if not self.variable_length and self.max_padding_length > len(to_ret["speaker_mask"]):
                to_ret["speaker_mask"] += [0] * (
                    self.max_padding_length - len(to_ret["speaker_mask"])
                )
            to_ret["speaker_mask"] = torch.tensor(to_ret["speaker_mask"], dtype=torch.int64)
            to_ret["speaker_mask"] = to_ret["speaker_mask"][: self.max_padding_length]

        # print(f'{to_ret["audio_token_starts"]=} {to_ret["audio_token_starts"].shape}')
        # print("to_ret[tokens]", to_ret["tokens"])
        # print("to_ret[labels]", to_ret["labels"])

        to_ret['first_audio_token_id'] = torch.tensor(
            self.tokenizer.convert_tokens_to_ids('<|audio_1_0|>'),
            dtype=torch.int64
        )

        return to_ret

    def is_skip(self):
        if self.processed_samples < self.skip_samples:
            if self.processed_samples % 1e3 == 0:
                print(
                    f"processed_samples {self.processed_samples} skip_samples {self.skip_samples}"
                )
            return True

    def show_statistic(self):
        log_interval = 10000
        if self.max_padding_length >= 2**17:
            log_interval = 500
        if self.max_padding_length >= 2**20:
            log_interval = 100

        if self.unjoint_samples % log_interval == 0:
            print(
                f"processed_samples {self.processed_samples} unjoint_samples {self.unjoint_samples} joint_samples {self.joint_samples} {[len(v['tokens']) for _, v in self.ret.items()]}",
                flush=True,
            )

        return False

    def load_sample(self, index):
        source_idx, idx = self.get_source_index(index)
        offset = self.offsets[source_idx][idx]
        data_path = self.source2jsonpath[source_idx]
        try:
            with open(data_path, 'r') as f:
                f.seek(offset)
                data = json.loads(f.readline().strip())
        except Exception as e:
            print(data_path)
            raise ValueError(f'error load jsonl {data_path} {offset}')

        data['source'] = source_idx
        if 'speaker' not in data:
            data['speaker'] = self.source2speaker[source_idx]

        return data

    def __getitem__(self, index):
        self.processor["audio"].load_model()
        while True:
            try:
                # sample = self.raw_data[index]
                sample = self.load_sample(index)
                # sample = copy.deepcopy(sample)
                sample = self.update_data_path(sample)
                # source = sample["source"]
                # print('sample["source"]', sample["source"])
                # import pdb; pdb.set_trace()
                # len0 = len(self.ret[0]['tokens']) if 0 in self.ret else 0
                # len1 = len(self.ret[1]['tokens']) if 1 in self.ret else 0
                # print(f'0: {len0} 1: {len1}')
                # import pdb; pdb.set_trace()

                self.processed_samples += 1
                if self.is_skip():
                    return {}

                # sample = self.raw_data[index]
                if self.cross_dataset_joint:
                    is_empty = False
                    (
                        max_ret_lengh,
                        max_ret_key,
                        min_ret_lengh,
                        min_ret_key,
                    ) = self.get_max_min_ret_length()
                else:
                    source = sample["source"]
                    is_empty = self.maybe_init_ret(source)

                    max_ret_lengh = min_ret_lengh = len(self.ret[source]["tokens"])
                    max_ret_key = min_ret_key = source

                is_begin = is_empty or self.reset_position_ids or self.reset_attention_mask

                ret = preprocess(
                    sample,
                    self.tokenizer,
                    self.image_token_length,
                    default_system_message=self.default_system_message,
                    processor=self.processor,
                    is_begin=is_begin,
                    max_num_frame=self.max_num_frame,
                    max_fps=self.max_fps,
                    only_last_conv_label=self.only_last_conv_label,
                    use_audio_special_token=self.use_audio_special_token,
                    extra_sensevoice_token=self.extra_sensevoice_token,
                    speaker_embedding_prob=self.speaker_embedding_prob,
                    spk2emb=self.spk2emb,
                )

                if ret is None:
                    return {}

                cur_length = len(ret["input_ids"])

                if cur_length > self.max_padding_length:
                    return {}

                self.unjoint_samples += 1

                if not self.dataset_joint:
                    to_ret = self.ret.pop(max_ret_key)

                    self.maybe_init_ret(max_ret_key, force=True)
                    self.add_ret(ret, max_ret_key)

                elif min_ret_lengh + cur_length > self.max_padding_length:
                    to_ret = self.ret.pop(max_ret_key)
                    self.joint_samples += 1

                    self.maybe_init_ret(max_ret_key, force=True)
                    self.add_ret(ret, max_ret_key)

                else:
                    to_ret = {}
                    self.add_ret(ret, min_ret_key)

                to_ret = self.process_ret(to_ret)

                self.show_statistic()
                # if len(to_ret) > 0:
                #     import pdb; pdb.set_trace()
                return to_ret

            except Exception as error:
                try:
                    with open(os.path.join(self.output_dir, "data_error.log"), "a") as f:
                        print("-" * 100, file=f)
                        print(traceback.format_exc(), file=f)
                        # print(self.raw_data[index], file=f)
                        print(self.load_sample(index), file=f)
                except Exception as error:
                    print(error)
                return {}


def preprocess(
    sample,
    tokenizer: transformers.PreTrainedTokenizer,
    image_token_length: int,
    default_system_message: str = "You are a helpful assistant.",
    processor=None,
    is_begin: bool = True,
    max_num_frame: int = 8,
    max_fps: int = 1,
    only_last_conv_label = False,
    audio_pad_pattern = [0,1,2,3,4,5,6,7],
    use_audio_special_token = True,
    extra_sensevoice_token = True,
    speaker_embedding_prob = 0.,
    spk2emb = None,
) -> Dict:
    # <|im_start|>system
    # You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # Hello, how are you?<|im_end|>
    # <|im_start|>assistantI'm doing great. How can I help you today?<|im_end|>
    # <|im_start|>user
    # I'd like to show off how chat templating works!<|im_end|>

    def is_audio_token(x):
        first_audio_token_id = tokenizer.convert_tokens_to_ids('<|audio_0_0|>')
        last_audio_token_id = tokenizer.convert_tokens_to_ids('<|audio_7_7_pad|>')
        return x >= first_audio_token_id and x <= last_audio_token_id

    def has_audio_token(x):
        for x_i in x:
            if is_audio_token(x_i):
                return True
        return False

    def process_audio(audio_processor, audio_paths):
        if type(audio_paths) is list:
            audio_token_list = []
            for ap in audio_paths:
                audio_tokens = audio_processor.process_audio(ap, is_discrete=True)
                audio_token_list += audio_tokens
            return audio_token_list
        return audio_processor.process_audio(audio_paths, is_discrete=True)

    def process_audio_for_speaker_embedding(audio_processor, audio_paths):
        if type(audio_paths) is list:
            return audio_processor.process_audio_for_speaker_embedding(audio_paths[-1])

        return audio_processor.process_audio_for_speaker_embedding(audio_paths)


    # speaker = sample['speaker']
    # if spk2emb is not None and speaker is not None:
    #     given_speaker_embedding = spk2emb.get(speaker, None)
    # else:
    #     given_speaker_embedding = None
    # import pdb; pdb.set_trace()
    speaker = sample['speaker']

    if speaker is not None and spk2emb is not None:
        given_speaker_embedding = spk2emb.get(speaker, None)
    elif 'speaker_wav' in sample:
        given_speaker_embedding = process_audio_for_speaker_embedding(processor["audio"], sample['speaker_wav'])
    else:
        given_speaker_embedding = None
    # import pdb; pdb.set_trace()
    human_roles = ["user", "human"]
    gpt_roles = ["assistant", "gpt"]
    system_roles = ["system", "observation"]

    IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
    IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

    VID_CONTEXT_ID = tokenizer(VID_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    VID_START_ID = tokenizer(VID_START_TOKEN, add_special_tokens=False).input_ids
    VID_END_ID = tokenizer(VID_END_TOKEN, add_special_tokens=False).input_ids

    PATCH_CONTEXT_ID = tokenizer(PATCH_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    PATCH_START_ID = tokenizer(PATCH_START_TOKEN, add_special_tokens=False).input_ids
    PATCH_END_ID = tokenizer(PATCH_END_TOKEN, add_special_tokens=False).input_ids

    AUD_CONTEXT_ID = tokenizer(AUD_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    AUD_START_ID = tokenizer(AUD_START_TOKEN, add_special_tokens=False).input_ids
    AUD_END_ID = tokenizer(AUD_END_TOKEN, add_special_tokens=False).input_ids

    IMG_TAG_ID = tokenizer(IMG_TAG_TOKEN, add_special_tokens=False).input_ids
    VID_TAG_ID = tokenizer(VID_TAG_TOKEN, add_special_tokens=False).input_ids
    AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids

    SPEAKER_TAG_IDS = tokenizer(SPEAKER_TAG_TOKEN, add_special_tokens=False).input_ids
    USER_SPEAKER_TAG_IDS = tokenizer(USER_SPEAKER_TAG_TOKEN, add_special_tokens=False).input_ids

    assert len(IMG_CONTEXT_ID) == 1
    assert len(IMG_START_ID) == 1
    assert len(IMG_END_ID) == 1

    assert len(VID_CONTEXT_ID) == 1
    assert len(VID_START_ID) == 1
    assert len(VID_END_ID) == 1

    assert len(PATCH_CONTEXT_ID) == 1
    assert len(PATCH_START_ID) == 1
    assert len(PATCH_END_ID) == 1

    assert len(SPEAKER_TAG_IDS) == 1

    IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
    IMG_START_ID = IMG_START_ID[0]
    IMG_END_ID = IMG_END_ID[0]

    VID_CONTEXT_ID = VID_CONTEXT_ID[0]
    VID_START_ID = VID_START_ID[0]
    VID_END_ID = VID_END_ID[0]

    PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
    PATCH_START_ID = PATCH_START_ID[0]
    PATCH_END_ID = PATCH_END_ID[0]

    AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
    AUD_START_ID = AUD_START_ID[0]
    AUD_END_ID = AUD_END_ID[0]

    IMG_TAG_ID = IMG_TAG_ID[0]
    VID_TAG_ID = VID_TAG_ID[0]
    AUD_TAG_ID = AUD_TAG_ID[0]

    SPEAKER_TAG_ID = SPEAKER_TAG_IDS[0]
    USER_SPEAKER_TAG_ID = USER_SPEAKER_TAG_IDS[0]

    BOS_ID = tokenizer.bos_token_id
    EOS_ID = tokenizer.eos_token_id

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids
    IM_START_IDS = tokenizer(IM_START, add_special_tokens=False).input_ids
    IM_END_IDS = tokenizer(IM_END, add_special_tokens=False).input_ids
    USER_IDS = tokenizer(USER, add_special_tokens=False).input_ids
    ASSISTANT_IDS = tokenizer(ASSISTANT, add_special_tokens=False).input_ids
    SYSTEM_IDS = tokenizer(SYSTEM, add_special_tokens=False).input_ids


    input_ids, targets = [], []
    images = []
    image_indices = []
    audios = []
    audio_indices = []

    messages = []
    if "conversations" in sample:
        messages = sample["conversations"]
    if len(messages) == 0 and "messages" in sample:
        messages = sample["messages"]

    # ----------------------------------------------------------------
    # add text to TTS
    if True:
        add_text = None
        # add_audio = None
        for j, sentence in enumerate(messages):
            content = sentence["content"]
            role = sentence["role"]
            if role == "user":
                if "Convert the text to speech." in content:
                    add_text = content.replace("Convert the text to speech.\n", "")
                    add_text = add_text.strip()

                # if "Convert the speech to text." in content:
                #     add_audio = sample["audios"][-1]

            if role == "assistant" and add_text is not None:
                sentence["content"] = add_text + content

            # if role == "assistant" and add_audio is not None:
            #     sentence["content"] = content + "\n<audio>"
            #     sample["audios"].append(add_audio)

        for j, sentence in enumerate(messages):
            content = sentence["content"]
            role = sentence["role"]
            if role == "assistant":
                if f"{AUD_TAG_TOKEN}{AUD_TAG_TOKEN}" in content and len(content.replace(AUD_TAG_TOKEN, "")) > 0:
                    # has text in content ("{text}{AUD_TAG_TOKEN}{AUD_TAG_TOKEN}")
                    sentence["content"] = content.replace(
                        f"{AUD_TAG_TOKEN}{AUD_TAG_TOKEN}", f"{CONV_END_TOKEN}{AUD_TAG_TOKEN}{AUD_TAG_TOKEN}"
                    )
                elif AUD_TAG_TOKEN in content and len(content.replace(AUD_TAG_TOKEN, "")) > 0:
                    # has text in content ("{text}{AUD_TAG_TOKEN}")
                    sentence["content"] = content.replace(AUD_TAG_TOKEN, f"{CONV_END_TOKEN}{AUD_TAG_TOKEN}")

        # merge contiguous <|audio|><|audio|> and handle audios accordingly
        new_sample = merge_audio_audio({
            "messages": copy.deepcopy(messages),
            "audios": copy.deepcopy(sample.get("audios", []))
        })
        messages = new_sample["messages"]
        sample["audios"] = new_sample["audios"]

    # ----------------------------------------------------------------
    # system
    has_system = False
    if is_begin:
        if messages[0]["role"] == "system":
            has_system = True
        else:
            has_system = False

        if (
            not has_system
            and default_system_message is not None
            and len(default_system_message) > 0
        ):
            messages = [{"role": "system", "content": default_system_message}] + messages
            has_system = True

    # ----------------------------------------------------------------
    # audio
    max_pad = max(audio_pad_pattern)


    used_audio_labels = []
    used_audio_tokens_by_layer = []
    speaker_embeddings = []
    if has_audio(sample) and processor["audio"].is_discrete:
        unused_audio_idxs = list(range(len(sample["audios"])))
        audio_tokens_list = [
            # processor["audio"].process_audio(x, is_discrete=True) for x in sample["audios"]
            process_audio(processor["audio"], x) for x in sample["audios"]
        ]
        audio_tokens_list = [
            "".join(f"<|audio_{i}|>" if \
            type(i) is not tuple and type(i) is not list \
            else f"<|audio_{i[0]}_{i[1]}|>" for i in x) for x in audio_tokens_list]
        num_codebook = processor["audio"].audio_tokenizer.xy_tokenizer.nq

        audio_tokens_by_layer = [
            [
                ''.join(
                    [f'<|audio_{i}_{p}_pad|>' for p in range(audio_pad_pattern[i])] +
                    re.findall(f'(<\\|audio_{i}_\\d+\\|>)', x) +
                    [f'<|audio_{i}_{p}_pad|>' for p in range(audio_pad_pattern[i], max_pad)]
                )
                for i in range(num_codebook)
            ] for x in audio_tokens_list
        ]
        audio_tokens_list_0 = [
            ''.join(
                re.findall(f'(<\\|audio_0_\\d+\\|>)', x)+[
                f'<|audio_0_{i}_pad|>' for i in range(max_pad)
            ]) for x in audio_tokens_list
        ]
        codebook_size = 1024
        audio_labels = [[
            [codebook_size+p for p in range(audio_pad_pattern[i])] +
            list(map(int, re.findall(f"<\\|audio_{i}_(\\d+)\\|>", x[i]))) +
            [codebook_size+p for p in range(audio_pad_pattern[i], max_pad)]
            for i in range(num_codebook)
        ] for x in audio_tokens_by_layer]

        audio_idx = 0

        # audio_idx2speaker = {}
        for j, sentence in enumerate(messages):
            content = sentence["content"]
            role = sentence["role"]
            # whether apply discrete tokenize to this role
            if processor["audio"].apply_to_role(role, is_discrete=True):
                while AUD_TAG_TOKEN in content:
                    content = content.replace(
                        AUD_TAG_TOKEN,
                        f"{AUD_START_TOKEN}{audio_tokens_list_0[audio_idx]}{AUD_END_TOKEN}",
                        1,
                    )
                    # extract speaker embedding
                    has_speaker_tag = speaker_embedding_prob > 0 and torch.rand(()) < speaker_embedding_prob
                    if has_speaker_tag:
                        audio = sample["audios"][audio_idx]
                        if spk2emb is None and given_speaker_embedding is None: # do not have spk2emb map
                            speaker_embedding = process_audio_for_speaker_embedding(processor["audio"], audio)
                            speaker_embeddings.append(speaker_embedding)
                        elif given_speaker_embedding is not None: # get speaker embedding from spk2emb map
                            # import pdb; pdb.set_trace()
                            speaker_embedding = given_speaker_embedding
                            speaker_embeddings.append(speaker_embedding)
                        else: # speaker is None or not found
                            pass
                        # content = SPEAKER_TAG_TOKEN + content

                    unused_audio_idxs.remove(audio_idx)
                    used_audio_labels.append(audio_labels[audio_idx])
                    used_audio_tokens_by_layer.append(
                        tokenizer(
                            audio_tokens_by_layer[audio_idx],
                            add_special_tokens=False
                        ).input_ids
                    )
                    audio_idx += 1
            else:
                audio_idx += content.count(AUD_TAG_TOKEN)

            sentence["content"] = content

    speaker_sp = f'Your Voice Embedding: {SPEAKER_TAG_TOKEN}'
    if len(speaker_embeddings) > 0:
        speaker_sp = f'Your Voice Embedding: {SPEAKER_TAG_TOKEN}'
        if messages[0]['role'] == 'system' and speaker_sp not in messages[0]['content']:
            messages[0]['content'] = '\n'.join([messages[0]['content'], speaker_sp])
        elif messages[0]['role'] == 'system' and speaker_sp in messages[0]['content']:

            pass
        else:
            messages = [{
                "role": "system",
                "content": speaker_sp
            }] + messages
        speaker_embeddings = [torch.stack(speaker_embeddings).mean(0)]
    # else:
    #     if messages[0]['role'] == 'system' and speaker_sp in messages[0]['content']:
    #         messages[0]['content'] = messages[0]['content'].replace(speaker_sp, '')
        # import pdb; pdb.set_trace()
    # ----------------------------------------------------------------
    # text
    for j, sentence in enumerate(messages):
        role = sentence["role"]
        content = sentence["content"]

        if role in human_roles:
            _input_id = (
                IM_START_IDS
                + USER_IDS
                + nl_tokens
                + tokenizer(content, add_special_tokens=False).input_ids
                + IM_END_IDS
                + nl_tokens
            )
            _target = [IGNORE_TOKEN_ID] * len(_input_id)

        elif role in gpt_roles:
            has_speaker_tag = content.startswith(SPEAKER_TAG_TOKEN)
            if has_speaker_tag:
                content = content[len(SPEAKER_TAG_TOKEN):]
            content_input_id = tokenizer(content, add_special_tokens=False).input_ids

            if processor["audio"].audio_tokenizer is not None:
                content_input_id = processor["audio"].text_audio_interval(
                    content_input_id,
                    AUD_START_ID,
                    AUD_END_ID,
                    use_audio_special_token=use_audio_special_token
                )

            _SPEAKER_TAG_IDS = SPEAKER_TAG_IDS if has_speaker_tag and has_audio_token(content_input_id) else []

            _input_id = (
                IM_START_IDS + ASSISTANT_IDS + nl_tokens + _SPEAKER_TAG_IDS + content_input_id + IM_END_IDS + nl_tokens
            )
            if only_last_conv_label and len(used_audio_labels) > 0: # has audio response
                if j == len(messages) - 1:
                    # only use labels in last turn of conversation and ignore others
                    _target = (
                        [IGNORE_TOKEN_ID] * len(IM_START_IDS)
                        + [IGNORE_TOKEN_ID] * len(ASSISTANT_IDS)
                        + [IGNORE_TOKEN_ID] * len(nl_tokens)
                        + [IGNORE_TOKEN_ID] * len(_SPEAKER_TAG_IDS)
                        + content_input_id
                        + IM_END_IDS
                        + nl_tokens
                    )
                else:
                    _target = [IGNORE_TOKEN_ID] * len(_input_id)
            else:
                _target = (
                    [IGNORE_TOKEN_ID] * len(IM_START_IDS)
                    + [IGNORE_TOKEN_ID] * len(ASSISTANT_IDS)
                    + [IGNORE_TOKEN_ID] * len(nl_tokens)
                    + [IGNORE_TOKEN_ID] * len(_SPEAKER_TAG_IDS)
                    + content_input_id
                    + IM_END_IDS
                    + nl_tokens
                )

        elif role in system_roles:
            _input_id = (
                IM_START_IDS
                + SYSTEM_IDS
                + nl_tokens
                + tokenizer(content, add_special_tokens=False).input_ids
                + IM_END_IDS
                + nl_tokens
            )
            _target = [IGNORE_TOKEN_ID] * len(_input_id)

        else:
            raise NotImplementedError

        # print(f"_input_id {_input_id}")
        input_ids += _input_id
        targets += _target

    # ----------------------------------------------------------------
    # image
    if has_image(sample):
        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]
        assert len(img_positions) == len(sample["images"]), sample

        new_input_ids = []
        new_targets = []
        st = 0
        for img_idx, img_pos in enumerate(img_positions):
            image_patches, (best_width, best_height) = processor[
                "image"
            ].process_images_with_subpatch(sample["images"][img_idx])
            images.append(image_patches)

            new_input_ids += input_ids[st:img_pos]
            new_targets += targets[st:img_pos]

            new_input_ids += [IMG_START_ID]
            new_targets += [IGNORE_TOKEN_ID]

            image_indice_b = torch.zeros(
                1, image_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            image_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            image_indice_b_s = torch.stack(
                [image_indice_b, image_indice_s], dim=0
            )  # 2, num_image, image_length
            image_indices.append(image_indice_b_s)

            new_input_ids += [IMG_CONTEXT_ID] * image_token_length
            new_targets += [IGNORE_TOKEN_ID] * image_token_length

            new_input_ids += [IMG_END_ID]
            new_targets += [IGNORE_TOKEN_ID]

            if len(image_patches) > 1:
                for i in range(0, best_height, processor["image"].patch_size):
                    new_input_ids += nl_tokens
                    new_targets += [IGNORE_TOKEN_ID] * len(nl_tokens)

                    for j in range(0, best_width, processor["image"].patch_size):
                        new_input_ids += [PATCH_START_ID]
                        new_targets += [IGNORE_TOKEN_ID]

                        image_indice_b = torch.zeros(
                            1, image_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        image_indice_s = (
                            torch.arange(
                                len(new_input_ids), len(new_input_ids) + image_token_length
                            )
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        image_indice_b_s = torch.stack(
                            [image_indice_b, image_indice_s], dim=0
                        )  # 2, num_image, image_length
                        image_indices.append(image_indice_b_s)

                        new_input_ids += [PATCH_CONTEXT_ID] * image_token_length
                        new_targets += [IGNORE_TOKEN_ID] * image_token_length

                        new_input_ids += [PATCH_END_ID]
                        new_targets += [IGNORE_TOKEN_ID]

            st = img_pos + 1

        new_input_ids += input_ids[st:]
        new_targets += targets[st:]

        input_ids = new_input_ids
        targets = new_targets

    # ----------------------------------------------------------------
    # video
    if has_video(sample):
        vid_positions = [i for i, x in enumerate(input_ids) if x == VID_TAG_ID]
        assert len(vid_positions) == len(sample["videos"]), sample

        new_input_ids = []
        new_targets = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            video_frames, _ = processor["image"].process_video(
                sample["videos"][vid_idx], max_num_frame, max_fps
            )

            new_input_ids += input_ids[st:vid_pos]
            new_targets += targets[st:vid_pos]

            images.append(video_frames)

            for _ in video_frames:
                new_input_ids += [VID_START_ID]
                new_targets += [IGNORE_TOKEN_ID]

                image_indice_b = torch.zeros(
                    1, image_token_length, dtype=torch.int64
                )  # This will change in collate_fn
                image_indice_s = (
                    torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                    .unsqueeze(0)
                    .repeat(1, 1)
                )
                image_indice_b_s = torch.stack(
                    [image_indice_b, image_indice_s], dim=0
                )  # 2, num_image, image_length
                image_indices.append(image_indice_b_s)

                new_input_ids += [VID_CONTEXT_ID] * image_token_length
                new_targets += [IGNORE_TOKEN_ID] * image_token_length

                new_input_ids += [VID_END_ID]
                new_targets += [IGNORE_TOKEN_ID]

            st = vid_pos + 1

        new_input_ids += input_ids[st:]
        new_targets += targets[st:]

        input_ids = new_input_ids
        targets = new_targets

    # ----------------------------------------------------------------
    # audio
    user_speaker_embeddings = []
    if has_audio(sample) and processor["audio"].is_contiguous:
        aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
        # assert len(aud_positions) == len(sample["audios"]), sample
        assert len(aud_positions) == len(unused_audio_idxs), sample

        new_input_ids = []
        new_targets = []
        st = 0
        for aud_idx, aud_pos in enumerate(aud_positions):
            aud_idx = unused_audio_idxs[aud_idx]
            audio = processor["audio"].process_audio(sample["audios"][aud_idx], is_contiguous=True)
            assert not torch.isnan(audio).any(), f"Data A1 nan: {audio=} {audio.size()=}"
            audios.append(audio)
            audio_token_length = audio.size(0) + (4 if extra_sensevoice_token else 0)

            user_speaker_embedding = process_audio_for_speaker_embedding(processor["audio"], sample["audios"][aud_idx])
            user_speaker_embeddings.append(user_speaker_embedding)

            new_input_ids += input_ids[st:aud_pos]
            new_targets += targets[st:aud_pos]

            new_input_ids += [USER_SPEAKER_TAG_ID, AUD_START_ID]
            new_targets += [IGNORE_TOKEN_ID, IGNORE_TOKEN_ID]

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
            new_targets += [IGNORE_TOKEN_ID] * audio_token_length

            new_input_ids += [AUD_END_ID]
            new_targets += [IGNORE_TOKEN_ID]

            st = aud_pos + 1

        new_input_ids += input_ids[st:]
        new_targets += targets[st:]

        input_ids = new_input_ids
        targets = new_targets

    if len(images) > 0:
        images = torch.cat(images, dim=0)

    if len(image_indices) > 0:
        image_indices = torch.cat(image_indices, dim=1)

    attention_mask = [1] * len(input_ids)


    if use_audio_special_token:
        audio_token_starts = [i+1 for i, x in enumerate(input_ids) if x == AUD_START_ID and is_audio_token(input_ids[i+1])]
        audio_token_ends = [i for i, x in enumerate(input_ids) if x == AUD_END_ID and is_audio_token(input_ids[i-1])]
        assert len(audio_token_starts) == len(audio_token_ends), f'{len(audio_token_starts)=} == {len(audio_token_ends)=}'
    else:
        audio_token_starts = [i+1 for i, x in enumerate(input_ids[:-1]) if (not is_audio_token(input_ids[i])) and is_audio_token(input_ids[i+1])]
        audio_token_ends = [i+1 for i, x in enumerate(input_ids[:-1]) if (not is_audio_token(input_ids[i+1])) and is_audio_token(input_ids[i])]
        assert len(audio_token_starts) == len(audio_token_ends), f'{len(audio_token_starts)=} == {len(audio_token_ends)=}'
    # used_audio_labels = [sum([at[i] for at in used_audio_labels], []) for i in range(num_codebook)]
    # used_audio_tokens_by_layer = [sum([at[i] for at in used_audio_tokens_by_layer], []) for i in range(num_codebook)]
    used_audio_labels = concat(used_audio_labels)
    used_audio_tokens_by_layer = concat(used_audio_tokens_by_layer)

    # import pdb; pdb.set_trace()


    speaker_mask = [1 if iid == SPEAKER_TAG_ID else 0 for iid in input_ids]
    user_speaker_mask = [1 if iid == USER_SPEAKER_TAG_ID else 0 for iid in input_ids]
    # import pdb; pdb.set_trace()
    assert sum(speaker_mask) == len(speaker_embeddings), f"{sum(speaker_mask)=} != {len(speaker_embeddings)=}"
    assert sum(user_speaker_mask) == len(user_speaker_embeddings), f"{sum(user_speaker_mask)=} != {len(user_speaker_embeddings)=}"

    # import pdb; pdb.set_trace()
    speaker_mask, speaker_embeddings = merge_speaker_mask_and_embeddings(user_speaker_mask,  user_speaker_embeddings, speaker_mask, speaker_embeddings)
    if len(speaker_embeddings) > 0:
        speaker_embeddings = torch.stack(speaker_embeddings)
        assert not torch.isnan(speaker_embeddings).any(), f"Data A1 nan: {speaker_embeddings=} {speaker_embeddings.size()=}"
    assert sum(speaker_mask) == len(speaker_embeddings), f"{sum(speaker_mask)=} != {len(speaker_embeddings)=}"
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
        images=images,
        image_indices=image_indices,
        audios=audios,
        audio_indices=audio_indices,
        audio_labels=used_audio_labels, # [(8, T1), (8, T2), ...] int
        audio_tokens=used_audio_tokens_by_layer, # [(8, T1), (8, T2), ...]
        audio_token_starts=audio_token_starts,
        audio_token_ends=audio_token_ends,
        speaker_embeddings=speaker_embeddings,
        speaker_mask=speaker_mask,
    )

def concat(codes): # N x 8 x T
    if len(codes) == 0:
        return []
    num_codebook = len(codes[0])
    concated = [sum([c[i] for c in codes], []) for i in range(num_codebook)]
    return concated

def has_video(sample):
    # video
    if (
        "videos" in sample
        and isinstance(sample["videos"], list)
        and None not in sample["videos"]
        and len(sample["videos"])
    ):
        return True
    return False


def has_image(sample):
    # image
    if (
        "images" in sample
        and isinstance(sample["images"], list)
        and None not in sample["images"]
        and len(sample["images"])
    ):
        return True
    return False


def has_audio(sample):
    # audio
    if (
        "audios" in sample
        and isinstance(sample["audios"], list)
        and None not in sample["audios"]
        and len(sample["audios"])
    ):
        return True
    return False


def has_audio_audio(messages):
    for sent in messages:
        if f'{AUD_TAG_TOKEN}{AUD_TAG_TOKEN}' in sent['content']:
            return True
    return False

def concat_audio_paths(p1, p2):
    if type(p1) is list and type(p2) is list:
        return p1 + p2
    elif type(p1) is list and type(p2) is str:
        return p1 + [p2]
    elif type(p1) is str and type(p2) is list:
        return [p1] + p2
    elif type(p1) is str and type(p2) is str:
        return [p1, p2]
    raise ValueError

def merge_audio_audio_once(sample):
    audio_idx = 0
    audios = []
    for sent in sample['messages']:
        # print(audio_idx)
        if f'{AUD_TAG_TOKEN}{AUD_TAG_TOKEN}' in sent['content']:
            audios.extend(sample['audios'][:audio_idx])
            sent['content'] = sent['content'].replace(f'{AUD_TAG_TOKEN}{AUD_TAG_TOKEN}', AUD_TAG_TOKEN)
            audios.append(concat_audio_paths(sample['audios'][audio_idx], sample['audios'][audio_idx+1]))
            audio_idx += 2
            break
        elif AUD_TAG_TOKEN in sent['content']:
            audio_idx += sent['content'].count(AUD_TAG_TOKEN)
    audios.extend(sample['audios'][audio_idx:])
    sample['audios'] = audios
    return sample

def merge_audio_audio(sample):
    while has_audio_audio(sample['messages']):
        sample = merge_audio_audio_once(sample)
    return sample

def merge_speaker_mask_and_embeddings(
    user_speaker_mask, user_speaker_embeddings, speaker_mask, speaker_embeddings
):
    _speaker_mask = []
    _speaker_embeddings = []
    user_speaker_ptr, speaker_ptr = 0, 0
    assert len(speaker_mask) == len(user_speaker_mask), f"{len(speaker_mask)=} == {len(user_speaker_mask)=}"
    for i in range(len(speaker_mask)):
        if speaker_mask[i] == 1:
            assert user_speaker_mask[i] == 0, f"{i=} {speaker_mask=} {user_speaker_mask=}"
            _speaker_mask.append(1)
            _speaker_embeddings.append(speaker_embeddings[speaker_ptr])
            speaker_ptr += 1
        elif user_speaker_mask[i] == 1:
            assert speaker_mask[i] == 0, f"{i=} {speaker_mask=} {user_speaker_mask=}"
            _speaker_embeddings.append(user_speaker_embeddings[user_speaker_ptr])
            _speaker_mask.append(1)
            user_speaker_ptr += 1
        else:
            _speaker_mask.append(0)
    assert speaker_ptr + user_speaker_ptr == sum(speaker_mask) + sum(user_speaker_mask), \
        f"{speaker_ptr=} + {user_speaker_ptr=} == {sum(speaker_mask)=} + {sum(user_speaker_mask)=}"
    return _speaker_mask, _speaker_embeddings
