import itertools
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import default_collate
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def collate_fn_deepspeed_old(batch):

    keys = list(set().union(*[set(x.keys()) for x in batch]))
    tmp_batch = [{} for _ in range(len(batch))]
    if "cu_seq_lens" in batch[0]:
        cu_seq_lens = [x["cu_seq_lens"] for x in batch]
    else:
        cu_seq_lens = None

    for k in keys:
        if "images" in k or k == "image_indices":
            for x, y in zip(tmp_batch, batch):
                if k in y:
                    x[k] = y.pop(k)
                    # print("x[image_indices]", x["image_indices"].size())

    new_batch = default_collate(batch)

    for k in keys:
        if "images" in k or k == "image_indices":
            cat_dim = 0 if k != "image_indices" else 1
            if k == "image_indices":
                cnt = 0
                for sample in tmp_batch:
                    if k in sample:
                        sample[k][0] = cnt
                    cnt += 1
            new_batch[k] = torch.cat([x[k] for x in tmp_batch if k in x], dim=cat_dim)
    # print("new_batch[image_indices]", new_batch["image_indices"].size())

    if cu_seq_lens is not None:
        seq_len = cu_seq_lens[0][-1]
        cu_seq_lens = [elem + i * seq_len for i, elem in enumerate(cu_seq_lens)]
        new_batch["cu_seq_lens"] = torch.cat(cu_seq_lens)

    return new_batch


def collate_fn_deepspeed(batch):

    keys = list(set().union(*[set(x.keys()) for x in batch]))

    tmp_batch = [{} for _ in range(len(batch))]
    if "cu_seq_lens" in batch[0]:
        cu_seq_lens = [x["cu_seq_lens"] for x in batch]
        max_seq_len = [x["max_seq_len"] for x in batch]

    else:
        cu_seq_lens = None
        max_seq_len = None

    if "images" in batch[0].keys():
        for tmp_x, x in zip(tmp_batch, batch):
            tmp_x["images"] = x.pop("images")
            tmp_x["image_indices"] = x.pop("image_indices")

    if "audios" in batch[0].keys():
        for tmp_x, x in zip(tmp_batch, batch):
            tmp_x["audios"] = x.pop("audios")
            tmp_x["audio_indices"] = x.pop("audio_indices")

    new_batch = default_collate(batch)

    if "images" in tmp_batch[0].keys():
        new_batch["images"] = torch.cat([x["images"] for x in tmp_batch], dim=0)

        for sample_idx, sample in enumerate(tmp_batch):
            sample["image_indices"][0, :, :] = sample_idx

        new_batch["image_indices"] = torch.cat([x["image_indices"] for x in tmp_batch], dim=1)

    if "audios" in tmp_batch[0].keys():
        new_batch["audios"] = list(itertools.chain.from_iterable([x["audios"] for x in tmp_batch]))
        # print(f"{[x.size() for x in sample['audios']]}")

        for sample_idx, sample in enumerate(tmp_batch):
            for j in range(len(sample["audio_indices"])):
                sample["audio_indices"][j][0, :, :] = sample_idx

        new_batch["audio_indices"] = list(
            itertools.chain.from_iterable([x["audio_indices"] for x in tmp_batch])
        )
        # print(f"{[x.size() for x in sample['audio_indices']]}")

    if cu_seq_lens is not None:
        seq_len = cu_seq_lens[0][-1]
        cu_seq_lens = [elem + i * seq_len for i, elem in enumerate(cu_seq_lens)]
        new_batch["cu_seq_lens"] = torch.cat(cu_seq_lens)
        new_batch["max_seq_len"] = max(max_seq_len)

    return new_batch
