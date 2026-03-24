import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import datasets
import torch
import transformers
from datasets import concatenate_datasets, load_dataset

from .data_collator import collate_fn_deepspeed
from .dataset_qwen3_jsonl_xyspk import Qwen3JsonlXYSpkDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_supervised_dataset_deepspeed(
    model_config,
    model_args,
    data_args,
    training_args,
    tokenizer,
    create_position_ids=True,
    create_loss_mask=False,
    shift_token=False,
):
    logging.info("building dataset...")

    cfg_path = data_args.dataset_name
    max_padding_length = model_args.model_max_length
    output_dir = training_args.output_dir

    create_attention_mask = data_args.create_attention_mask
    create_attention_mask_2d = data_args.create_attention_mask_2d

    image_size = model_args.image_size
    image_token_length = model_args.image_token_length

    max_num_frame = model_args.max_num_frame
    max_fps = model_args.max_fps

    reset_position_ids = data_args.reset_position_ids
    reset_attention_mask = data_args.reset_attention_mask
    variable_length = data_args.variable_length

    min_patch_grid = model_args.min_patch_grid
    max_patch_grid = model_args.max_patch_grid
    high_resolution_type = getattr(model_args, 'vision_high_resolution_type', None)
    normalize_type = getattr(model_args, 'vision_normalize_type', None)

    audio_tokenizer_path = model_args.audio_tokenizer_path
    audio_tokenizer_type = model_args.audio_tokenizer_type
    text_audio_interval_ratio = model_args.text_audio_interval_ratio

    seed = training_args.seed
    cross_dataset_joint = data_args.cross_dataset_joint
    dataset_joint = data_args.dataset_joint

    only_last_conv_label = getattr(data_args, "only_last_conv_label", False)
    use_audio_special_token = getattr(data_args, "use_audio_special_token", True)
    extra_sensevoice_token = getattr(data_args, "extra_sensevoice_token", True)
    speaker_embedding_prob = getattr(data_args, "speaker_embedding_prob", 0.)
    spk2emb = getattr(data_args, "spk2emb", None)
    if getattr(data_args, "dataset_class", None) == "qwen3_jsonl_xyspk":
        TrainDataset = Qwen3JsonlXYSpkDataset
    else:
        raise NotImplementedError

    train_dataset = TrainDataset(
        cfg_path,
        tokenizer,
        image_size=image_size,
        image_token_length=image_token_length,
        max_padding_length=max_padding_length,
        variable_length=variable_length,
        output_dir=output_dir,
        training_args=None,
        shift_token=shift_token,
        create_position_ids=create_position_ids,
        create_attention_mask=create_attention_mask,
        create_attention_mask_2d=create_attention_mask_2d,
        create_loss_mask=create_loss_mask,
        max_num_frame=max_num_frame,
        max_fps=max_fps,
        reset_position_ids=reset_position_ids,
        reset_attention_mask=reset_attention_mask,
        min_patch_grid=min_patch_grid,
        max_patch_grid=max_patch_grid,
        high_resolution_type=high_resolution_type,
        normalize_type=normalize_type,
        seed=seed,
        cross_dataset_joint=cross_dataset_joint,
        dataset_joint=dataset_joint,
        audio_tokenizer_type=audio_tokenizer_type,
        audio_tokenizer_path=audio_tokenizer_path,
        text_audio_interval_ratio=text_audio_interval_ratio,
        use_megatron=False,
        only_last_conv_label=only_last_conv_label,
        use_audio_special_token=use_audio_special_token,
        extra_sensevoice_token=extra_sensevoice_token,
        speaker_embedding_prob=speaker_embedding_prob,
        spk2emb=spk2emb,
    )
    eval_dataset = None

    data_collator = collate_fn_deepspeed

    return dict(train=train_dataset, validation=eval_dataset, data_collator=data_collator)


def build_supervised_dataset_megatron(
    args,
    tokenizer,
    create_position_ids=True,
    create_loss_mask=False,
    shift_token=False,
):
    logging.info("building dataset...")

    assert len(args.data_path) == 1
    cfg_path = args.data_path[0]
    max_padding_length = args.max_padding_length
    audio_max_padding_length = args.audio_max_padding_length
    video_chunk_size = args.video_chunk_size
    output_dir = args.save

    dataset_type = args.dataset_type

    create_attention_mask = args.create_attention_mask_in_dataloader
    create_attention_mask_2d = args.create_attention_mask_in_dataloader
    # create_attention_mask=False
    # create_attention_mask_2d=True

    assert args.img_h == args.img_w
    image_size = args.img_h
    image_token_length = args.image_token_length

    max_num_frame = args.max_num_frame
    max_fps = args.max_fps

    position_embedding_type = args.position_embedding_type
    render_text_to_images = args.render_text_to_images

    reset_position_ids = args.reset_position_ids
    reset_attention_mask = args.reset_attention_mask
    # reset_position_ids=True
    # reset_attention_mask=True

    min_patch_grid = args.min_patch_grid
    max_patch_grid = args.max_patch_grid
    high_resolution_type = args.vision_high_resolution_type
    normalize_type = args.vision_normalize_type

    audio_tokenizer_path = args.audio_tokenizer_path
    audio_tokenizer_type = args.audio_tokenizer_type
    text_audio_interval_ratio = args.text_audio_interval_ratio

    seed = args.seed
    cross_dataset_joint = args.cross_dataset_joint
    dataset_joint = args.dataset_joint
    skip_samples = args.skip_samples

    if "qwen2" in dataset_type:
        TrainDataset = Qwen2Dataset
    elif "qwen3" in dataset_type:
        TrainDataset = Qwen3Dataset
    elif dataset_type == "mistral":
        raise NotImplementedError
        TrainDataset = MistralDataset
    elif dataset_type == "llama3":
        TrainDataset = Llama3Dataset
    elif "deepseek" in dataset_type:
        TrainDataset = DeepSeekDataset
    elif "pretrain" in dataset_type:
        TrainDataset = PretrainDataset
    else:
        print(f"{dataset_type=}")
        raise NotImplementedError

    train_dataset = TrainDataset(
        cfg_path,
        tokenizer,
        image_size=image_size,
        image_token_length=image_token_length,
        max_padding_length=max_padding_length,
        variable_length=False,
        output_dir=output_dir,
        training_args=None,
        shift_token=shift_token,
        create_position_ids=create_position_ids,
        create_attention_mask=create_attention_mask,
        create_attention_mask_2d=create_attention_mask_2d,
        create_loss_mask=create_loss_mask,
        max_num_frame=max_num_frame,
        max_fps=max_fps,
        reset_position_ids=reset_position_ids,
        reset_attention_mask=reset_attention_mask,
        min_patch_grid=min_patch_grid,
        max_patch_grid=max_patch_grid,
        high_resolution_type=high_resolution_type,
        normalize_type=normalize_type,
        seed=seed,
        cross_dataset_joint=cross_dataset_joint,
        dataset_joint=dataset_joint,
        audio_tokenizer_type=audio_tokenizer_type,
        audio_tokenizer_path=audio_tokenizer_path,
        audio_max_padding_length=audio_max_padding_length,
        video_chunk_size=video_chunk_size,
        text_audio_interval_ratio=text_audio_interval_ratio,
        use_megatron=True,
        skip_samples=skip_samples,
        position_embedding_type=position_embedding_type,
        render_text_to_images=render_text_to_images,
    )
    eval_dataset = None

    return train_dataset, None, None
