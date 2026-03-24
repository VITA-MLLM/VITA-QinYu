#!/usr/bin/env python
# coding=utf-8

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
torch.multiprocessing.set_start_method('spawn', force=True)

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    # Trainer,
    # TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers import AutoModel, AutoConfig
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from trainer_v4_51_3 import Trainer


import vita_qinyu.models
from vita_qinyu import build_supervised_dataset_deepspeed
from vita_qinyu.tokenizer import update_tokenizer, get_audio_tokenizer
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# logger = logging.getLogger(__name__)
from loguru import logger


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    attn_implementation: Optional[str] = field(default=None, metadata={"help": ""})

    audio_model_name_or_path: str = field(default=None, metadata={"help": ""})
    audio_tokenizer_path: list[str] = field(default=None, metadata={"help": ""})
    audio_tokenizer_type: str = field(default=None, metadata={"help": ""})
    text_audio_interval_ratio: list[int] = field(default=None, metadata={"help": ""})
    audio_model_freeze: bool = field(default=False, metadata={"help": ""})

    vision_model_name_or_path: str = field(default=None, metadata={"help": ""})
    vision_model_type: Optional[str] = field(default=None, metadata={"help": ""})
    vision_model_freeze: bool = field(default=False, metadata={"help": ""})

    language_model_freeze: bool = field(default=False, metadata={"help": ""})

    vision_projector_type: str = field(default="mlp", metadata={"help": ""})
    vision_projector_pre_norm: bool = field(default=False, metadata={"help": ""})
    vision_downsample_ratio: float = field(default=0.5, metadata={"help": ""})

    image_size: int = field(default=448, metadata={"help": ""})
    image_token_length: int = field(default=1025, metadata={"help": ""})
    max_num_frame: int = field(default=16, metadata={"help": ""})
    max_fps: int = field(default=1, metadata={"help": ""})
    min_patch_grid: int = field(default=1, metadata={"help": ""})
    max_patch_grid: int = field(default=12, metadata={"help": ""})
    vision_process_type: str = field(default="dynamic", metadata={"help": ""})
    vision_normalize_type: str = field(default="imagenet", metadata={"help": ""})

    model_max_length: int = field(default=4096, metadata={"help": ""})

    A: int = field(default=0, metadata={"help": ""})
    B: str = field(default=None, metadata={"help": ""})
    C: bool = field(default=False, metadata={"help": ""})

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    create_attention_mask: bool = field(default=True, metadata={"help": "create_attention_mask"})
    create_attention_mask_2d: bool = field(default=False, metadata={"help": "create_attention_mask_2d"})
    reset_position_ids: bool = field(default=False, metadata={"help": ""})
    reset_attention_mask: bool = field(default=False, metadata={"help": ""})
    cross_dataset_joint: bool = field(default=False, metadata={"help": ""})

    # dataset_joint: bool = field(default=False, metadata={"help": ""})
    # variable_length: bool = field(default=True, metadata={"help": ""})
    dataset_joint: bool = field(default=True, metadata={"help": ""})
    variable_length: bool = field(default=False, metadata={"help": ""})

    D: int = field(default=0, metadata={"help": ""})
    E: str = field(default=None, metadata={"help": ""})
    F: bool = field(default=False, metadata={"help": ""})

    only_last_conv_label: bool = field(default=False, metadata={"help": ""})
    use_audio_special_token: bool = field(default=True, metadata={"help": ""})
    extra_sensevoice_token: bool = field(default=True, metadata={"help": ""})
    dataset_class: Optional[str] = field(default=None, metadata={"help": ""})
    speaker_embedding_prob: float = field(default=0., metadata={"help": ""})
    spk2emb: Optional[str] = field(default=None, metadata={"help": ""})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    vision_model_lr_mult: float = field(default=1.0, metadata={"help": ""})
    vision_model_lr_decay_rate: float = field(default=1.0, metadata={"help": ""})

    mtp_model_lr_mult: float = field(default=1.0, metadata={"help": ""})


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    if isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

    if isinstance(m, torch.nn.RMSNorm):
        torch.nn.init.ones_(m.weight)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # test_npu(training_args)
    # torch_npu.npu.set_device("npu")
    training_args.gradient_checkpointing_kwargs = {'use_reentrant': False}

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    logger.info(f"{model_args=}")

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "model_max_length": model_args.model_max_length,
        "attn_implementation": model_args.attn_implementation,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        config.audio_model_name_or_path = model_args.audio_model_name_or_path
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        config.audio_model_name_or_path = model_args.audio_model_name_or_path
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
    config.use_cache = False
    logger.info(f"{config.__class__.__name__=} {config=}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.pad_token_id is None:
        if tokenizer.pad_id is not None:
            tokenizer.pad_token_id = tokenizer.pad_id
        else:
            tokenizer.pad_token_id = tokenizer.eod_id

    logger.info(f"{tokenizer.__class__.__name__=} {len(tokenizer)=}")
    if "HYTokenizer" in tokenizer.__class__.__name__:
        pass
    else:
        tokenizer = update_tokenizer(tokenizer, model_args.audio_tokenizer_type)
    logger.info(f"{tokenizer.__class__.__name__=} {len(tokenizer)=}")
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            output_loading_info=True
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    if model_args.text_audio_interval_ratio is not None:
        model.generation_config.mtp_inference_mode = model_args.text_audio_interval_ratio

    if model_args.vision_model_name_or_path:
        logger.info(f"Loading {model_args.vision_model_name_or_path}")
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        vision_config = AutoConfig.from_pretrained(
            model_args.vision_model_name_or_path,
            trust_remote_code=True,
        )
        vision_config.attention_dropout = 0.0
        vision_config.drop_path_rate = 0.0
        vision_model = AutoModel.from_pretrained(
            model_args.vision_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        # missing_keys, unexpected_keys = model.model.vision_model.load_state_dict(vision_model.state_dict())
        # logger.info(f"missing_keys {missing_keys}")
        # logger.info(f"unexpected_keys {unexpected_keys}")
        # import time
        # time.sleep(torch.distributed.get_rank() * 10)
        # print(vision_model.state_dict())
        # print(model.model.vision_model.state_dict())
        model.model.vision_model = vision_model

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
        new_embedding_size = model.get_input_embeddings().weight.shape[0]
        logger.info(f"{new_embedding_size=}")
    else:
        new_embedding_size = embedding_size

    if model_args.vision_model_freeze:
        model.model.vision_model.requires_grad_(False)
        for name, param in model.named_parameters():
            if ".vision_model." in name:
                param.requires_grad = False
                param.requires_grad_(False)
                logger.info(f"=> set param {name} {param.size()} requires_grad to False.")
            else:
                pass

    if model_args.audio_model_freeze:
        logger.info(f"=> Loading audio model weight...")
        from funasr.train_utils.load_pretrained_model import load_pretrained_model
        load_pretrained_model(
            model=model.model.audio_model.model,
            path=f"{model_args.audio_model_name_or_path}/model.pt",
            ignore_init_mismatch=True,
            oss_bucket=None,
            scope_map=[],
            excludes=None,
        )
        model.model.audio_model.requires_grad_(False)
        for name, param in model.named_parameters():
            if ".audio_model." in name:
                param.requires_grad = False
                param.requires_grad_(False)
                logger.info(f"=> set param {name} {param.size()} requires_grad to False.")
            else:
                pass

    if model_args.language_model_freeze:
        for name, param in model.named_parameters():
            if ".vision_model." in name:
                pass
            elif ".vision_projection." in name:
                pass
            elif ".audio_model." in name:
                pass
            elif ".audio_projection." in name:
                pass
            elif ".speaker_projection." in name:
                pass
            else:
                param.requires_grad = False
                param.requires_grad_(False)
                # logger.info(f"=> set param {name} {param.size()} requires_grad to False.")

        # Unfreeze the new token embeddings
        for idx in range(embedding_size, new_embedding_size):
            model.get_input_embeddings().weight.data[idx].requires_grad = True
            model.get_input_embeddings().weight.data[idx].requires_grad_(True)
            # logger.info(f"=> set input_embedding {idx} requires_grad to True.")

    if getattr(model.config, "model_type", None) == "mixtral" and deepspeed.is_deepspeed_zero3_enabled():
        require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

        model.config.use_cache = False
        if 'output_router_logits' in model.config.to_dict():
            logger.info('set output_router_logits as True')
            model.config.output_router_logits = True
            model.config.router_aux_loss_coef = 0.001

    if training_args.gradient_checkpointing and not model_args.language_model_freeze:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad = True
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.gradient_checkpointing:
        if hasattr(model, "vision_model"):
            model.model.vision_model.gradient_checkpointing = True
            model.model.vision_model.encoder.gradient_checkpointing = True
        model.model._set_gradient_checkpointing()

        logger.info(f"Enabling gradient checkpointing")

    logger.info(f"model {model}")


    # init new layers
    for missing_key in loading_info["missing_keys"]:
        if "model.audio_projection" in missing_key:
            missing_module = model.model.audio_projection
            logger.info(f"random init weights for {missing_module}")
            missing_module.apply(init_weights)
            break
    for missing_key in loading_info["missing_keys"]:
        if "model.speaker_projection" in missing_key:
            missing_module = model.model.speaker_projection
            logger.info(f"random init weights for {missing_module}")
            missing_module.apply(init_weights)
            break

    # init mtp layers
    if hasattr(model.config, "num_nextn_predict_layers"):
        for mtp_idx in range(model.config.num_nextn_predict_layers):
            layer_idx = model.config.num_hidden_layers - model.config.num_nextn_predict_layers + mtp_idx

            # ---------------------------------------------------------------------------------------
            # layers
            is_missing = False
            for missing_key in loading_info["missing_keys"]:
                if f"model.layers.{layer_idx}" in missing_key:
                    is_missing = True
                    break

            if is_missing:
                logger.info(f"=> load state dict from layer {layer_idx -1 } to layer {layer_idx}")
                model.model.layers[layer_idx].load_state_dict(model.model.layers[layer_idx - 1].state_dict())
                for param1, param2 in zip(model.model.layers[layer_idx].parameters(), model.model.layers[layer_idx - 1].parameters()):
                    assert torch.equal(param1, param2), "权重复制失败"

            for name, param in model.model.layers[layer_idx].named_parameters():
                param.requires_grad = True
                param.requires_grad_(True)

            # ---------------------------------------------------------------------------------------
            # mtp_projs
            is_missing = False
            for missing_key in loading_info["missing_keys"]:
                if f"mtp_projs.{mtp_idx}" in missing_key:
                    is_missing = True
                    break

            if is_missing and mtp_idx > 0:
                logger.info(f"=> load state dict from mtp_proj {mtp_idx -1 } to mtp_proj {mtp_idx}")
                model.mtp_projs[mtp_idx].load_state_dict(model.mtp_projs[mtp_idx - 1].state_dict())
                for param1, param2 in zip(model.mtp_projs[mtp_idx].parameters(), model.mtp_projs[mtp_idx - 1].parameters()):
                    assert torch.equal(param1, param2), "权重复制失败"

            model.mtp_projs[mtp_idx].requires_grad = True
            model.mtp_projs[mtp_idx].requires_grad_(True)

            # ---------------------------------------------------------------------------------------
            # mtp_embed_norms
            is_missing = False
            for missing_key in loading_info["missing_keys"]:
                if f"mtp_embed_norms.{mtp_idx}" in missing_key:
                    is_missing = True
                    break
            if is_missing and mtp_idx == 0: # force initialize to 1.
                model.mtp_embed_norms[mtp_idx].weight.data.fill_(1.)

            if is_missing and mtp_idx > 0:
                logger.info(f"=> load state dict from mtp_embed_norm {mtp_idx -1 } to mtp_embed_norm {mtp_idx}")
                model.mtp_embed_norms[mtp_idx].load_state_dict(model.mtp_embed_norms[mtp_idx - 1].state_dict())
                for param1, param2 in zip(model.mtp_embed_norms[mtp_idx].parameters(), model.mtp_embed_norms[mtp_idx - 1].parameters()):
                    assert torch.equal(param1, param2), "权重复制失败"

            model.mtp_embed_norms[mtp_idx].requires_grad = True
            model.mtp_embed_norms[mtp_idx].requires_grad_(True)

            # ---------------------------------------------------------------------------------------
            # mtp_hidden_norms
            is_missing = False
            for missing_key in loading_info["missing_keys"]:
                if f"mtp_hidden_norms.{mtp_idx}" in missing_key:
                    is_missing = True
                    break

            if is_missing and mtp_idx == 0: # force initialize to 1.
                model.mtp_hidden_norms[mtp_idx].weight.data.fill_(1.)

            if is_missing and mtp_idx > 0:
                logger.info(f"=> load state dict from mtp_hidden_norm {mtp_idx -1 } to mtp_hidden_norm {mtp_idx}")
                model.mtp_hidden_norms[mtp_idx].load_state_dict(model.mtp_hidden_norms[mtp_idx - 1].state_dict())
                for param1, param2 in zip(model.mtp_hidden_norms[mtp_idx].parameters(), model.mtp_hidden_norms[mtp_idx - 1].parameters()):
                    assert torch.equal(param1, param2), "权重复制失败"

            model.mtp_hidden_norms[mtp_idx].requires_grad = True
            model.mtp_hidden_norms[mtp_idx].requires_grad_(True)


    def print_grad_status(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"=> parameter {name} requires_grad is True.")
    # print_grad_status(model)

    # Load data
    lm_datasets = build_supervised_dataset_deepspeed(
        model_config=config,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )
    tokenized_datasets = lm_datasets

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    if "data_collator" in lm_datasets:
        default_data_collator = lm_datasets["data_collator"]

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
