
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

from .modeling_qwen3 import Qwen3MTPSenseVoiceForCausalLM
from .configuration_qwen3 import Qwen3MTPSenseVoiceConfig

AutoConfig.register("qwen3_sensevoice_xy", Qwen3MTPSenseVoiceConfig)
AutoModelForCausalLM.register(Qwen3MTPSenseVoiceConfig, Qwen3MTPSenseVoiceForCausalLM)
# AutoTokenizer.register(Qwen3MTPSenseVoiceConfig, Qwen3MTPSenseVoiceTokenizer)

Qwen3MTPSenseVoiceConfig.register_for_auto_class()
# Qwen3MTPSenseVoiceModel.register_for_auto_class("AutoModel")
Qwen3MTPSenseVoiceForCausalLM.register_for_auto_class("AutoModelForCausalLM")
