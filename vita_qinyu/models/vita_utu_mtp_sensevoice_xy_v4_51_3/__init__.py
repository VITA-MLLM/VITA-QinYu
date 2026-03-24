from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

from .modeling_utu_v1 import VITAUTUV1ForCausalLM
from .configuration_utu_v1 import UTUV1Config

AutoConfig.register("vita_utu_v1", UTUV1Config)
AutoModelForCausalLM.register(UTUV1Config, VITAUTUV1ForCausalLM)
# AutoTokenizer.register(UTUV1Config, UTUV1Tokenizer)

UTUV1Config.register_for_auto_class()
VITAUTUV1ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
