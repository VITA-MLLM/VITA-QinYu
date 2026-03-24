import os, sys

# ------------ 最优先执行：Patch huggingface_hub ------------
import huggingface_hub
from huggingface_hub import hf_hub_download  # 导入高版本中存在的替代函数


# 若高版本中无 cached_download，则用 hf_hub_download 包装实现
if not hasattr(huggingface_hub, 'cached_download'):
    def cached_download(url, cache_dir=None, force_download=False, proxies=None, etag_timeout=10, resume_download=False, user_agent=None):
        """用 hf_hub_download 模拟 cached_download 的功能（适配高版本）"""
        # 从 url 中提取 repo_id 和 filename（简单处理，需根据实际 url 格式调整）
        # 示例 url 格式：https://huggingface.co/{repo_id}/resolve/main/{filename}
        import re
        match = re.search(r'https://huggingface\.co/([^/]+)/resolve/([^/]+)/(.+)', url)
        if not match:
            raise ValueError(f"Unsupported url format: {url}")
        repo_id = match.group(1)
        revision = match.group(2)
        filename = match.group(3)

        # 调用 hf_hub_download 实现缓存下载
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            etag_timeout=etag_timeout,
            resume_download=resume_download,
            user_agent=user_agent
        )

    # 注入到 huggingface_hub 模块中
    huggingface_hub.cached_download = cached_download
    print("Successfully patched 'cached_download' using hf_hub_download")

sys.path.append('models')

# 在导入任何模块之前，先 monkey patch transformers
import transformers
import transformers.models.auto.configuration_auto

# ------------ 关键修改1：完善4.48.3的架构映射（适配生成式模型） ------------
# 1. 配置映射：添加qwen3→qwen2/qwen的映射（适配4.48.3的CONFIG_MAPPING）
if 'qwen3' not in transformers.models.auto.configuration_auto.CONFIG_MAPPING:
    # 优先用qwen2配置，无则用qwen，最后用AutoConfig
    if 'qwen2' in transformers.models.auto.configuration_auto.CONFIG_MAPPING:
        transformers.models.auto.configuration_auto.CONFIG_MAPPING['qwen3'] = transformers.models.auto.configuration_auto.CONFIG_MAPPING['qwen2']
    elif 'qwen' in transformers.models.auto.configuration_auto.CONFIG_MAPPING:
        transformers.models.auto.configuration_auto.CONFIG_MAPPING['qwen3'] = transformers.models.auto.configuration_auto.CONFIG_MAPPING['qwen']
    else:
        transformers.models.auto.configuration_auto.CONFIG_MAPPING['qwen3'] = transformers.AutoConfig

# 2. 模型映射：4.48.3中生成式模型用MODEL_FOR_CAUSAL_LM_MAPPING（核心！）
if hasattr(transformers.models.auto.modeling_auto, 'MODEL_FOR_CAUSAL_LM_MAPPING'):
    model_mapping = transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING
    if 'qwen3' not in model_mapping:
        if 'qwen2' in model_mapping:
            model_mapping['qwen3'] = model_mapping['qwen2']
        elif 'qwen' in model_mapping:
            model_mapping['qwen3'] = model_mapping['qwen']
        else:
            model_mapping['qwen3'] = transformers.AutoModelForCausalLM
# 兼容旧版MODEL_MAPPING（兜底）
elif hasattr(transformers.models.auto.modeling_auto, 'MODEL_MAPPING'):
    if 'qwen3' not in transformers.models.auto.modeling_auto.MODEL_MAPPING:
        if 'qwen2' in transformers.models.auto.modeling_auto.MODEL_MAPPING:
            transformers.models.auto.modeling_auto.MODEL_MAPPING['qwen3'] = transformers.models.auto.modeling_auto.MODEL_MAPPING['qwen2']
        else:
            transformers.models.auto.modeling_auto.MODEL_MAPPING['qwen3'] = transformers.AutoModelForCausalLM

# 设置环境变量，强制使用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# 在导入 hyperpyyaml 之前，先 monkey patch pydoc
import pydoc
original_locate = pydoc.locate

def safe_locate(name, forceload=0):
    """安全的 locate 函数，避免 transformers 导入问题"""
    try:
        return original_locate(name, forceload)
    except Exception as e:
        #import ipdb; ipdb.set_trace()
        if 'qwen3' in str(e) or 'transformers' in str(e):
            # 如果是 transformers 相关的问题，返回 None
            print(f"Warning: Skipping import of {name} due to transformers issue")
            return None
        else:
            raise e

pydoc.locate = safe_locate

# 直接 monkey patch AutoModel.from_pretrained
from transformers import AutoModel
original_from_pretrained = AutoModel.from_pretrained

def safe_from_pretrained(*args, **kwargs):
    """安全的 from_pretrained 函数，处理 qwen3 模型"""
    try:
        #import ipdb; ipdb.set_trace()
        return original_from_pretrained(*args, **kwargs)
    except Exception as e:
        import ipdb; ipdb.set_trace()
        if 'qwen3' in str(e):
            print(f"Warning: Using fallback for qwen3 model: {args[0]}")
            # 尝试使用 qwen2 作为 fallback
            model_name = args[0]
            if 'Qwen3' in model_name:
                fallback_name = model_name.replace('Qwen3', 'Qwen2')
                print(f"Trying fallback model: {fallback_name}")
                return original_from_pretrained(fallback_name, *args[1:], **kwargs)
            else:
                raise e
        else:
            raise e

AutoModel.from_pretrained = safe_from_pretrained

from hyperpyyaml import load_hyperpyyaml
import torch
from typing import List, Optional


class VoiceDesignInfer:
    def __init__(self, flow_ckpt: str, config_path: str, Qwen_ckpt: str, device: Optional[str] = None,
                 n_timesteps: int = 20, temperature: float = 1.0):
        # 外部传入的参数
        external_params = { "model_name": Qwen_ckpt }

        with open(config_path, 'r') as f:
            configs = load_hyperpyyaml(f,overrides=external_params)
        self.model = configs['flow']

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.n_timesteps = n_timesteps
        self.temperature = temperature

        self.load_model(flow_ckpt)

    def _load_from_zero_dir(self, ckpt_dir: str):
        try:
            import deepspeed
            # Prefer a utility if available in deepspeed versions
            if hasattr(deepspeed.utils.zero_to_fp32, 'get_fp32_state_dict_from_zero_checkpoint'):
                state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
            elif hasattr(deepspeed.utils.zero_to_fp32, 'load_state_dict_from_zero_checkpoint'):
                state_dict = deepspeed.utils.zero_to_fp32.load_state_dict_from_zero_checkpoint(ckpt_dir)
            else:
                raise RuntimeError('Unsupported deepspeed version for direct ZeRO merge. Use zero_to_fp32.py script to merge first.')
            return state_dict
        except ImportError:
            raise RuntimeError('deepspeed not installed in this environment. Please merge ZeRO checkpoint to a single .pt using zero_to_fp32.py and pass that path.')

    def load_model(self, ckpt_path: str):
        if os.path.isdir(ckpt_path):
            # Load from DeepSpeed ZeRO checkpoint directory
            state_dict = self._load_from_zero_dir(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            # Some checkpoints may wrap state dict under a key
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[Warn] Missing keys: {len(missing)}')
        if unexpected:
            print(f'[Warn] Unexpected keys: {len(unexpected)}')
        self.model.to(self.device)
        self.model.eval()
        print(f'Loaded model from {ckpt_path}')

    @torch.inference_mode()
    def infer(self, input_text_list: List[str], save_path: Optional[str] = None):
        spk_embedding = self.model.inference(
            text_descriptions=input_text_list,
            n_timesteps=self.n_timesteps,
            temperature=self.temperature
        )
        for i in range(len(input_text_list)):
            text = input_text_list[i]

            if len(text.split('|')) == 4: # 兜底
                #男|青年|性格直率、果断，忠于宗门，办事严谨，沉稳有责任心|玄青道宗核心弟子
                role_desc_list = text.split('|')
                if role_desc_list[0] == '未知':
                    role_desc_list[0]='男'
                if role_desc_list[1] == '未知':
                    role_desc_list[1] = '青年'
                if role_desc_list[2] == '未知':
                    role_desc_list[2] ='普通人'
                if role_desc_list[3] == '未知':
                    role_desc_list[3] = '积极向上'
                text = f'该角色是一个{role_desc_list[1]}{role_desc_list[0]}性，身份是{role_desc_list[3]}, {role_desc_list[2]}'

        if save_path is not None:
            save_dict = {}
            for i, text in enumerate(input_text_list):
                print(f'Input text {i+1}: {text}')
                save_dict[i] = {
                    'text': text,
                    'spk_embedding': spk_embedding[i].detach().cpu().numpy().tolist()
                }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(save_dict, save_path)
            print(f'spk embedding saved to {save_path}')
        
        spk_embedding = spk_embedding.detach().cpu().tolist()
        return spk_embedding


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf/config_qwen3_0.6B.yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to merged .pt or DeepSpeed ZeRO directory (e.g., checkpoints/epoch_0_step_15000)')
    parser.add_argument('--output', type=str, default='save_embedding/test_spkembedding.pt')
    parser.add_argument('--n_timesteps', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    model = VoiceDesignInfer(
        flow_ckpt=args.ckpt,
        config_path=args.config,
        device=args.device,
        n_timesteps=args.n_timesteps,
        temperature=args.temperature,
    )

    input_texts = [
        '该角色是一个幼年男性，身份是太孙殿下，天资聪颖、沉稳自省、偶尔自负，展现出超越年龄的成熟与睿智',
        '该角色是一个老年男性，身份是陈桃的叔父和保护者，谨慎稳重、对外界警惕，传统朴实',
        '该角色是一个老年男性，身份是工业设计专家，热情专业，学者气质，传统气功爱好者',
        '该角色是一个老年男性，身份是宋家老友和门卫，善良忠诚、朴实无华'
        ]

    model.infer(input_texts, args.output)





