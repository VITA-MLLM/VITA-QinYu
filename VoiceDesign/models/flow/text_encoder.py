import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import re

def format_character_description(description: str) -> str:
    """
    将半结构化的角色描述标准化为自然语言句子。

    Args:
        description (str): 输入的角色描述，格式为 "该角色是一个「...」年「...」性， 性格「...」"。

    Returns:
        str: 标准化后的自然语言描述。

    Example:
        >>> desc = "该角色是一个「中」年「男」性， 性格「豪爽直-率、重义气，行事果断粗暴，气质霸气带痞气」"
        >>> format_character_description(desc)
        '中年男性，性格豪爽直率、重义气，行事果断粗暴，气质霸气带痞气。'
    """
    # 正则表达式，用于匹配和提取「」中的内容
    # (.*?) 是一个非贪婪匹配，用于捕获括号内的所有内容
    pattern = r"该角色是一个「(.*?)」年「(.*?)」性， 性格「(.*?)」"

    match = re.search(pattern, description)

    if match:
        age = match.group(1).strip()
        gender = match.group(2).strip()
        personality = match.group(3).strip()

        # 重新组合成一个更自然的句子
        # 确保每个部分都有内容
        parts = []
        if age and gender:
            parts.append(f"{age}年{gender}性")
        if personality:
            parts.append(f"性格{personality}")

        formatted_desc = "，".join(parts) + "。"

        # 移除非预期的双逗号等
        formatted_desc = formatted_desc.replace('，，', '，')
        return formatted_desc
    else:
        # 如果格式不匹配，进行简单的清理后直接返回
        # 移除模板中的固定文本和符号
        cleaned_desc = description.replace("该角色是一个", "").replace("年", "").replace("性", "").replace("性格", "")
        cleaned_desc = re.sub(r"[「」]", "", cleaned_desc).strip()
        return cleaned_desc if cleaned_desc else "一个通用角色。"


class T5TextEncoder(nn.Module):
    """T5文本编码器，用于将文本描述编码为条件向量"""

    def __init__(self, model_name='t5-base', max_length=512, hidden_dim=768, output_dim=192):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        # 加载T5 tokenizer和encoder
        print('model_name: ', model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.text_encoder = T5EncoderModel.from_pretrained(model_name)

        # 冻结T5参数（可选）
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 投影层，将T5输出投影到目标维度
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_list):
        """
        将文本列表编码为条件向量

        Args:
            text_list (list): 文本描述列表 (半结构化或已标准化)

        Returns:
            torch.Tensor: 编码后的条件向量 [batch_size, max_length, hidden_dim]
            torch.Tensor: 注意力掩码 [batch_size, max_length]
        """
        # 标准化所有输入的文本描述
        standardized_text_list = [format_character_description(desc) for desc in text_list]

        # Tokenize文本
        batch = self.tokenizer(
            standardized_text_list,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # 使用当前模块所在设备，保证与下游 DiT 一致
        device = next(self.parameters()).device
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        # 通过T5编码器
        with torch.no_grad():
            text_features = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]  # [batch_size, seq_len, hidden_dim]

        # 投影到目标维度
        text_features = self.projection(text_features)

        return text_features, attention_mask

    def encode_single_text(self, text):
        """
        编码单个文本

        Args:
            text (str): 输入文本

        Returns:
            torch.Tensor: 编码后的特征 [1, seq_len, hidden_dim]
            torch.Tensor: 注意力掩码 [1, seq_len]
        """
        return self.forward([text])

# 新增：Roberta 文本编码器
class RobertaTextEncoder(nn.Module):
    """Roberta 文本编码器，将文本编码为条件向量，并投影到 output_dim"""
    def __init__(self, model_name='roberta-base', max_length=512, hidden_dim=768, output_dim=192):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        print('model_name: ', model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_list):
        standardized_text_list = [format_character_description(desc) for desc in text_list]
        batch = self.tokenizer(
            standardized_text_list,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        device = next(self.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.text_encoder(**batch)  # last_hidden_state: [B, L, hidden_dim]
            text_features = out.last_hidden_state
        text_features = self.projection(text_features)  # [B, L, output_dim]
        # 对序列维度进行平均池化，得到 [B, output_dim]
        masked_text_features = text_features * batch['attention_mask'].unsqueeze(-1)
        pooled_text_features = masked_text_features.sum(dim=1) / batch['attention_mask'].sum(dim=1, keepdim=True)
        return pooled_text_features, batch['attention_mask']

# 新增：Qwen3/Qwen2 系列 Embedding 文本编码器（使用 HF AutoModel）
class QwenEmbeddingTextEncoder(nn.Module):
    """
    使用 Qwen 系列 embedding/text 模型，将文本编码为句向量并投影到 output_dim。
    返回: (text_features [B, output_dim], attention_mask [B, L])
    """
    def __init__(self, model_name='Qwen/Qwen2.5-7B', max_length=512, hidden_dim=None, output_dim=192):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        print('model_name: ', model_name)
        # 某些 Qwen 模型需要 trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        # 官方示例建议左填充以便 last-token pooling
        if hasattr(self.tokenizer, 'padding_side'):
            self.tokenizer.padding_side = 'left'
        self.text_encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        # 以实际模型 hidden_size 为准，避免与配置不一致
        model_hidden = getattr(self.text_encoder.config, 'hidden_size', None)
        if model_hidden is None:
            model_hidden = getattr(self.text_encoder.config, 'hidden_dim', None)
        if model_hidden is None and hidden_dim is not None:
            model_hidden = hidden_dim
        assert model_hidden is not None, "Cannot infer hidden size from Qwen model config; please set hidden_dim"



        self.projection = nn.Linear(model_hidden, output_dim)

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]

    def forward(self, text_list):
        standardized_text_list = [format_character_description(desc) for desc in text_list]
        batch = self.tokenizer(
            standardized_text_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        device = next(self.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.text_encoder(**batch)
            token_feats = getattr(out, 'last_hidden_state', None)
            if token_feats is None:
                token_feats = getattr(out, 'embeddings', None)
            if token_feats is None:
                raise RuntimeError('Qwen model output does not contain last_hidden_state/embeddings')
            pooled = self._last_token_pool(token_feats, batch['attention_mask'])  # [B, hidden]
            pooled = F.normalize(pooled, p=2, dim=1)
        proj = self.projection(pooled)  # [B, output_dim]
        return proj, batch['attention_mask']

