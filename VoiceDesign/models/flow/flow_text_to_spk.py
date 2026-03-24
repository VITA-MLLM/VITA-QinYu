from typing import Dict, Optional
import torch
import torch.nn as nn
from omegaconf import DictConfig

# 导入我们修改后的、专用的CFM类
#from .flow_matching_text_to_spk import ConditionalCFM

'''
作用：训练入口/推理入口
训练时，从 batch 取 text 与目标 spk_embedding，调用Flow matching文件的（1）loss计算函数(预测向量场时)。
推理时，直接调用Flow matching文件的（3）forward方法进行推理。

'''

class Text2SpeakerEmbeddingFlow(nn.Module):
    """
    一个专门用于从文本描述生成Speaker Embedding的Flow Matching模型。
    该模型结构简洁，直接将文本条件注入扩散解码器。
    """
    def __init__(self,
                 spk_embed_dim: int,
                 decoder: torch.nn.Module):
        super().__init__()
        self.spk_embed_dim = spk_embed_dim
        self.decoder = decoder

        if not hasattr(self.decoder, 'text_encoder'):
             raise ValueError("Decoder must have a 'text_encoder' attribute.")

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        计算训练损失。

        Args:
            batch (dict): 包含 'speaker_embedding' 和 'text_descriptions' 的字典。
            device (torch.device): 计算设备。

        Returns:
            Dict[str, Optional[torch.Tensor]]: 包含损失的字典。
        """
        target_spk_emb = batch['spk_embedding'].to(device)
        text_descriptions = batch['text']

        # 检查输入维度
        if target_spk_emb.dim() != 2 or target_spk_emb.shape[1] != self.spk_embed_dim:
            raise ValueError(f"Expected speaker_embedding to have shape [B, {self.spk_embed_dim}], but got {target_spk_emb.shape}")

        # 1. 调整扩散目标: 将Speaker Embedding视为长度为1的序列
        # [B, Dim] -> [B, Dim, 1]
        x1 = target_spk_emb.unsqueeze(-1)

        # 2. 计算损失
        # ConditionalCFM内部会处理所有逻辑
        loss, _ = self.decoder.compute_loss(
            x1=x1,
            text_descriptions=text_descriptions
        )

        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  text_descriptions: list,
                  n_timesteps: int = 20,
                  temperature: float = 1.0) -> torch.Tensor:
        """
        根据文本描述生成Speaker Embedding。

        Args:
            text_descriptions (list): 文本描述列表。
            n_timesteps (int): 扩散步数。
            temperature (float): 采样温度。

        Returns:
            torch.Tensor: 生成的Speaker Embedding, shape [B, Dim]。
        """
        b = len(text_descriptions)
        device = next(self.decoder.parameters()).device

        # 准备模型推理所需的输入形状
        mu = torch.zeros(b, self.spk_embed_dim, 1, device=device)
        mask = torch.ones(b, 1, 1, device=device)

        # 调用decoder(ConditionalCFM)的forward方法进行推理
        generated_spk_emb_seq = self.decoder(
            mu=mu,
            mask=mask,
            n_timesteps=n_timesteps,
            temperature=temperature,
            text_descriptions=text_descriptions
        )

        # 将输出从 [B, Dim, 1] reshape回 [B, Dim]
        generated_spk_emb = generated_spk_emb_seq.squeeze(-1)

        return generated_spk_emb

