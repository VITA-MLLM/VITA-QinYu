from typing import Optional, Dict
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

# 导入原始的BASECFM，我们的新类仍然继承自它
from matcha.models.components.flow_matching import BASECFM
from .text_encoder import T5TextEncoder, RobertaTextEncoder, QwenEmbeddingTextEncoder


'''
（1）实现预测向量场时的loss计算函数compute_loss、
    训练：采样时间 t 和噪声，构造 (y, u)，用 estimator(y, mask, t, cond) 预测导数(向量场)，和 u 做 MSE。
（2）ODE求解器的实现函数solve_euler（调用dit_estimator文件中的函数）、
    推理：forward(mu, mask, n_timesteps, text_descriptions) 用欧拉法 solve_euler 多步积分，步步调用 estimator(·)，输出 [B, Dim, 1]。
（3）模型推理函数forward（调用solve_euler函数）
'''


class ConditionalCFMForSpeakerEmbedding(BASECFM):
    """
    一个专用于从文本生成Speaker Embedding的Conditional Flow Matching模块。
    它被设计为处理固定长度的向量（视为长度为1的序列），并仅使用文本作为条件。
    """
    def __init__(self,
                 spk_embed_dim: int,
                 cfm_params: DictConfig,
                 estimator: torch.nn.Module,
                 use_text_conditioning: bool = True,
                 text_encoder_config: Optional[Dict] = None):
        super().__init__(
            n_feats=spk_embed_dim, # n_feats is the speaker embedding dimension
            cfm_params=cfm_params,
            n_spks=0, # n_spks is no longer relevant in this context
            spk_emb_dim=spk_embed_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.use_text_conditioning = use_text_conditioning

        self.estimator = estimator

        # 初始化文本编码器
        if use_text_conditioning:
            if text_encoder_config is None:
                raise ValueError("text_encoder_config must be provided and must match the encoder used in training")
            model_name = text_encoder_config.get('model_name', None)
            if model_name is None:
                raise ValueError("text_encoder_config.model_name is required")
            name_l = str(model_name).lower()
            if 'roberta' in name_l:
                self.text_encoder = RobertaTextEncoder(**text_encoder_config)
            elif 't5' in name_l:
                self.text_encoder = T5TextEncoder(**text_encoder_config)
            elif 'qwen' in name_l:
                self.text_encoder = QwenEmbeddingTextEncoder(**text_encoder_config)
            else:
                raise ValueError(f"Unsupported text encoder model_name: {model_name}. Expected 'roberta-*' or 't5-*'.")
        else:
            self.text_encoder = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, text_descriptions=None):
        """
        Performs the forward diffusion process to generate a speaker embedding.
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # 准备文本条件 for inference
        text_condition = None
        if self.use_text_conditioning and text_descriptions is not None and self.text_encoder is not None:
            text_features, _ = self.text_encoder(text_descriptions)
            text_condition = text_features  # 直接使用，不需要取均值

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, c=text_condition)

    def solve_euler(self, x, t_span, mu, mask, c=None):
        """
        Fixed Euler solver for the ODE.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, t, cond=c)
            if self.inference_cfg_rate > 0:
                if c is None:
                    # No text condition available: skip CFG branch to avoid passing cond=None
                    cfg_dphi_dt = dphi_dt
                else:
                    cfg_cond = torch.zeros_like(c)
                    cfg_dphi_dt = self.estimator(x, mask, t, cond=cfg_cond)
                dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1]

    def compute_loss(self, x1, text_descriptions: list):
        """
        Computes the diffusion loss for text-to-speaker-embedding generation.
        """
        b, _, _ = x1.shape
        mask = torch.ones(b, 1, 1, device=x1.device)

        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)

        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        text_condition = None
        if self.use_text_conditioning and text_descriptions is not None and self.text_encoder is not None:
            text_features, _ = self.text_encoder(text_descriptions)
            # text_features 已经是 [B, output_dim] 形状，直接使用
            text_condition = text_features

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            if text_condition is not None:
                text_condition = text_condition * cfg_mask.view(-1, 1)

        pred = self.estimator(y, mask, t.squeeze(), cond=text_condition)
        flow_loss = F.mse_loss(pred, u)
        return flow_loss, y


