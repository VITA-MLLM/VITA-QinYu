#!/usr/bin/env python3
"""
文本条件Flow Matching示例脚本

这个脚本演示了如何使用修改后的cosyvoice flow matching代码，
通过文本描述来生成对应的speaker embedding。
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from flow.flow_matching import ConditionalCFM
from flow.text_encoder import T5TextEncoder, CrossAttentionModule


from flow.kandit_estimator import Decoder

class TextConditionedEstimator(Decoder):
    """基于KANDIT的文本条件估计器"""
    
    def __init__(self, in_channels, hidden_dim=768, spk_emb_dim=64, text_hidden_dim=768):
        # 计算gin_channels：speaker embedding + 文本条件
        gin_channels = spk_emb_dim + text_hidden_dim
        
        super().__init__(
            hidden_channels=in_channels,
            out_channels=in_channels,  # 输出mel特征
            filter_channels=512,
            dropout=0.1,
            n_layers=8,
            n_heads=8,
            kernel_size=3,
            gin_channels=gin_channels
        )
        
        self.text_hidden_dim = text_hidden_dim
        
    def forward(self, x, mask, mu, t, spks=None, cond=None, text_condition=None):
        """
        Args:
            x: 输入特征 [batch_size, n_feats, mel_timesteps]
            mask: 掩码 [batch_size, 1, mel_timesteps]
            mu: 条件特征 [batch_size, n_feats, mel_timesteps]
            t: 时间步 [batch_size]
            spks: speaker embedding [batch_size, spk_emb_dim]
            cond: 条件特征 [batch_size, n_feats, mel_timesteps]
            text_condition: 文本条件 [batch_size, seq_len, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # 处理文本条件
        if text_condition is not None:
            # 将文本条件池化为单个向量
            text_pooled = text_condition.mean(dim=1)  # [batch_size, text_hidden_dim]
        else:
            text_pooled = torch.zeros(batch_size, self.text_hidden_dim, device=x.device)
        
        # 拼接speaker embedding和文本条件
        if spks is not None:
            combined_condition = torch.cat([spks, text_pooled], dim=1)  # [batch_size, spk_emb_dim + text_hidden_dim]
        else:
            combined_condition = text_pooled
        
        # 调用父类的forward方法
        return super().forward(x, mask, mu, t, spks=combined_condition, cond=cond)


def create_text_conditioned_model():
    """创建文本条件的flow matching模型"""
    
    # 配置参数
    cfm_params = DictConfig({
        'sigma_min': 1e-06,
        'solver': 'euler',
        't_scheduler': 'cosine',
        'training_cfg_rate': 0.2,
        'inference_cfg_rate': 0.7,
        'reg_loss_type': 'l1'
    })
    
    # 创建基于KANDIT的扩散模型主干网络
    estimator = Decoder(
        hidden_channels=80,      # mel特征维度
        out_channels=80,         # 输出mel特征
        filter_channels=512,     # 隐藏层维度
        dropout=0.1,
        n_layers=8,             # 网络层数
        n_heads=8,              # 注意力头数
        kernel_size=3,
        gin_channels=64 + 768   # speaker embedding + text features
    )
    
    # 创建文本编码器配置
    text_encoder_config = {
        'model_name': 't5-base',
        'max_length': 512,
        'hidden_dim': 768
    }
    
    # 创建条件flow matching模型
    model = ConditionalCFM(
        in_channels=80,
        cfm_params=cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator=estimator,     # 使用KANDIT主干网络
        use_text_conditioning=True,
        text_encoder_config=text_encoder_config
    )
    
    return model


def example_training_step():
    """示例训练步骤"""
    
    # 创建模型
    model = create_text_conditioned_model()
    model.train()
    
    # 模拟数据
    batch_size = 4
    mel_timesteps = 100
    n_feats = 80
    spk_emb_dim = 64
    
    # 输入数据
    x1 = torch.randn(batch_size, n_feats, mel_timesteps)  # 目标mel特征
    mask = torch.ones(batch_size, 1, mel_timesteps)  # 掩码
    mu = torch.randn(batch_size, n_feats, mel_timesteps)  # 条件特征
    spks = torch.randn(batch_size, spk_emb_dim)  # speaker embedding
    cond = torch.randn(batch_size, n_feats, mel_timesteps)  # 条件
    feat_len = torch.tensor([mel_timesteps] * batch_size)  # 特征长度
    
    # 文本描述
    text_descriptions = [
        "年轻女性的声音，听起来年轻可爱",
        "成熟男性的声音，低沉有力",
        "儿童的声音，活泼天真",
        "老年人的声音，温和慈祥"
    ]
    
    # 前向传播
    loss, y = model.compute_loss(
        x1=x1,
        mask=mask,
        mu=mu,
        spks=spks,
        cond=cond,
        feat_len=feat_len,
        text_descriptions=text_descriptions
    )
    
    print(f"Flow Loss: {loss.item():.4f}")
    print(f"Output shape: {y.shape}")
    
    return loss, y


def example_inference():
    """示例推理"""
    
    # 创建模型
    model = create_text_conditioned_model()
    model.eval()
    
    # 模拟数据
    batch_size = 1
    mel_timesteps = 100
    n_feats = 80
    spk_emb_dim = 64
    
    # 输入数据
    mu = torch.randn(batch_size, n_feats, mel_timesteps)  # 条件特征
    mask = torch.ones(batch_size, 1, mel_timesteps)  # 掩码
    spks = torch.randn(batch_size, spk_emb_dim)  # speaker embedding
    cond = torch.randn(batch_size, n_feats, mel_timesteps)  # 条件
    
    # 文本描述
    text_descriptions = ["年轻女性的声音，听起来年轻可爱"]
    
    # 推理
    with torch.no_grad():
        output = model(
            mu=mu,
            mask=mask,
            n_timesteps=20,
            temperature=1.0,
            spks=spks,
            cond=cond,
            text_descriptions=text_descriptions
        )
    
    print(f"Generated output shape: {output.shape}")
    
    return output


if __name__ == "__main__":
    print("=== 文本条件Flow Matching示例 ===\n")
    
    print("1. 训练步骤示例:")
    loss, y = example_training_step()
    
    print("\n2. 推理示例:")
    output = example_inference()
    
    print("\n=== 示例完成 ===") 