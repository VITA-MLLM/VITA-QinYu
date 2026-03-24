import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import pack, repeat



# reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py
class Decoder(nn.Module):
    def __init__(self, 
                 hidden_channels, 
                 out_channels, 
                 filter_channels, 
                 dropout=0.05, 
                 n_layers=1, 
                 n_heads=4, 
                 kernel_size=3, 
                 gin_channels=0):
        '''
        args 

        for  cosyvoice flow matching 
        out_channels = mel_channels + mel_channels
        filter channels mean hidden 
        gin_channels = spk_embedding_channels + mel_cond_channels    
        
        '''
    

        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels

        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(hidden_channels, hidden_channels, filter_channels)

        
        self.blocks = nn.ModuleList([DitWrapper(hidden_channels, filter_channels, n_heads, kernel_size, dropout, gin_channels, hidden_channels) for _ in range(n_layers)])
        
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1) 

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, mask, mu, t, c=None, cond=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            c (_type_): shape (batch_size, gin_channels)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        x = pack([x, mu], "b * t")[0]

        if cond is not None:
            x = pack([x, cond], "b * t")[0]


        t = self.time_mlp(self.time_embeddings(t))
        # x = torch.cat((x, mu), dim=1)

        for block in self.blocks:
            x = block(x, c, t, mask)

        output = self.final_proj(x * mask)

        return output * mask

'''

Dit Wrapper and time embedding
'''

class DitWrapper(nn.Module):
    """ add FiLM layer to condition time embedding to DiT """
    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0, time_channels=0):
        super().__init__()
        self.time_fusion = FiLMLayer(hidden_channels, time_channels)
        self.conv1 = ConvNeXtBlock(hidden_channels, filter_channels, gin_channels)
        self.conv2 = ConvNeXtBlock(hidden_channels, filter_channels, gin_channels)
        self.conv3 = ConvNeXtBlock(hidden_channels, filter_channels, gin_channels)
        self.block = DiTConVBlock(hidden_channels, hidden_channels, num_heads, kernel_size, p_dropout, gin_channels)
            
    def forward(self, x, c, t, x_mask):
        x = self.time_fusion(x, t) * x_mask

        x = self.conv1(x, c, x_mask)
        x = self.conv2(x, c, x_mask)
        x = self.conv3(x, c, x_mask)
        x = self.block(x, c, x_mask)
        return x

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Reference: https://arxiv.org/abs/1709.07871
    """
    def __init__(self, in_channels, cond_channels):

        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Conv1d(cond_channels, in_channels * 2, 1)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.film(c.unsqueeze(2)), chunks=2, dim=1)
        return gamma * x + beta
    
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, filter_channels, gin_channels):
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = StyleAdaptiveLayerNorm(in_channels, gin_channels)
        self.pwconv = nn.Sequential(nn.Linear(in_channels, filter_channels),
                                    nn.GELU(),
                                    nn.Linear(filter_channels, in_channels))

    def forward(self, x, c, x_mask) -> torch.Tensor:
        residual = x
        x = self.dwconv(x) * x_mask
        x = self.norm(x.transpose(1, 2), c)
        x = self.pwconv(x).transpose(1, 2)
        x = residual + x
        return x * x_mask

class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels):
        """
        Style Adaptive Layer Normalization (SALN) module.

        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        """
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels

        # ori: self.saln = nn.Linear(cond_channels, in_channels * 2, 1)
        self.saln = nn.Linear(cond_channels, in_channels * 2, 1)
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.constant_(self.saln.bias.data[:self.in_channels], 1)
        nn.init.constant_(self.saln.bias.data[self.in_channels:], 0)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.saln(c.unsqueeze(1)), chunks=2, dim=-1)
        return gamma * self.norm(x) + beta
        
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, filter_channels),
            nn.SiLU(inplace=True),
            nn.Linear(filter_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)




'''
Transformer Blocks

'''

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

        
class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret



class AttentionWithFastKAN(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = FastKANLayer(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = FastKANLayer(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x







class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels
    
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
    self.drop = nn.Dropout(p_dropout)
    self.act1 = nn.GELU(approximate="tanh")

  def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask
    


# modified from https://github.com/sh-lee-prml/HierSpeechpp/blob/main/modules.py#L390    
class DiTConVBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionWithFastKAN(hidden_channels, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.mlp = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(gin_channels, hidden_channels) if gin_channels != hidden_channels else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_channels, 6 * hidden_channels, bias=True)
        )
            
    def forward(self, x, c, x_mask):
        """
        Args:
            x : [batch_size, channel, time]
            c : [batch_size, channel]
            x_mask : [batch_size, 1, time]
        return the same shape as x
        """
        x = x * x_mask
        # attn_mask = x_mask.unsqueeze(1) * x_mask.unsqueeze(-1) # shape: [batch_size, 1, time, time]
        # attn_mask = attn_mask.to(torch.bool)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(2).chunk(6, dim=1) # shape: [batch_size, channel, 1]

        x1 = self.modulate(self.norm1(x.transpose(1,2)).transpose(1,2), shift_msa, scale_msa)
        x1 =  self.attn(x1.transpose(1,2)).transpose(1, 2)
        x = x + gate_msa * x1 * x_mask
        x2 = self.modulate(self.norm2(x.transpose(1,2)).transpose(1,2), shift_mlp, scale_mlp)
        x2 = self.mlp(x2, x_mask)
        x = x + gate_mlp * x2
        
        # no condition version
        # x = x + self.attn(self.norm1(x.transpose(1,2)).transpose(1,2),  attn_mask)
        # x = x + self.mlp(self.norm1(x.transpose(1,2)).transpose(1,2), x_mask)
        return x
    
    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift
  

class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


if __name__ == '__main__':
    mel = torch.randn(1, 100, 80)
    spks = torch.randn(1, 100)
    mask = torch.randint(0, 2, (1, 1, 80))
    cond = mel
    spk = mel
    mu = mel
    t = torch.randn(1,)
    decoder = Decoder(hidden_channels=300, out_channels=100, 
                      filter_channels=512, dropout=0.1,
                      n_layers=8, n_heads=2, kernel_size=3,
                      gin_channels=100)
    out = decoder(x=mel, mu=mel, spks=spks, t=t, cond=mel, mask=mask)
    breakpoint()
    print('ok')
