import math

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from exe_contrast import SimpleRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F





def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def Conv1d_with_init(in_channels, out_channels, kernel_size):

    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class STAR(nn.Module):
    def __init__(self, d_series, num_heads=8):
        super().__init__()
        """
        STar Aggregate-Redistribute Module (minimal, stable)
        输入/输出: [B, N=2, T, C]，其中 C = d_series
        """
        C = d_series


        self.ln_chan = nn.LayerNorm(2*C)
        self.attn_channel = Attention(
            dim=2*C, num_heads=num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.1
        )


        self.gen3 = Conv1d_with_init(2*C, C, 1)
        self.gen4 = Conv1d_with_init(C, C, 1)

    def forward(self, x):
        B, N, T, C = x.shape

        combined_mean2 = torch.mean(x, dim=1, keepdim=True)       # [B,1,T,C]
        combined_mean2_rep = combined_mean2.expand(-1, N, -1, -1)

        combined_mean_cat = torch.cat([x, combined_mean2_rep], -1)  # [B,2,T,2C]


        z = combined_mean_cat.permute(0, 2, 1, 3).reshape(B*T, N, 2*C)  # [B*T,2,2C]
        z = self.ln_chan(z)
        z = self.attn_channel(z)                                         # [B*T,2,2C]
        z = z.reshape(B, T, N, 2*C).permute(0, 2, 1, 3)                  # [B,2,T,2C]


        h = z.reshape(B, N*T, 2*C).permute(0, 2, 1)  # [B,2C, N*T] 作为 Conv1d 的 (B, C_in, L)
        h = F.gelu(self.gen3(h))                     # [B,C, N*T]
        h = self.gen4(h)                             # [B,C, N*T]
        h = h.permute(0, 2, 1).reshape(B, N, T, C)   # [B,N,T,C]

        out = F.gelu(h + x)
        return out



    def _build_embedding(self, num_steps, dim=64):

        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)

        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model,hour_size=24, weekday_size = 7):
        super(TemporalEmbedding, self).__init__()

        hour_size = hour_size
        weekday_size = weekday_size

        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.timeconv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1)

    def forward(self, x):

        x = x.long()
        hour_x = self.hour_embed(x[:,:,0])
        weekday_x = self.weekday_embed(x[:,:,1])
        timeemb = self.timeconv(hour_x.transpose(1,2)+weekday_x.transpose(1,2)).transpose(1,2)

        return timeemb
class DataEmbedding(nn.Module):
    def __init__(self, d_model, size1 = 24, size2=7 ):
        super(DataEmbedding, self).__init__()
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, hour_size  = size1, weekday_size = size2)

    def forward(self, x):

        '''
        x: N, T, C, H, W
        x_mark: N, T, D
        '''
        TimeEmb = self.temporal_embedding(x)
        return  TimeEmb

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.linear = nn.Linear(hidden_size,  out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size1, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        self.attn_time = Attention(hidden_size1, num_heads=num_heads, qkv_bias=True, attn_drop=0, proj_drop=0.1,**block_kwargs)
        self.cross = STAR(hidden_size1, hidden_size1)
        self.norm2 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size1 * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size1, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size1, 6 * hidden_size1, bias=True)
        )

    def forward(self, x, c):
        B, N, T, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn_time(modulate(self.norm1(x), shift_msa, scale_msa).reshape(B * N, T, C)).reshape(B,N,T,C)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.cross(x)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=1):
        super().__init__()
        self.channels = config["channels"]

        self.pretrained_contrast_model = SimpleRegressor()
        for p in self.pretrained_contrast_model.si_encoder.parameters():
            p.requires_grad = False

        self.t_embedder = TimestepEmbedder(self.channels)

        self.input_projection = nn.Linear(inputdim, self.channels)

        self.mean_projection = nn.Sequential(
            nn.Linear(2, self.channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.channels, self.channels, bias=False),
        )


        self.Embedding = DataEmbedding(self.channels)

        self.x_projection = nn.Linear(2*self.channels, self.channels)
        self.final_layer = FinalLayer(self.channels, 1)


        self.blocks = nn.ModuleList([
            DiTBlock(self.channels, num_heads=8, mlp_ratio=2) for _ in range(5)
        ])

        self.cross_atten_layer = nn.TransformerDecoderLayer(d_model=self.channels, nhead=8, batch_first=True)


    def forward(self, x, cond_info, diffusion_step, image, poi, human, mean_value):
        B, K, L = x.shape
        reshaped_tensor = torch.zeros((B, L, K), dtype=cond_info.dtype, device=cond_info.device)
        reshaped_tensor[:, :, 0] = torch.arange(L) % 24
        reshaped_tensor[:, :, 1] = torch.arange(L) // 24
        cond_info = self.Embedding(reshaped_tensor)
        timeemb= cond_info
        time_feature = timeemb.unsqueeze(1).repeat(1, 2, 1, 1)

        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = F.relu(x)

        mean_img = self.pretrained_contrast_model.predict_mean(image, poi, human)

        mean_x= self.mean_projection(mean_img)
        mean_x = mean_x.unsqueeze(1).unsqueeze(1).repeat(1, 2,L, 1)

        static_feature = mean_x

        diffusion_emb = self.t_embedder(diffusion_step).unsqueeze(1).unsqueeze(1).repeat(1,K,L,1)


        dynamic_feature = self.cross_atten_layer(static_feature.reshape(B, 2*L,-1),  time_feature.reshape(B, 2*L,-1)).reshape(B,2,L,-1) + time_feature

        context_feature = self.x_projection(torch.cat([static_feature, dynamic_feature], dim=-1))
        context_feature = F.relu(context_feature)

        x = x + diffusion_emb

        for block in self.blocks:
            x = block(x, context_feature)

        x = self.final_layer(x, context_feature).reshape(B, K, L)


        return x



