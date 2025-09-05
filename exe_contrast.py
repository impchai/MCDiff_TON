import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights







class PreLNResidual(nn.Module):

    def __init__(self, dim, fn: nn.Module, dropout=0.1, use_gate=True, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.fn = fn
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.gate = nn.Parameter(torch.tensor(1.0)) if use_gate else None

    def forward(self, x):
        y0 = self.fn(self.norm(x))
        y = self.drop(y0)
        if self.gate is not None:
            y = torch.tanh(self.gate) * y   # 可学习门控，训练更稳
        return y + y0


class SimpleRegressor(nn.Module):
    def __init__(self, poi_dim=14, human_dim=1, use_img=True, freeze_img=True):
        super().__init__()
        self.use_img = use_img

        if use_img:
            self.si_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.si_encoder.fc = nn.Identity()
            if freeze_img:
                for p in self.si_encoder.parameters():
                    p.requires_grad = False


        self.projector_si = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )



        self.poinet = nn.Sequential(
            # nn.LayerNorm(20),
            nn.Linear(14, 64),
            # nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.humannet = nn.Sequential(
            # nn.LayerNorm(20),
            nn.Linear(1, 64),
            # nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(64, 64),
        )



        self.data_m_pro = nn.Sequential(
            nn.Linear(2, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
        )

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.01), dtype=torch.float32))

        # -------------------------------
        ctx_mlp = nn.Sequential(
            nn.Linear(3 * 64, 64, bias=False),
            nn.GELU(),  # 建议用 GELU，和 Transformer 系列更配
            nn.Dropout(0.1),
            nn.Linear(64, 64, bias=False),
        )
        ent_mlp = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64, bias=False),
        )
        self.proj_context = PreLNResidual(3 * 64, ctx_mlp, dropout=0.0, use_gate=True)
        self.projector_ent = PreLNResidual(64, ent_mlp, dropout=0.0, use_gate=True)

        self.mean_predictor = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.register_buffer("y_mean", torch.zeros(2))  # [2]
        self.register_buffer("y_std", torch.ones(2))  # [2]


        self.ctx_head_ctr = nn.Sequential(
            nn.Linear(64, 64, bias=False), nn.GELU(), nn.Linear(64, 64, bias=False)
        )

    def set_target_stats(self, mean, std):
        self.y_mean.copy_(mean)
        self.y_std.copy_(std.clamp_min(1e-8))



    def _encode_context(self, si, poi, human):
        si = self.si_encoder(si)
        si_f = self.projector_si(si)                        # [B,64]
        human = self.humannet(human.sum(-1, keepdim=True))  # [B,64]
        poi = self.poinet(poi)                              # [B,64]
        h = torch.cat([human, poi, si_f], dim=1)
        ctx = self.proj_context(h)
        return ctx#F.normalize(ctx, dim=-1)                     # 归一化

    def _encode_entity(self, y_norm):  # 这里假定输入已经是标准化后的 [B,2]
        x = self.data_m_pro(y_norm.float())
        ent = self.projector_ent(x)
        return ent

    @torch.no_grad()
    def predict_mean(self, si, poi, human):
        ctx = self._encode_context(si, poi, human)
        out = self.mean_predictor(ctx)  # [B,4] or [B,2]
        mu = out

        return mu * self.y_std + self.y_mean



    def forward(self, si, poi, human, y, tau=0.07, lam_reg=1.0, lam_ctr=0.2):
        # --- 标准化标签 ---
        y_norm = (y - self.y_mean) / self.y_std

        # --- 编码 ---
        ctx_raw = self._encode_context(si, poi, human)   # [B,64]
        ent_raw = self._encode_entity(y_norm)            # [B,64]

        # --- 回归 ---
        pred = self.mean_predictor(ctx_raw)            # [B,2]
        loss_reg = F.huber_loss(pred, y_norm, delta=1.0)

        # --- 对比学习 ---
        zc = F.normalize(self.ctx_head_ctr(ctx_raw), dim=-1)  # [B,64]
        ze = F.normalize(ent_raw, dim=-1)                     # [B,64]

        logits = self.logit_scale *(zc @ ze.t())# / tau                          # [B,B]
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ctr = F.cross_entropy(logits, labels)

        loss = lam_reg * loss_reg + lam_ctr * loss_ctr
        return loss




