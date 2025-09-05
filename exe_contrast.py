import torch
import yaml
import os
from torch.optim import  AdamW
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
        y = self.fn(self.norm(x))
        y = self.drop(y)
        if self.gate is not None:
            y = torch.tanh(self.gate) * y
        return y


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
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )  #

        self.humannet = nn.Sequential(
            nn.Linear(1, 64),
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
            nn.GELU(),
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
        return ctx

    def _encode_entity(self, y_norm):
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

        y_norm = (y - self.y_mean) / self.y_std

        ctx_raw = self._encode_context(si, poi, human)   # [B,64]
        ent_raw = self._encode_entity(y_norm)            # [B,64]


        pred = self.mean_predictor(ctx_raw)              # [B,2]
        loss_reg = F.huber_loss(pred, y_norm, delta=1.0)

        zc = F.normalize(self.ctx_head_ctr(ctx_raw), dim=-1)  # [B,64]
        ze = F.normalize(ent_raw, dim=-1)                     # [B,64]

        logits = (zc @ ze.t()) / tau                          # [B,B]
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ctr = F.cross_entropy(logits, labels)

        loss = lam_reg * loss_reg + lam_ctr * loss_ctr
        return loss

def compute_target_stats(loader, device):
    m = torch.zeros(2, device=device)
    v = torch.zeros(2, device=device)
    n = 0
    for batch in loader:
        y = batch[4].to(device).float()        # data_mean: [B,2]
        m += y.sum(0)
        v += (y*y).sum(0)
        n += y.size(0)
    mean = m / n
    var  = (v / n) - mean**2
    std  = torch.sqrt(var.clamp_min(1e-8))
    return mean, std


def trainer(args, model, train_loader, optimizer, epoch, total_epoch):
    loss_epoch = []
    model.train()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        img, poi, human, data_mean = batch[1].to(args.device).float(),batch[2].to(args.device).float(),batch[3].to(args.device).float(),batch[4].to(args.device).float()
        loss= model(img, poi, human, data_mean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_epoch.append(loss.item())
    return np.mean(loss_epoch)




def validate(args, model, val_loader, epoch=None, total_epoch=None, tau=0.07):
    model.eval()
    loss_epoch, top1_epoch = [], []
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            img   = batch[1].to(args.device).float()
            poi   = batch[2].to(args.device).float()
            human = batch[3].to(args.device).float()
            data_mean = batch[4].to(args.device).float()

            loss= model(img, poi, human, data_mean)
            loss_epoch.append(loss.item())

    model.train()


def validate_reg_metrics(args, model, val_loader):
    model.eval()
    Y_list, Yhat_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            img   = batch[1].to(args.device).float()
            poi   = batch[2].to(args.device).float()
            human = batch[3].to(args.device).float()
            y     = batch[4].to(args.device).float()
            y_hat = model.predict_mean(img, poi, human)
            Y_list.append(y); Yhat_list.append(y_hat)
    model.train()
    Y, Yhat = torch.cat(Y_list), torch.cat(Yhat_list)

    mse = torch.mean((Y - Yhat)**2).item()
    mae = torch.mean(torch.abs(Y - Yhat)).item()

    return mse, mae


@torch.no_grad()
def eval_mean(
    args,
    model,                 # Pair_CLIP_SI
    val_loader,
):

    model.eval()


    Y_list, Yhat_list = [], []
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            img   = batch[1].to(args.device).float()
            poi   = batch[2].to(args.device).float()
            human = batch[3].to(args.device).float()
            y     = batch[4].to(args.device).float()  # data_mean



            y_hat = model.predict_mean(img, poi, human)
            Y_list.append(y)
            Yhat_list.append(y_hat)
    model.train()

    Y     = torch.cat(Y_list, dim=0)
    Y_hat = torch.cat(Yhat_list, dim=0)


    diff = Y_hat - Y
    mse  = torch.mean(diff**2, dim=0)                 # per-dim
    mae  = torch.mean(torch.abs(diff), dim=0)
    rmse = torch.sqrt(mse)


    Y_mean = torch.mean(Y, dim=0, keepdim=True)
    ss_tot = torch.sum((Y - Y_mean)**2, dim=0)
    ss_res = torch.sum((Y - Y_hat)**2, dim=0)
    r2     = 1.0 - ss_res / (ss_tot + 1e-12)


    mse_all  = torch.mean((Y - Y_hat)**2).item()
    mae_all  = torch.mean(torch.abs(Y - Y_hat)).item()
    rmse_all = np.sqrt(mse_all)

    print(
        f"  micro:  MSE={mse_all:.6f}  RMSE={rmse_all:.6f}  MAE={mae_all:.6f}\n"
        f"  dim0:   MSE={float(mse[0]):.6f}  RMSE={float(rmse[0]):.6f}  MAE={float(mae[0]):.6f}  R2={float(r2[0]):.4f}\n"
        f"  dim1:   MSE={float(mse[1]):.6f}  RMSE={float(rmse[1]):.6f}  MAE={float(mae[1]):.6f}  R2={float(r2[1]):.4f}"
    )





def contrastive_pretrain(args, foldername, train_loader, valid_loader, test_loader):
    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(foldername, exist_ok=True)
    print('-------------------------contrastive stage begin..-------------------------')

    epoch_contrast = 1000
    val_interval = 50  #
    best_val_loss = float("inf")

    model0 = SimpleRegressor()
    model0 = model0.to(args.device)
    opt0 = AdamW([p for p in model0.parameters() if p.requires_grad], lr=config["train"]["lr"], weight_decay=1e-3)


    dm_mean, dm_std = compute_target_stats(train_loader, args.device)
    model0.set_target_stats(dm_mean, dm_std)
    print("target mean:", dm_mean.tolist(), " std:", dm_std.tolist())


    for epoch in range(0, epoch_contrast):
        trainer(args, model0, train_loader, opt0, epoch, epoch_contrast)

        if ((epoch + 1) % val_interval) == 0:
            validate(args, model0, valid_loader, epoch=epoch, total_epoch=epoch_contrast)
            val_mse, val_mae = validate_reg_metrics(args, model0, valid_loader)
            print(
                f"[Val] Epoch {epoch + 1}/{epoch_contrast}  MSE:{val_mse:.6f}  MAE:{val_mae:.6f}")


            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_path = os.path.join(foldername, "best.tar")
                torch.save(model0.state_dict(), best_path)
                print(f"New best (by MSE) saved → {best_path}  MSE={best_val_loss:.6f}")

            eval_mean(args, model0, valid_loader)

    print('-------------------------contrastive stage finished-------------------------')


