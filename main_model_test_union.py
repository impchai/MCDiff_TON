import numpy as np
import torch
import torch.nn as nn
from diff_models_withAtten_modify_test_union import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, diffusion, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.mse = nn.MSELoss()
        self.diffusion = diffusion
        self.emb_time_dim = config["diffusion"]["channels"]
        self.emb_feature_dim = config["diffusion"]["channels"]



        # 计算总的嵌入维度
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        # 配置扩散模型s
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["side_dim_time"] = self.emb_time_dim


        self.diffmodel = diff_CSDI(config_diff, inputdim=1)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat) # 累乘函数
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):

        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, observed_data):
        B, N, L = observed_data.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb),生成时间嵌入
        return time_embed

    def calc_loss_valid(
        self, observed_data, side_info, is_train, image, poi, human, mean_value
    ):
        '''
        计算在模型验证过程中的损失, 取多个时间步并计算均值
        '''
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, side_info, is_train, image, poi, human, mean_value,set_t=t
            )
            loss_sum += loss.detach()


        return loss_sum / self.num_steps


    def calc_loss(
        self, observed_data, side_info, is_train, image, poi, human,mean_value, set_t=-1
    ):
        '''
        计算模型在给定时间步 t 下的损失
        '''
        B, N, L = observed_data.shape


        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        x_start = observed_data

        loss_dict = self.diffusion.training_losses(
            self.diffmodel, x_start, side_info, t, image, poi, human, mean_value
        )
        loss_diff = loss_dict["loss"].mean()
        loss_pred = self.diffmodel.pretrained_contrast_model(image, poi, human, mean_value)
        total_loss =  loss_diff + 0.01 *loss_pred

        return total_loss

    def set_input_to_diffmodel(self, noisy_data, observed_data):
        total_input = noisy_data  # (B,1,K,L)

        return total_input

    def impute(self, observed_data, side_info, image, poi, human, mean_value):
        '''
        对观测数据进行多次插补，生成多个可能的插补样本
        返回包含生成的插补样本的张量
        '''
        B,K, L = observed_data.shape
        imputed_samples =self.diffusion.p_sample_loop(
                    self.diffmodel, observed_data.shape, observed_data, side_info, image, poi, human, mean_value, clip_denoised=True, model_kwargs=None, progress=False,
                    device=self.device
                ).detach()

        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_tp,
            image,
            poi,
            human,
            mean_value
        ) = self.process_data(batch)

        side_info = observed_tp
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, side_info, is_train, image, poi, human, mean_value)

    def evaluate(self, batch):
        (
            observed_data,
            observed_tp,
            image,
            poi,
            human,
            mean_value,
        ) = self.process_data(batch)

        with torch.no_grad():
            side_info = observed_tp
            samples = self.impute(observed_data, side_info, image, poi, human, mean_value)

        return samples, observed_data, observed_tp

class CSDI_Value(CSDI_base):
    def __init__(self, config, diffusion, device, target_dim=1):
        super(CSDI_Value, self).__init__(target_dim, config, diffusion, device)

    def process_data(self, batch):
        observed_data = batch[0].to(self.device).float()
        length = batch[0].shape[-1]
        observed_tp = torch.arange(length).repeat(len(batch[0]), 1).to(self.device).float()
        image = batch[1].to(self.device).float()
        poi = batch[2].to(self.device).float()
        human = batch[3].to(self.device).float()
        mean_value = batch[4].to(self.device).float()


        return (
            observed_data,
            observed_tp,
            image,
            poi,
            human,
            mean_value,
        )
