import argparse
import torch
import datetime
import json
import yaml
import os
import time
from main_model_test_union import CSDI_Value
from dataset_physio_4traffic_new import get_dataloader
from utils_test import train, evaluate
from diffusion import create_diffusion
import numpy as np
import random


def setup_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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

parser = argparse.ArgumentParser(description="MCDiff")
parser.add_argument("--traindata", type=str, default='TrafficData')

parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=30)
parser.add_argument("--city", type=str, default='nc',help='[nc, nj]')

args = parser.parse_args()
print(args)
setup_seed(args.seed)
def trainer(args, model, train_loader, optimizer, epoch, total_epoch):
    loss_epoch = []
    model.train()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        img, poi, human, data_mean = batch[1].to(args.device).float(),batch[2].to(args.device).float(),batch[3].to(args.device).float(),batch[4].to(args.device).float()
        loss = model(img, poi, human, data_mean)

        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())
    print(f"TrainEpoch [{epoch}/{total_epoch}\t train_loss_epoch:{np.mean(loss_epoch)}")
    return np.mean(loss_epoch)

city = args.city


start_time =  time.time()

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/"+ args.traindata +'_' + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


data_process_time =  time.time()
data_process_used = data_process_time-start_time



train_loader, valid_loader, test_loader = get_dataloader(
    batchsize=config["train"]["batch_size"],
    seed=args.seed,
    city = city,
)


diffusion = create_diffusion(timestep_respacing="", diffusion_steps = config["diffusion"]["num_steps"])
model = CSDI_Value(config, diffusion, args.device).to(args.device)


dm_mean, dm_std = compute_target_stats(train_loader, args.device)
model.diffmodel.pretrained_contrast_model.set_target_stats(dm_mean, dm_std)
print("target mean:", dm_mean.tolist(), " std:", dm_std.tolist())

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_trainable_params}")


train(
    model,
    config["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=foldername,
)


trainings_time =  time.time()
training_used = trainings_time-start_time

model.load_state_dict(torch.load(foldername + "/model.pth"))
evaluate(model, valid_loader, nsample=args.nsample, scaler=1, foldername=foldername)

inferring_time =  time.time()
inferring_used = inferring_time - trainings_time

print('训练时长',training_used)
print('推断时长', inferring_used)