import os

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from PIL import Image
import numpy as np



def compute_mean_std(img_dir):

    image_files = sorted(os.listdir(img_dir))
    means = []
    stds = []

    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image) / 255.0  #

        means.append(image.mean(axis=(0, 1)))  # (H, W, C) -> (C,)
        stds.append(image.std(axis=(0, 1)))

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std



def data_input(city):
    # 新的保存路径
    save_path = f'./data_prepare_{city}/selected_data.npz'
    selected_img_dir = f'./data_prepare_{city}/selected_images/'

    data = np.load(save_path, allow_pickle=True)
    observed_traffic = data['observed_traffic']
    observed_users = data['observed_users']
    poi_data = data['poi_data']
    human_data = data['human_data']

    mean, var = compute_mean_std(selected_img_dir)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean, var)
    ])


    image_files = sorted(os.listdir(selected_img_dir))
    images = []
    for img_file in image_files:
        img_path = os.path.join(selected_img_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        image = transform(image)
        images.append(image)
    images = torch.stack(images)

    return observed_traffic, observed_users, images, poi_data, human_data

class MyDataset(Dataset):
    def __init__(self, data, image, poi, human, transform = None):
        self.data = data
        self.data_mean = data.mean(axis=-1)
        self.image = image
        self.poi = poi
        self.human = human


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        batch_data = self.data[idx]
        batch_image = self.image[idx]
        batch_poi = self.poi[idx]
        batch_human = self.human[idx]
        batch_mean = self.data_mean[idx]


        return batch_data, batch_image, batch_poi, batch_human, batch_mean


def get_dataloader(batchsize,seed, city):

    np.random.seed(seed)


    traffic_data, user_data, img, poi, human = data_input(city)

    train_idx, temp_idx = train_test_split(np.arange(len(traffic_data)), test_size=0.3, random_state=12)

    data_scaled_traffic = traffic_data
    data_scaled_u = user_data


    scaled_traffic = np.expand_dims(data_scaled_traffic, axis=1)
    scaled_u  = np.expand_dims(data_scaled_u, axis=1)
    data_scaled = np.concatenate([scaled_traffic, scaled_u], axis=1)  # 在第0维拼接


    dataset = MyDataset(data_scaled, img, poi, human)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, temp_idx)
    test_dataset = Subset(dataset, temp_idx)
    batch_size = batchsize

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return  train_loader, test_loader, val_loader


