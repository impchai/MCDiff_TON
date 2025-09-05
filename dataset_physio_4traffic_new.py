import os

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import pandas as pd



#计算图像集合的均值和方差
def compute_mean_std(img_dir):
    """
    计算数据集的每个通道的均值和标准差
    """
    image_files = sorted(os.listdir(img_dir))
    means = []
    stds = []

    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        image = Image.open(img_path).convert('RGB')  # 打开并转换为 RGB 格式
        image = np.array(image) / 255.0  # 将像素值归一化到 [0, 1]

        # 分别计算 R、G、B 通道的均值和标准差
        means.append(image.mean(axis=(0, 1)))  # (H, W, C) -> (C,)
        stds.append(image.std(axis=(0, 1)))

    # 将所有图像的均值和标准差合并，计算整体均值和标准差
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std

def data_input(city):

    img_dir = './data_prepare_{}/e_fig/'.format(city)

    mean, var = compute_mean_std(img_dir)

    transform = transforms.Compose([
        transforms.ToPILImage(),            # 将 NumPy 数组转换为 PIL 图像
        transforms.Resize((224, 224), interpolation=Image.LANCZOS),      # 调整图片大小
        transforms.ToTensor(),              # 转换为张量
        transforms.Normalize(mean, var)  # 归一化 [-1, 1]
    ])
    # 读取并预处理所有图像
    images = []
    image_files = sorted(os.listdir(img_dir))  # 获取并排序所有 PNG 文件

    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)# 打开并转换为 RGB 格式
        image = transform(image)  # 应用 transform 进行预处理
        images.append(image)
    images = torch.stack(images)  # 将所有图像堆叠成一个张量

    observed_traffic = np.load('./data_prepare_{}/final_image_traffic_users.npz'.format(city))['image_traffic'][:,:168]
    observed_users = np.load('./data_prepare_{}/final_image_traffic_users.npz'.format(city))['image_users'][:, :168]


    constant_indices = []

    for i in range(observed_traffic.shape[0]):
        if np.all(observed_traffic[i] == observed_traffic[i, 0]):  # 如果序列中的所有元素与第一个元素相同
            constant_indices.append(i)
    # 删除常数序列
    observed_traffic = np.delete(observed_traffic, constant_indices, axis=0)
    observed_users = np.delete(observed_users, constant_indices, axis=0)



    observed_traffic = observed_traffic/observed_traffic.max()

    observed_users = observed_users / observed_users.max()

    indices_to_keep = [i for i in range(images.shape[0]) if i not in constant_indices]
    images = torch.index_select(images, 0, torch.tensor(indices_to_keep))

    df = pd.read_csv(
        './data_prepare_{}/poi_category_matrix_per_satellite_image2.txt'.format(city),
        encoding='utf-8',
        delim_whitespace=True,
    )

    poi_data = df.iloc[:, 1:].values

    poi_data = np.delete(poi_data, constant_indices, axis=0)
    poi_data = poi_data/np.max(poi_data)
    with open('./data_prepare_{}/sat_and_pop3.pkl'.format(city), 'rb') as f:
        data = pickle.load(f)
    human_data = np.array(list(data.values()))[:,0]
    human_data = np.delete(human_data, constant_indices, axis=0)

    human_data = human_data/np.max(human_data)
    return observed_traffic[100:500],observed_users[100:500], images[100:500], poi_data[100:500], human_data[100:500]



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


    traffic_data, user_data, img, poi, human = data_input(city)  # 从 data_filter 获取原始数据

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return  train_loader, test_loader, val_loader


