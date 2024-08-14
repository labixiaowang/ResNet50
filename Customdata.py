import os
from PIL import Image
from torch.utils.data import Dataset
# 自定义数据集
class CustomImageDataset(Dataset):
    def __init__(self, data_file_path, transform=None):
        self.data = []
        #a = 1
        with open(data_file_path, 'r') as file:
            for line in file.readlines():
                img_path,label = line.strip().split(',')
                self.data.append((img_path, int(label)))  # 假设标签是整数

        self.transform = transform

    def __len__(self):
        """
        返回数据集中样本的数量
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根据索引获取单个样本数据
        """
        try:
            img_path, label = self.data[index]
            # 打开图像文件
            img = Image.open(os.path.join(img_path)).convert('RGB')  # 替换为你的数据根目录
            # 对图像进行预处理（如果提供了transform参数）
            if self.transform:
                img = self.transform(img)
                #print(f"img_shape:{img.shape}")

            return img, label  # 返回处理后的图像和对应的标签

        except Exception as e:
            print(f"Error loading image file {img_path}: {e}")
            return None, None  # 返回空值，或者其他你认为合适的处理结果