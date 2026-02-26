import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

def load_image(img_path):
    img = Image.open(img_path)
    return np.array(img)

def load_images(images_dir, images_list):
    images = {}

    def load_image_for_key(sub_dir):
        base_path = os.path.join(images_dir, sub_dir)
        try:
            img = load_image(base_path + '.jpg')
        except FileNotFoundError: 
            try:
                img = load_image(base_path + '.JPG')
            except FileNotFoundError:
                print(f"Image not found for {sub_dir} (.jpg or .JPG)")
                return
        images[sub_dir] = img

    with ThreadPoolExecutor() as executor:
        for sub_dir_list in images_list:
            sub_dir = sub_dir_list
            executor.submit(load_image_for_key, sub_dir)

    return images

class Aircraft_Dataset(Dataset):
    def __init__(self, pattern, device, load_cache=True, cut_box=True):
        super(Aircraft_Dataset, self).__init__()
        self.load_cache = load_cache
        self.cut_box = cut_box
        self.device = device

        self.images_dir = f"datasets/RawData/Aircraft/images"
        with open(f"datasets/SupplementaryData/Aircraft/{pattern}.txt", 'r') as file:
            lines = file.readlines()
        self.label = {}
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            self.label[parts[0]] = parts[1]
        self.length = len(self.label)
        self.images_list = list(self.label.keys())
        if load_cache:
            self.images = load_images(self.images_dir, self.images_list)

        with open(f"datasets/SupplementaryData/Aircraft/fine_class.txt", 'r') as file:
            lines = file.readlines()
        self.label_index = {}
        for i, line in enumerate(lines):
            parts = line.strip()
            key = parts
            self.label_index[key] = i

        if cut_box:
            with open(f"datasets/SupplementaryData/Aircraft/box.txt", 'r') as file:
                lines = file.readlines()
            self.box = {}
            for line in lines:
                parts = line.strip().split()
                key = parts[0]
                values = list(map(int, parts[1:]))
                self.box[key] = values

        self.pattern = pattern
        self.to_tensor = transforms.ToTensor()
        input_size = 518
        self.resize = transforms.Resize([input_size, input_size])
        self.train_preprocess = transforms.Compose([
            transforms.Pad(30),
            transforms.RandomCrop([input_size, input_size])
        ])
        self.test_preprocess = transforms.Compose([
            transforms.Pad(30),
            transforms.CenterCrop([input_size, input_size])
        ])
        
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0)
        self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.random_rotation = transforms.RandomRotation(12, expand=False)
        self.random_erasing = transforms.RandomErasing(p=0.9, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random')
        self.random_grayscale = transforms.RandomGrayscale(p=0.1)
        self.gaussian_blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.load_cache:
            filename = self.images_list[index]
            image = self.images[filename]
            label = self.label[filename]
            label_index = torch.tensor([self.label_index[label]]).to(self.device)
        else:
            filename = self.images_list[index]
            images_dir = os.path.join(self.images_dir, filename + '.jpg')
            image = Image.open(images_dir)
            label = self.label[filename]
            label_index = torch.tensor([self.label_index[label]]).to(self.device)
            image = np.array(image)

        if self.cut_box:
            x_min, y_min, x_max, y_max = self.box[filename]
            image = image[y_min:y_max, x_min:x_max]

        image = self.to_tensor(image).to(self.device)
        if image.shape[0]==1:
            image = image.repeat(3, 1, 1)
        image = self.resize(image)

        if self.pattern == 'train':
            image = self.train_preprocess(image)
            image = self.color_jitter(image)
            image = self.gaussian_blur(image)  
            image = self.normalize(image)
            image = self.random_horizontal_flip(image)
            image = self.random_rotation(image)
            image = self.random_erasing(image) 
            image = self.random_grayscale(image)
        else:
            image = self.test_preprocess(image)
            image = self.normalize(image)

        batch = {
            'img': image,
            'label_index': label_index,
            'image_index': torch.tensor([index]).to(self.device),
        }
        return batch

class CUB_Dataset(Dataset):
    def __init__(self, pattern, device, load_cache=True, cut_box=True):
        super(CUB_Dataset, self).__init__()
        self.load_cache = load_cache
        self.cut_box = cut_box
        self.device = device

        self.images_dir = f"autodl-fs/datasets/RawData/CUB/images"
        with open(f"autodl-fs/datasets/SupplementaryData/CUB/{pattern}.txt", 'r') as file:
            lines = file.readlines()
        self.images_list = []
        self.label_index = []
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            self.images_list.append(parts[0])
            self.label_index.append(int(parts[1]))
        self.length = len(self.images_list)
        if load_cache:
            self.images = load_images(self.images_dir, self.images_list)

        if cut_box:
            with open(f"autodl-fs/datasets/SupplementaryData/CUB/box.txt", 'r') as file:
                lines = file.readlines()
            self.box = {}
            for line in lines:
                parts = line.strip().split()
                key = parts[0]
                values = list(map(int, parts[1:]))
                self.box[key] = values

        self.pattern = pattern
        self.to_tensor = transforms.ToTensor()
        input_size = 518
        self.resize = transforms.Resize([input_size, input_size])
        self.train_preprocess = transforms.Compose([
            transforms.Pad(30),
            transforms.RandomCrop([input_size, input_size])
        ])
        self.test_preprocess = transforms.Compose([
            transforms.Pad(30),
            transforms.CenterCrop([input_size, input_size])
        ])
        
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0)
        self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.random_rotation = transforms.RandomRotation(12, expand=False)
        self.random_erasing = transforms.RandomErasing(p=0.9, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random')
        self.random_grayscale = transforms.RandomGrayscale(p=0.1)
        self.gaussian_blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.load_cache:
            filename = self.images_list[index]
            image = self.images[filename]
            label_index = torch.tensor([self.label_index[index]]).to(self.device)
        else:
            filename = self.images_list[index]
            images_dir = os.path.join(self.images_dir, filename + '.jpg')
            image = Image.open(images_dir)
            label_index = torch.tensor([self.label_index[index]]).to(self.device)
            image = np.array(image)

        if self.cut_box:
            x_min, y_min, w, h = self.box[filename]
            image = image[y_min:y_min+h, x_min:x_min+w]

        image = self.to_tensor(image).to(self.device)
        if image.shape[0]==1:
            image = image.repeat(3, 1, 1)
        image = self.resize(image)

        if self.pattern == 'train':
            image = self.train_preprocess(image)
            image = self.color_jitter(image)
            image = self.gaussian_blur(image)  
            image = self.normalize(image)
            image = self.random_horizontal_flip(image)
            image = self.random_rotation(image)
            image = self.random_erasing(image) 
            image = self.random_grayscale(image)
        else:
            image = self.test_preprocess(image)
            image = self.normalize(image)

        batch = {
            'img': image,
            'label_index': label_index,
            'image_index': torch.tensor([index]).to(self.device),
        }
        return batch


class Car_Dataset(Dataset):
    def __init__(self, pattern, device, load_cache=True, cut_box=True):
        super(Car_Dataset, self).__init__()
        self.load_cache = load_cache
        self.cut_box = cut_box
        self.device = device

        self.images_dir = f"autodl-fs/datasets/RawData/Car/images"
        with open(f"autodl-fs/datasets/SupplementaryData/Car/{pattern}.txt", 'r') as file:
            lines = file.readlines()
        self.images_list = []
        self.label_index = []
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            self.images_list.append(parts[0])
            self.label_index.append(int(parts[1]))
        self.length = len(self.images_list)
        if load_cache:
            self.images = load_images(self.images_dir, self.images_list)

        if cut_box:
            with open(f"autodl-fs/datasets/SupplementaryData/Car/box.txt", 'r') as file:
                lines = file.readlines()
            self.box = {}
            for line in lines:
                parts = line.strip().split()
                key = parts[0]
                values = list(map(int, parts[1:]))
                self.box[key] = values

        self.pattern = pattern
        self.to_tensor = transforms.ToTensor()
        input_size = 518
        self.resize = transforms.Resize([input_size, input_size])
        self.train_preprocess = transforms.Compose([
            transforms.Pad(30),
            transforms.RandomCrop([input_size, input_size])
        ])
        self.test_preprocess = transforms.Compose([
            transforms.Pad(30),
            transforms.CenterCrop([input_size, input_size])
        ])
        
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0)
        self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.random_rotation = transforms.RandomRotation(12, expand=False)
        self.random_erasing = transforms.RandomErasing(p=0.9, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random')
        self.random_grayscale = transforms.RandomGrayscale(p=0.1)
        self.gaussian_blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.load_cache:
            filename = self.images_list[index]
            image = self.images[filename]
            label_index = torch.tensor([self.label_index[index]]).to(self.device)
        else:
            filename = self.images_list[index]
            images_dir = os.path.join(self.images_dir, filename + '.jpg')
            image = Image.open(images_dir)
            label_index = torch.tensor([self.label_index[index]]).to(self.device)
            image = np.array(image)

        if self.cut_box:
            x_min, y_min, x_max, y_max = self.box[filename]
            image = image[y_min:y_max, x_min:x_max]

        image = self.to_tensor(image).to(self.device)
        if image.shape[0]==1:
            image = image.repeat(3, 1, 1)
        image = self.resize(image)

        if self.pattern == 'train':
            image = self.train_preprocess(image)
            image = self.color_jitter(image)
            image = self.gaussian_blur(image)  
            image = self.normalize(image)
            image = self.random_horizontal_flip(image)
            image = self.random_rotation(image)
            image = self.random_erasing(image) 
            image = self.random_grayscale(image)
        else:
            image = self.test_preprocess(image)
            image = self.normalize(image)

        batch = {
            'img': image,
            'label_index': label_index,
            'image_index': torch.tensor([index]).to(self.device),
        }
        return batch

class Dogs_Dataset(Dataset):
    def __init__(self, pattern, device, load_cache=True, cut_box=True):
        super(Dogs_Dataset, self).__init__()
        self.load_cache = load_cache
        self.cut_box = cut_box
        self.device = device

        self.images_dir = f"autodl-fs/datasets/RawData/Dogs/images"
        with open(f"autodl-fs/datasets/SupplementaryData/Dogs/{pattern}.txt", 'r') as file:
            lines = file.readlines()
        self.images_list = []
        self.label_index = []
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            self.images_list.append(parts[0])
            self.label_index.append(int(parts[1]))
        self.length = len(self.images_list)
        if load_cache:
            self.images = load_images(self.images_dir, self.images_list)

        if cut_box:
            with open(f"autodl-fs/datasets/SupplementaryData/Dogs/box.txt", 'r') as file:
                lines = file.readlines()
            self.box = {}
            for line in lines:
                parts = line.strip().split()
                key = parts[0]
                values = list(map(int, parts[1:]))
                self.box[key] = values

        self.pattern = pattern
        self.to_tensor = transforms.ToTensor()
        input_size = 518
        self.resize = transforms.Resize([input_size, input_size])
        self.train_preprocess = transforms.Compose([
            transforms.Pad(30),
            transforms.RandomCrop([input_size, input_size])
        ])
        self.test_preprocess = transforms.Compose([
            transforms.Pad(30),
            transforms.CenterCrop([input_size, input_size])
        ])
        
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0)
        self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.random_rotation = transforms.RandomRotation(12, expand=False)
        self.random_erasing = transforms.RandomErasing(p=0.9, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random')
        self.random_grayscale = transforms.RandomGrayscale(p=0.1)
        self.gaussian_blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.load_cache:
            filename = self.images_list[index]
            image = self.images[filename]
            label_index = torch.tensor([self.label_index[index]]).to(self.device)
        else:
            filename = self.images_list[index]
            images_dir = os.path.join(self.images_dir, filename + '.jpg')
            image = Image.open(images_dir)
            label_index = torch.tensor([self.label_index[index]]).to(self.device)
            image = np.array(image)

        if self.cut_box:
            x_min, y_min, x_max, y_max = self.box[filename]
            image = image[y_min:y_max, x_min:x_max]

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        image = self.to_tensor(image).to(self.device)
        image = self.resize(image)

        if self.pattern == 'train':
            image = self.train_preprocess(image)
            image = self.color_jitter(image)
            image = self.gaussian_blur(image)  
            image = self.normalize(image)
            image = self.random_horizontal_flip(image)
            image = self.random_rotation(image)
            image = self.random_erasing(image) 
            image = self.random_grayscale(image)
        else:
            image = self.test_preprocess(image)
            image = self.normalize(image)

        batch = {
            'img': image,
            'label_index': label_index,
            'image_index': torch.tensor([index]).to(self.device),
        }
        return batch
    
def PACF_Collate(batch):
    img = []
    label_index = []
    image_index = []

    for idx, item in enumerate(batch):
        img.append(item["img"])
        label_index.append(item["label_index"])
        image_index.append(item["image_index"])

    batch = {
        'img': img,
        'label_index': torch.cat(label_index),
        'image_index': torch.cat(image_index),
    }

    return batch
    