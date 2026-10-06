from PIL import Image
from torch.utils.data import Dataset
from data import util as Util
import torchvision.transforms as transforms
import random
import numpy as np
class UIEDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.data_len = data_len
        self.split = split

        self.input_path = Util.get_paths_from_images('{}/input_{}'.format(dataroot, resolution))
        self.target_path = Util.get_paths_from_images('{}/target_{}'.format(dataroot, resolution))

        self.dataset_len = len(self.target_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        target = Image.open(self.target_path[index]).convert("RGB")
        input = Image.open(self.input_path[index]).convert("RGB")

        [input, target] = Util.transform_augment([input, target], split=self.split, min_max=(-1, 1))

        return input,target

class UIEDataset_test(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.data_len = data_len
        self.split = split

        self.input_path = Util.get_paths_from_images('{}/input_{}'.format(dataroot, resolution))
        self.target_path = Util.get_paths_from_images('{}/target_{}'.format(dataroot, resolution))

        self.dataset_len = len(self.target_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        self.discrete_angles=[0,90, 180, 270,0,0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        target = Image.open(self.target_path[index]).convert("RGB")
        input = Image.open(self.input_path[index]).convert("RGB")
        angle = random.choice(self.discrete_angles)  # 从指定角度中随机选一个
        input = input.rotate(angle)# expand=True, fillcolor=(0, 0, 0)
        target = target.rotate(angle)
        [input, target] = Util.transform_augment([input, target], split=self.split, min_max=(-1, 1))

        return input,target

class UIEDatasetsample(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.data_len = data_len
        self.split = split

        self.input_path = Util.get_paths_from_images('{}/input_{}'.format(dataroot, resolution))
        self.target_path = Util.get_paths_from_images('{}/target_{}'.format(dataroot, resolution))

        self.dataset_len = len(self.target_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        target = Image.open(self.target_path[index]).convert("RGB")
        input = Image.open(self.input_path[index]).convert("RGB")

        [input, target] = Util.transform_augment([input, target], split=self.split, min_max=(-1, 1))

        #return input,target
        if self.split == 'train':
            return {'target': target, 'input': input, 'Index': index}
        else:
            return input,index


class UIEDatasetunpaire(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.data_len = data_len
        self.split = split

        self.input_path = Util.get_paths_from_images('{}/input_{}'.format(dataroot, resolution))
        self.candidate_path = Util.get_paths_from_images('{}/target_{}'.format(dataroot, resolution))

        self.dataset_len = len(self.candidate_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        candidate = Image.open(self.candidate_path[index]).convert("RGB")
        input = Image.open(self.input_path[index]).convert("RGB")
        
        strong_data =input
        
        [input, candidate,strong_data] = Util.transform_augment([input, candidate,strong_data], split=self.split, min_max=(-1, 1))
        p_name = self.candidate_path[index]
        return candidate,input,strong_data,p_name
       








    



    
