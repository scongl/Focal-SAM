import torch
import random
import numpy as np
import os, sys
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image


class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # Actually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch

class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label

class iNaturalistDataLoader(DataLoader):
    """
    iNaturalist Data Loader
    """
    category_method = "name"
    categories_json = "./data_txt/iNaturalist18/categories.json"

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, resolution=224, retain_epoch_size=True, 
                 train_txt= './data_txt/iNaturalist18/iNaturalist18_train.txt', 
                 eval_txt= './data_txt/iNaturalist18/iNaturalist18_val.txt'):
        # train_trsfm = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
        # ])
        # test_trsfm = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
        # ])

        id2cname, cname2lab = self.read_category_info()

        self.names = []
        self.labels = []

        if training:
            self.txt = train_txt
        else:
            self.txt = eval_txt

        with open(self.txt) as f:
            for line in f:
                _name = id2cname[int(line.split()[1])]
                self.names.append(_name)
                self.labels.append(cname2lab[_name])

        self.classnames = self.get_classnames()

        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        normalize = transforms.Normalize(mean=mean, std=std)
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        # test_trsfm = transforms.Compose([
        #     transforms.Resize(resolution),
        #     transforms.CenterCrop(resolution),
        #     transforms.ToTensor(),
        #     normalize,
        # ])

        test_trsfm = transforms.Compose([
            transforms.Resize(resolution * 8 // 7),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ])

        if training:
            dataset = LT_Dataset(data_dir, train_txt , train_trsfm)
            val_dataset = LT_Dataset(data_dir, eval_txt, test_trsfm)
        else: # test
            dataset = LT_Dataset(data_dir, eval_txt, test_trsfm)
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        self.n_samples = len(self.dataset)

        num_classes = 8142

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None
        
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True
        }

        self.val_init_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': True
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

    @classmethod
    def read_category_info(self):
        with open(self.categories_json, "rb") as file:
            category_info = json.load(file)
        
        id2cname = {}
        for id, info in enumerate(category_info):
            cname = info[self.category_method]
            id2cname[id] = cname

        cnames_unique = sorted(set(id2cname.values()))
        cname2lab = {c: i for i, c in enumerate(cnames_unique)}
        return id2cname, cname2lab

    def get_classnames(self):
        container = set()
        for label, name in zip(self.labels, self.names):
            container.add((label, name))
        mapping = {label: classname for label, classname in container}
        classnames = [mapping[label] for label in sorted(mapping.keys())]
        return classnames

    def split_validation(self):
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.val_init_kwargs)
