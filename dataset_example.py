#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:37:59 2020

@author: biototem
"""
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image
import random

class simple_dataset(data.Dataset):
    def __init__(self, data_list=None,label_list = None,shuffle = False,transform=None):
        if data_list is None or label_list is None:
            raise ValueError('data_list and label_list should not be None!')
        
        if str(type(data_list)) == "<class 'list'>" or str(type(label_list)) == "<class 'list'>":
            raise ValueError('data_list and label_list should not be None!')
            
        if len(data_list) == 0 or len(label_list) == 0:
            raise ValueError('the length of data_list and label_list should not be Zero!')
            
        if len(data_list) != len(label_list):
            raise ValueError('the length of data_list should be equal to the length of label_list!')
    
        self.data_source = [(data_list[x], label_list[x]) for x in range(len(data_list))]
        # 这一步相当于简易版的make_dateset方法,注意到data_source的list中每一个内容都是一个二元元组，前者是data，后者是label

        self.shuffle = shuffle
        self.transform = transform
        
    def shuffletraindata(self):
        if self.shuffle:
            self.data_source = random.sample(self.data_source, len(self.data_source))
            # 使用random.sample代替官方内置方法中的torch.randint()/torch.randperm().tolist()
    
    def __getitem__(self,index):
        img_file_path,label = self.data_source[index]
        img = Image.open(img_file_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data_source)
    
    
    
            
        