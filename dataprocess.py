import torch                                                                                                                                                                                                
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time

import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image
import os

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

def DealwithData(filepath):
	csvfile = open(filepath, newline='')
	data = csv.reader(csvfile)
	data = list(data)
	data.pop(0)

	labels = dict()
	num = 0

	for img, label in data:
		if label in labels:
			continue
		else:
			labels[label] = num
			num = num+1
	
	return data, labels
###  Deal with test 
def get_img_file(filepath):
    imagelist = []
    for parent, dirnames, filenames in os.walk(filepath):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist

def getTrainPair(img_dir='../data/training_data/training_data', data=None, labels=None):
	train_data = [ [os.path.join(img_dir+'/', img+'.jpg'), labels[label] ] for img, label in data ]
	return train_data

def default_loader(fp):
	return Image.open(fp).convert('RGB')

class MyDataset(Dataset):
	def __init__(self, train_data, transform=image_transforms['train'], loader=default_loader):
		super(MyDataset, self).__init__()
		self.imgs = train_data
		self.transform = transform
		self.loader=loader
	def __getitem__(self, index):
		input = self.loader(self.imgs[index][0])
		label = self.imgs[index][1]
		if self.transform:
			input = self.transform(input)
		
		return input, label
	def __len__(self):
		return len(self.imgs)
