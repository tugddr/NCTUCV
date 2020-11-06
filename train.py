import torch                                                                                                                                                                                                
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import dataprocess as Dp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

dir = '../data/'
train_dir = os.path.join(dir, 'training_data/training_data')
test_dir = os.path.join(dir, 'testing_data/testing_data')
labelcsv = os.path.join(dir, 'training_labels.csv')

def train():

if __name__ == '__main__':
	_data, _labels = Dp.DealwithData(labelcsv)
	train_data = Dp.getTrainPair(train_dir, _data, _labels)
	dataset = Dp.MyDataset(train_data)

	#train(model, )

