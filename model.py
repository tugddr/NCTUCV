import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import os

def getModel():
	model = models.resnet50(pretrained=True)
	
	#for param in model.parameters():
	#	param.requires_grad = False

	#model = nn.Sequential(*list(res.children())[:-2])
	fc_inputs = model.fc.in_features
	model.fc = nn.Linear(fc_inputs,196)

	model = model.to('cuda:0')
	return model
