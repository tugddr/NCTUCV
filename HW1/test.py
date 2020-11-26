import torch                                                                                                                                                                                                
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torch.autograd import Variable
import dataprocess as Dp
import model as Model
import numpy as np
import csv
import os

dir = '../data/'
train_dir = os.path.join(dir, 'training_data/training_data')
test_dir = os.path.join(dir, 'testing_data/testing_data')
labelcsv = os.path.join(dir, 'training_labels.csv')

batch_size = 60
num_classes = 196

def test(model, test_data, labels):
	model.eval()
	test_data_size = len(test_data)
	
	outputs = []
	outputs.append(['id','label'])
	for i in range(0, len(test_data)):
		if i%100==0:
			print(i)
		img = Dp.default_loader(test_data[i])
		img = Dp.image_transforms['test'](img).float()
		img = img.unsqueeze_(0)
		img = Variable(img)
		img = img.to(device)
		output = model(img)
		ret, pred = torch.max(output, 1)
		
		filename = test_data[i].split('/')[-1]
		filename = filename.split('.')[0]
		label = ""
		for key in labels:
			if pred.item()==labels[key]:
				label = key
				break;

		outputs.append([filename,label])
	print("Finished!")
	with open('result/hw1.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for row in outputs:
			writer.writerow(row)

if __name__ == '__main__' :
	device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
	model = torch.load('models/hw1_model4_81.pt')
	model.to(device)
	model.eval()
	
	test_data = Dp.get_img_file(test_dir)	
	data, labels = Dp.DealwithData(labelcsv)

	test(model, test_data, labels)
