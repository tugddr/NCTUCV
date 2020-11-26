import torch                                                                                                                                                                                                
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import dataprocess as Dp
import model as Model
import numpy as np
import matplotlib.pyplot as plt
import os

dir = '../data/'
train_dir = os.path.join(dir, 'training_data/training_data')
#test_dir = os.path.join(dir, 'testing_data/testing_data')
labelcsv = os.path.join(dir, 'training_labels.csv')

batch_size = 64
num_classes = 196

def train(model, train_data, train_data_size, valid_data, valid_data_size, loss_function, optimizer, epochs):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	history = []
	best_acc = 0.0
	best_epoch = 0

	for epoch in range(epochs):
		epoch_start = time.time()
		print("Epoch: {}/{}".format(epoch+1, epochs))

		model.train()

		train_loss = 0.0
		train_acc = 0.0
		valid_loss = 0.0
		valid_acc = 0.0

		for i, (inputs, labels) in enumerate(train_data):
			inputs = inputs.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()

			outputs = model(inputs)

			loss = loss_function(outputs, labels)

			loss.backward()

			optimizer.step()

			train_loss += loss.item() * inputs.size(0)

			ret, predictions = torch.max(outputs.data, 1)
			correct_counts = predictions.eq(labels.data.view_as(predictions))

			acc = torch.mean(correct_counts.type(torch.FloatTensor))

			train_acc += acc.item() * inputs.size(0)

		with torch.no_grad():
			model.eval()

			for j, (inputs, labels) in enumerate(valid_data):
				inputs = inputs.to(device)
				labels = labels.to(device)

				outputs = model(inputs)

				loss = loss_function(outputs, labels)

				valid_loss += loss.item() * inputs.size(0)

				ret, predictions = torch.max(outputs.data, 1)
				correct_counts = predictions.eq(labels.data.view_as(predictions))

				acc = torch.mean(correct_counts.type(torch.FloatTensor))

				valid_acc += acc.item() * inputs.size(0)

		avg_train_loss = train_loss/train_data_size
		avg_train_acc = train_acc/train_data_size

		avg_valid_loss = valid_loss/valid_data_size
		avg_valid_acc = valid_acc/valid_data_size

		history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

		if best_acc < avg_valid_acc:
			best_acc = avg_valid_acc
			best_epoch = epoch + 1
			torch.save(model, 'models/hw1_best_model.pt')

		epoch_end = time.time()

		print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
		epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
		))
		print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

		torch.save(model, 'models/'+'hw1'+'_model4_'+str(epoch+1)+'.pt')
	return model, history	
	

if __name__ == '__main__':
	model = Model.getModel()
	###
	###		Get cooresponding LABEL
	###
	_data, _labels = Dp.DealwithData(labelcsv)
	data_label = Dp.getTrainPair(train_dir, _data, _labels)
	###
	###		Seperate Train and Valid
	###
	_train = data_label[0:11001]
	_valid = data_label[11001:-1]

	data = dict()
	
	data['train'] = Dp.MyDataset(_train)
	data['valid'] = Dp.MyDataset(_valid)
	
	train_data_size = len(data['train'])
	valid_data_size = len(data['valid'])

	train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
	valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)

	loss_func = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	###
	###		Training
	###
	final_model, history = train(model, train_data, train_data_size, valid_data, valid_data_size, loss_func, optimizer, 1000)
	torch.save(history, 'models/hw1_history4.pt')
	torch.save(final_model, 'models/hw1_final_model4.pt')
