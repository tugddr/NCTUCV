import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import glob
import os
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from engine import train_one_epoch, evaluate
import utils
import transforms as T

dir="train_images"

def get_transform(train):
	transforms = []
	# converts the image, a PIL image, into a PyTorch Tensor
	transforms.append(T.ToTensor())
	if train:
		# during training, randomly flip the training images
		# and ground-truth for data augmentation
		transforms.append(T.RandomHorizontalFlip(0.5))
		transforms.append(T.RandomResize(0.5))
	#transforms.append(T.ToTensor())
	return T.Compose(transforms)

class MyDataset(torch.utils.data.Dataset):
	def __init__(self, root, train=True):
		self.root = root
		self.transforms = get_transform(train)
		# load all image files, sorting them to
		# ensure that they are aligned
		self.coco = COCO("dataset/pascal_train.json")
		self.coco.cats
		self.imgs = list(self.coco.imgs.keys())
	
	def __getitem__(self, idx):
		# load images ad masks
		img_id = self.imgs[idx]
		img_info = self.coco.imgs[img_id]
		img_path = os.path.join(self.root, img_info['file_name'])
		img = cv2.imread(img_path)[:,:,::-1].copy()
		
		annids = self.coco.getAnnIds(imgIds=img_id)
		anns = self.coco.loadAnns(annids)
		num_objs = len(annids)
		masks=[]
		for i in range(num_objs):
			 masks.append(self.coco.annToMask(anns[i]))
		#mask=np.array(mask)
		
		boxes = []
		labels = []
		areas = []
		iscrowd = []
		for i in range(num_objs):
			pos = anns[i]['bbox']
			xmin = pos[0]
			xmax = pos[0]+pos[2]
			ymin = pos[1]
			ymax = pos[1]+pos[3]
			boxes.append([xmin, ymin, xmax, ymax])
			labels.append(anns[i]['category_id'])	
			areas.append(anns[i]['area'])
			iscrowd.append(anns[i]['iscrowd'])
		
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		# there is only one class
		labels = torch.as_tensor(labels, dtype=torch.int64)
		masks = torch.as_tensor(masks, dtype=torch.uint8)
		
		img_id = torch.as_tensor([img_id])
		areas = torch.as_tensor(areas)
		iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)
		
		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["masks"] = masks
		target["image_id"] = img_id
		target["area"] = areas
		target["iscrowd"] = iscrowd
		if self.transforms is not None:
			img, target = self.transforms(img, target)
		
		return img, target
	
	def __len__(self):
		return len(self.imgs)
