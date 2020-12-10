import datetime
import os
import time
import cv2
import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from itertools import groupby
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
#from utils import binary_mask_to_rle
import transforms as T
import process as P
import model as Model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import glob
from pycocotools import mask as m

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
dir = "test_images"

def binary_mask_to_rle(binary_mask):
	rle = {'counts': [], 'size': list(binary_mask.shape)}
	counts = rle.get('counts')
	for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
		if i == 0 and value >= 1:
			counts.append(0)
		counts.append(len(list(elements)))

	return rle

def main():
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Data loading code
	print("Loading data")
	#dataset_test=P.MyDataset('test_images',False)
	num_classes=21
	print("Creating test data")
	#data_loader_test = torch.utils.data.DataLoader(
	#	dataset_test, batch_size=1,
	#	collate_fn=utils.collate_fn)
	
	image_files = []
	os.chdir(dir)
	
	for filename in os.listdir(os.getcwd()):
		if filename.endswith(".jpg"):
			image_files.append(dir + '/' + filename)
	os.chdir("..")	
	print("Creating model")
	model = Model.get_model(num_classes)
	model.eval()
	model.to(device)
	checkpoint = torch.load(os.path.join('models/model5_1.pth'), map_location='cpu')
	model.load_state_dict(checkpoint['model'])

	print("Start testing")

	cocoGt = COCO("dataset/test.json")
	#from utils import binary_mask_to_rle
	coco_dt = []
	start_time = time.time()

	yy = 0
	for imgid in cocoGt.imgs:
	#	if yy==1:
	#		break
		yy = yy+1
		print('[{}/{}]'.format(yy,len(cocoGt.imgs)))
		#img = Image.open("test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name']).convert('RGB')
		img = cv2.imread("test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1].copy()
		trn = transforms.ToTensor()
		img = trn(img)
		#print(img.size)
		img = img.unsqueeze_(0)
		#img = Variable(img)
		outputs = model(img.to(device))
		outputs = outputs[0]

		n_instances = len(outputs['scores'])
		if len(outputs['labels']) > 0: # If any objects are detected in this image
			for i in range(n_instances): # Loop all instances
				# save information of the instance in a dictionary then append on coco_dt list
				pred = {}
				pred['image_id'] = imgid
				pred['category_id'] = int(outputs['labels'][i])
				id, mask = imgid,outputs['masks'][i][0].detach()
				mask = mask.tolist()
				jj=0
				for j in mask:
					kk=0
					for k in j:
						if k>0:
							mask[jj][kk]=1
						kk = kk+1
					jj = jj+1
				mask = np.asarray(mask, dtype=np.uint8)

				b_a = np.asfortranarray(np.array(mask==1, dtype=np.bool))
				mm = m.encode(b_a)['counts'].decode("utf-8")

				pred['segmentation'] = {'counts': str(mm), 'size': list(mask.shape)}#binary_mask_to_rle(mask)
				#sss = []
				#for bb in range(len(pred['segmentation']['counts'])):
				#	sss.append("{0:b}".format(pred['segmentation']['counts'][bb]))
			
				#print(list(map(int, ''.join(sss))))
				#print(m.encode(np.array([list(map(int, ''.join(sss)))], dtype=np.bool))['counts'].decode("utf-8"))
				#print(bytes(mm).decode("utf-8"))
				pred['score'] = float(outputs['scores'][i])
				coco_dt.append(pred)
		
	with open("0616080_6.json", "w") as f:
	    json.dump(coco_dt, f)
	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print('Testing time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
