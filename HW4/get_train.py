import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image
#from torchvision.transforms import Compose, ToTensor
#from dd import DatasetFromFolderEval
image_dir = "./Dataset/training_hr_images"

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img


def get_train(image_dir):
    path = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
    return path


def generate_training_data():
	image_filenames = get_train(image_dir)
	i=0
	for filepath in image_filenames:
		i+=1
		filename = os.path.basename(filepath)
		filename, form = os.path.splitext(filename)
		im = load_img(filepath)
		im.save("./train_augmentation/"+filename+form)
		## Flip
		lr = im.transpose(Image.FLIP_LEFT_RIGHT)
		lr.save("./train_augmentation/"+filename+"_lr"+form)
		tb = im.transpose(Image.FLIP_TOP_BOTTOM)
		tb.save("./train_augmentation/"+filename+"_tb"+form)

		lr_tb = lr.transpose(Image.FLIP_TOP_BOTTOM)
		lr_tb.save("./train_augmentation/"+filename+"_lr_tb"+form)
		
		## Rotate
		rot90 = im.transpose(Image.ROTATE_90)
		rot90.save("./train_augmentation/"+filename+"_90"+form)
		rot270 = im.transpose(Image.ROTATE_270)
		rot270.save("./train_augmentation/"+filename+"_270"+form)
		
		## FR
		tbrot90 = rot90.transpose(Image.FLIP_TOP_BOTTOM) 
		tbrot90.save("./train_augmentation/"+filename+"_tb90"+form)
		tbrot270 = rot270.transpose(Image.FLIP_TOP_BOTTOM) 
		tbrot270.save("./train_augmentation/"+filename+"_tb270"+form)
		
		print(str(i)+" "+filename+" done!")

if __name__ == '__main__':
	generate_training_data()
