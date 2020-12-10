import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from torchvision import transforms

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import utils
import transforms as T
import process as P
import model as Model

def main(args):
	os.environ["CUDA_VISIBLE_DEVICES"] = "6, 8"
	utils.init_distributed_mode(args)
	print(args)
	
	device = torch.device(args.device)
	# Data loading code
	print("Loading data")
	dataset=P.MyDataset('train_images',True)
	dataset_valid=P.MyDataset('train_images',False)
	# split the dataset in train and test set
	torch.manual_seed(1)
	indices = torch.randperm(len(dataset)).tolist()
	dataset = torch.utils.data.Subset(dataset, indices[:-50])
	dataset_valid = torch.utils.data.Subset(dataset_valid, indices[-50:])
	num_classes=21
	# define training and validation data loaders
	#data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
	#data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
	#dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True))
	#dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False))
	
	print("Creating data loaders")
	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
		valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_valid)
	else:
		train_sampler = torch.utils.data.RandomSampler(dataset)
		valid_sampler = torch.utils.data.SequentialSampler(dataset_valid)
		
	if args.aspect_ratio_group_factor >= 0:
		group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
		train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
	else:
		train_batch_sampler = torch.utils.data.BatchSampler(
			train_sampler, args.batch_size, drop_last=True)
		
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
		collate_fn=utils.collate_fn)
	
	data_loader_valid = torch.utils.data.DataLoader(
		dataset_valid, batch_size=1,
		sampler=valid_sampler, num_workers=args.workers,
		collate_fn=utils.collate_fn)
	
	print("Creating model")
	model = Model.get_model(num_classes)
	#model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained)
	model.to(device)
	
	model_without_ddp = model
	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		model_without_ddp = model.module
		
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(
		params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	
	# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

	if args.resume:
		checkpoint = torch.load(args.resume, map_location='cpu')
		model_without_ddp.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
	
	if args.test_only:
		evaluate(model, data_loader_test, device=device)
		return
		
	print("Start training")
	
	start_time = time.time()
	for epoch in range(args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)
		train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
		lr_scheduler.step()
		if args.output_dir:
			model_with_ddp = model.module
			utils.save_on_master({
				'model': model_without_ddp.state_dict(),
				'optimizer': optimizer.state_dict(),
				'lr_scheduler': lr_scheduler.state_dict(),
				'args': args},
				os.path.join(args.output_dir, 'model5_{}.pth'.format(epoch)))
		
		# evaluate after every epoch
		evaluate(model, data_loader_valid, device=device)
	
	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training')

    parser.add_argument('--data-path', default='train_images', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.0075, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='models', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
