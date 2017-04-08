"""
Transfer learning from ImageNet to PascalVOC classification.

This code is modified from 'Transfer Learning Tutorial' by
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_
"""
# License: BSD
# Author: Sasank Chilamkurthy, Hyeonwoo Noh

import argparse
import copy
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms

from src import datasets
from src import voc_tool

splits = ['train', 'val', 'test']

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def GetVocDatasets(params):
	voc_datasets = {split: datasets.VocClassification(params['voc_devkit_dir'],
		params['voc_version'], split, transform=data_transforms[split])
		for split in splits}	
	return voc_datasets

def GetVocLoaders(params, voc_datasets):
	voc_loaders = {split: torch.utils.data.DataLoader(voc_datasets[split],
		batch_size=params['batch_size'], shuffle=(split=='train'),
		num_workers=params['num_data_loading_workers'])
		for split in splits}
	return voc_loaders

def GetVocEvaluators(params):
	voc_evaluators = {split: voc_tool.VocEvaluateClassification(
		params['voc_devkit_dir'], params['voc_version'], split)
		for split in splits}
	return voc_evaluators

class SigmoidLinear(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(SigmoidLinear, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)
		self.sigmoid = nn.Sigmoid()

		linear_params = list(self.linear.parameters())
		linear_params[0].data.normal_(0, 0.01)
		linear_params[0].data.fill_(0)
		
	def forward(self, x):
		return self.sigmoid(self.linear(x))

def GetPretrainedModel(params, num_classes):
	if params['model'] == 'resnet18':
		model = models.resnet18(pretrained=True)
	elif params['model'] == 'resnet34':
		model = models.resnet34(pretrained=True)
	elif params['model'] == 'resnet50':
		model = models.resnet50(pretrained=True)
	elif params['model'] == 'resnet101':
		model = models.resnet101(pretrained=True)
	elif params['model'] == 'resnet152':
		model = models.resnet152(pretrained=True)
	else:
		raise ValueError('Unknown model type')
	num_features = model.fc.in_features
	model.fc = SigmoidLinear(num_features, num_classes)
	return model

def OptimizerScheduler(params, model, epoch):
	lr = params['initial_learning_rate'] * (0.1**(
		epoch // params['learning_rate_decay_epoch']))

	if epoch % params['learning_rate_decay_epoch'] == 0:
		print ('Learning rate is set to {}'.format(lr))

	optimizer = optim.SGD(model.parameters(), lr=lr,
		momentum=params['momentum'])
	return optimizer

def Test(params, model, voc_loaders, voc_evaluators, split):
	results = {}
	for i, data in enumerate(voc_loaders[split]):
		images = Variable(data[0].cuda())
		labels = Variable(data[1].cuda())
		image_ids = list(data[2])

		confidences = model(images)

		# Accumulate results
		for b, image_id in enumerate(image_ids):
			results[image_id] = confidences.data[b]

	evaluation_summarys = voc_evaluators[split].Evaluate(results)
	meanAP = evaluation_summarys['meanAP']
	print ('Evaluation [{}] MeanAP: {:.4f}'.format(split, meanAP))
	return evaluation_summarys

def Train(params, model, criterion, optim_scheduler, voc_loaders,
	voc_evaluators):
	best_model = model
	best_meanAP = 0.0
	best_epoch = 0
	best_summary = {}
	meanAP_historys = {'train': [], 'val': []}

	num_epochs = params['num_epochs']	
	for epoch in range(num_epochs):
		print ('Epoch {}/{}'.format(epoch+1, num_epochs))
		print ('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':
				optimizer = optim_scheduler(params, model, epoch)

			running_loss = 0.0
			num_iterations = len(voc_loaders[phase])
			results = {}
			for i, data in enumerate(voc_loaders[phase]):
				images = Variable(data[0].cuda())
				labels = Variable(data[1].cuda())
				image_ids = list(data[2])

				optimizer.zero_grad()

				# Forward backward
				confidences = model(images)
				loss = criterion(confidences, labels)

				if phase == 'train':
					loss.backward()
					optimizer.step()

				# Accumulate results
				for b, image_id in enumerate(image_ids):
					results[image_id] = confidences.data[b]

				# Statistics
				running_loss += loss.data[0]
				if params['verbose']:
					print ('{} Epoch: {} Iteration: {}/{} Loss: {:.4f}'.format(
						phase, epoch+1, (i+1), num_iterations, running_loss / (i+1)))

			evaluation_summarys = voc_evaluators[phase].Evaluate(results)
			epoch_meanAP = evaluation_summarys['meanAP']
			print ('{} Epoch: {} Loss: {:.4f} meanAP: {:.4f}'.format(
				phase, epoch+1, running_loss / num_iterations, epoch_meanAP))

			meanAP_historys[phase].append(epoch_meanAP)

			# deep copy the best model
			if phase == 'val' and epoch_meanAP > best_meanAP:
				best_meanAP = epoch_meanAP
				best_epoch = epoch+1
				best_model = copy.deepcopy(model)
				best_summary = copy.deepcopy(evaluation_summarys)
		print ()

	print ('Best val meanAP at epoch {}: {:4f}'.format(best_epoch, best_meanAP))
	return best_model, best_epoch, best_summary, meanAP_historys

def _GetArguments():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--interactive", action="store_true", default=False,
		help="Run the script in an interactive mode")
	parser.add_argument("--verbose", action="store_true", default=False,
		help="Visualize all logs")
	parser.add_argument("--voc_devkit_dir", default="data/VOCdevkit",
		help="Root directory of VOC development kit")
	parser.add_argument("--voc_version", default="VOC2007",
		help="Target VOC dataset version")
	parser.add_argument("--model", default="resnet18",
		help="Pretrained model")
	parser.add_argument("--batch_size", default=100, type=int,
		help="Batch size")
	parser.add_argument("--num_data_loading_workers", default=4, type=int,
		help="Number of data loading workers")
	parser.add_argument("--num_epochs", default=50, type=int,
		help="Number of epochs for training")
	parser.add_argument("--initial_learning_rate", default=0.1, type=float,
		help="Initial learning rate for SGD")
	parser.add_argument("--learning_rate_decay_epoch", default=50, type=int,
		help="How frequently the learning rate will be decayed")
	parser.add_argument("--momentum", default=0.9, type=float,
		help="Momentum for learning rate of SGD")
	parser.add_argument("--seed", default=111, type=int,
		help="Random seed")
	parser.add_argument("--save_dir", default="temp_result",
		help="Directory for saving results of train / test")
	args = parser.parse_args()
	params = vars(args)
	print (json.dumps(params, indent=2))
	return params	

def main(params):
	if not torch.cuda.is_available():
		raise ValueError("Cuda is not available")

	print ('Set random seed to: {}'.format(params['seed']))
	random.seed(params['seed'])
	torch.manual_seed(params['seed'])
	torch.cuda.manual_seed(params['seed'])

	print ('Get voc datasets.. ', end='', flush=True)
	voc_datasets = GetVocDatasets(params)
	print ('Done')

	voc_loaders = GetVocLoaders(params, voc_datasets)

	voc_evaluators = GetVocEvaluators(params)

	num_classes = voc_datasets['train'].num_classes
	model = GetPretrainedModel(params, num_classes)
	model = model.cuda()

	criterion = nn.BCELoss()

	best_model, best_epoch, best_summary, meanAP_historys =\
		Train(params, model, criterion, OptimizerScheduler, voc_loaders,
		voc_evaluators)

	test_evaluation_summarys = Test(params, best_model, voc_loaders,
		voc_evaluators, 'test')

	if not os.path.isdir(params['save_dir']):
		print ("Directory {} doesn't exist. Create one.".format(
			params['save_dir']))
		os.makedirs(params['save_dir'])

	weight_save_path = os.path.join(params['save_dir'], 'model.pt')
	torch.save(best_model, weight_save_path)
	print ("Best model is saved: {}".format(weight_save_path))

	summary_save_path = os.path.join(params['save_dir'], 'summary.json')
	summary = {
		'best_epoch': best_epoch,
		'best_summary': best_summary,
		'meanAP_historys': meanAP_historys,
		'test_summary': test_evaluation_summarys,
	}	
	json.dump(summary, open(summary_save_path, 'w'))
	print ("Summary json file is saved: {}".format(summary_save_path))

if __name__ == "__main__":
	params = _GetArguments()

	if not params['interactive']:
		main(params)
