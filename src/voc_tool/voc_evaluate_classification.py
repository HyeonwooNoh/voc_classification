''' Pascal VOC classification task evaluation
'''

import numpy as np
import os

_VOC_PATHS = {
	'annotation_path': 'Annotations/%s.xml',
	'image_path': 'JPEGImages/%s.jpg',
	'image_set_path': 'ImageSets/Main/%s.txt',
	'class_image_set_path': 'ImageSets/Main/%s_%s.txt',
	'segmentation_class_annotation_path': 'SegmentationClass/%s.png',
	'segmentation_instance_annotation_path': 'SegmentationObject/%s.png',
	'segmentation_image_set_path': 'ImageSets/Segmentation/%s.txt',
}

_VOC_VERSIONS = ['VOC2006', 'VOC2007']
_VOC_SPLITS = ['train', 'val', 'test']

_VOC_CLASSES = {
	_VOC_VERSIONS[0]: ['bicycle', 'bus', 'car', 'cat', 'cow', 'dog', 'horse',
      'motorbike', 'person', 'sheep'],
	_VOC_VERSIONS[1]: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
}

def _GetImageSet(voc_devkit_dir, voc_version, split):
	voc_data_dir = os.path.join(voc_devkit_dir, voc_version)
	image_set_path = _VOC_PATHS['image_set_path'] % split
	with open(os.path.join(voc_data_dir, image_set_path), 'r') as f:
		return [line.split()[0] for line in f.readlines()]

class VocEvaluateClassification():

	def __init__(self, voc_devkit_dir, voc_version, split):
		if voc_version not in _VOC_VERSIONS:
			raise ValueError('Unknown voc_version')
		if split not in _VOC_SPLITS:
			raise ValueError('Unknown split')

		self.voc_devkit_dir = voc_devkit_dir
		self.voc_version = voc_version
		self.voc_data_dir = os.path.join(voc_devkit_dir, voc_version)
		self.split = split
		self.classes = _VOC_CLASSES[voc_version]

		self.image_set = _GetImageSet(voc_devkit_dir, voc_version, split)
	
		self.ground_truths = self.LoadGroundTruths()

	def LoadGroundTruths(self):
		ground_truths = {}
		for cls in self.classes:
			class_image_set_path = os.path.join(self.voc_data_dir,
				_VOC_PATHS['class_image_set_path'] % (cls, self.split))
			with open(class_image_set_path, 'r') as f:
				ground_truths[cls] = {line.split()[0]: int(line.split()[1])
					for line in f.readlines()}
		return ground_truths

	''' 
	Input results should have the following format:
	{image_id: confidence array}
	'''
	def Evaluate(self, results):
		cls_results = {}
		for i, cls in enumerate(self.classes):
			cls_results[cls] = {}
			for image_id, confidences in results.items():
				cls_results[cls][image_id] = confidences[i]

		average_precision_summary = {}
		precisions_summary = {}
		recalls_summary = {}
		for cls in self.classes:
			average_precision, precisions, recalls =\
				self.EvaluateClass(cls, cls_results[cls])
			average_precision_summary[cls] = average_precision
			precisions_summary[cls] = precisions
			recalls_summary[cls] = recalls

		meanAP = np.array(list(average_precision_summary.values())).mean()
		evaluation_summarys = {
			'meanAP': meanAP,
			'average_precision_summary': average_precision_summary,
			'precisions_summary': precisions_summary,
			'recalls_summary': recalls_summary,
		}
		return evaluation_summarys
	
	def EvaluateClass(self, cls, cls_results):
		num_examples = len(self.image_set)

		gts = np.ones(num_examples) * (-np.inf)
		for i, image_id in enumerate(self.image_set):
			gts[i] = self.ground_truths[cls][image_id]
		
		confidences = np.ones(len(gts)) * (-np.inf)
		for i, image_id in enumerate(self.image_set):
			confidences[i] = cls_results[image_id]

		# Descending order
		sorted_index = np.argsort(confidences)[::-1]
		
		true_positives = gts[sorted_index] > 0
		false_positives = gts[sorted_index] < 0
		true_positives = np.cumsum(true_positives)
		false_positives = np.cumsum(false_positives)
		recalls = true_positives / np.sum(gts > 0)

		# This logit is slightly complicated to avoid division by zero.
		eps = 1e-10
		positives = false_positives + true_positives
		precisions = true_positives / (positives + (positives == 0.0) * eps)

		# Compute average prediction
		average_precision = 0;
		# (0 ~ 1, interval: 0.1). 1.1 is used to include 1.0 in the range.
		for threshold in np.arange(0, 1.1, 0.1):
			precisions_at_recall_threshold = precisions[recalls >= threshold]
			if precisions_at_recall_threshold.size > 0:
				max_precision = np.max(precisions_at_recall_threshold)
			else:
				max_precision = 0
			average_precision = average_precision + max_precision/11;
		# Return precisions and recalls as lists for easier serialization.
		return average_precision, list(precisions), list(recalls)
						
voc_devkit_dir = 'data/VOCdevkit'
voc_version = 'VOC2007'
split = 'val'

voc_eval = VocEvaluateClassification(voc_devkit_dir, voc_version, split)
