''' Pascal VOC dataset class

This classes has assumption that the image files are placed as in the standard
Parscal VOC data directory hierarchy.
'''

import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import xmltodict

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

def _GetObjectAnnotation(voc_devkit_dir, voc_version, image_id):
	voc_data_dir = os.path.join(voc_devkit_dir, voc_version)
	annotation_path = os.path.join(voc_data_dir,
		_VOC_PATHS['annotation_path'] % image_id)
	annotation = xmltodict.parse(open(annotation_path,'rb'), xml_attribs=True)
	object_annotations = annotation['annotation']['object']
	if type(object_annotations) == list:
		return object_annotations
	else:
		return [object_annotations]

def _LoadImage(voc_devkit_dir, voc_version, image_id):
	voc_data_dir = os.path.join(voc_devkit_dir, voc_version)
	image_path = os.path.join(voc_data_dir, _VOC_PATHS['image_path'] % image_id)
	return Image.open(image_path).convert('RGB')	

class VocClassification(data.Dataset):

	def __init__(self, voc_devkit_dir, voc_version, split,
		transform=None, target_transform=None):
		if voc_version not in _VOC_VERSIONS:
			raise ValueError('Unknown voc_version')
		if split not in _VOC_SPLITS:
			raise ValueError('Unknown split')

		self.voc_devkit_dir = voc_devkit_dir
		self.voc_version = voc_version
		self.voc_data_dir = os.path.join(voc_devkit_dir, voc_version)
		self.classes = _VOC_CLASSES[voc_version]
		self.split = split

		self.transform = transform
		self.target_transform = target_transform

		self.image_set = _GetImageSet(self.voc_devkit_dir, self.voc_version,
			self.split)
		self.num_examples = len(self.image_set)
		self.num_classes = len(self.classes)

		self.onehot_labels = self.GetOnehotLabels()

	def __getitem__(self, index):
		image = _LoadImage(self.voc_devkit_dir, self.voc_version,
			self.image_set[index])
		target = self.onehot_labels[index]

		if self.transform is not None:
			image = self.transform(image)	
	
		if self.target_transform is not None:
			target = self.target_transform(target)

		return image, target

	def GetOnehotLabels(self):
		onehot_labels = np.zeros((self.num_examples, self.num_classes))
		for i, image_id in enumerate(self.image_set):
			object_annotations = _GetObjectAnnotation(self.voc_devkit_dir,
				self.voc_version, image_id)
			for obj_anno in object_annotations:
				onehot_labels[i, self.classes.index(obj_anno['name'])] = 1
		return onehot_labels
