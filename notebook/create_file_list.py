import numpy as np
import lmdb
import sys
sys.path.insert(0, '../python')
import caffe
import os
import cv2
from scipy import misc
from os.path import expanduser
from pycocotools.coco import COCO
from pycocotools.mask import *
import skimage.io as io
import cv2
import matplotlib.pyplot as plt

# dirs
data_type = 'val2014'
coco_root = '/home/x6huang/data/coco/'
annFile = '%s/annotations/instances_%s.json' % (coco_root, data_type)
image_root = coco_root + 'images/' + data_type + '/'
labelmap_root = coco_root + 'preprocessed/labelmap/' + data_type + '/'

# output dir
label_dir = '%s/preprocessed/labelmap/%s' % (coco_root, data_type)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

# load annotations
coco = COCO(annFile)
cat_ids = coco.getCatIds()
cat_dict = {}
for k, cat_id in enumerate(cat_ids):
    cat_dict[cat_id] = k + 1

# get name of images
f = open(data_type + '_image_gt_list.txt','w')

img_ids = coco.getImgIds()
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    # get paths
    image_path = image_root + img['file_name']
    labelmap_path = labelmap_root + img['file_name'][:-3] + 'png'
    f.write(image_path + ' ' + labelmap_path + '\n')
f.close()