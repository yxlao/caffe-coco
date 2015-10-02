# process annotation to label map, saved as png using uint 0 - 255
# valid map from 0 to 80, where 0 is backround

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

# base dir
home_dir = expanduser("~")
coco_root = home_dir + '/data/coco'

# annotation dir
# data_type = 'val2014'
data_type = 'train2014'
annFile = '%s/annotations/instances_%s.json' % (coco_root, data_type)

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

# process annotation to label map
img_ids = coco.getImgIds()
for img_id in img_ids:
    # load annotation object
    img = coco.loadImgs(img_id)[0]
    # class label map
    cat_map = np.zeros((img['height'], img['width']))
    # foreground map
    foreground_map = cat_map != 0
    # object area map
    area_map = np.zeros((img['height'], img['width']))
    # get annotations
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # process annotations
    for ann in anns:
        annarea = ann['area']
        segm = ann['segmentation']
        cat_id = ann['category_id']
        if type(segm) == dict:
            segm = [segm]
        Rs = frPyObjects(segm, img['height'], img['width'])
        masks = decode(Rs)
        masks = masks.max(axis=2)
        # part of the new mask that is overlapped with a previous mask
        overlap_mask = masks & foreground_map
        # part of the new mask that has no overlap
        nonoverlap_mask = masks - overlap_mask
        # part of the overlapped mask that should be updated
        update_mask = nonoverlap_mask | (overlap_mask & (annarea < area_map))
        # ppdate
        cat_map[update_mask.astype(bool)] = cat_dict[cat_id]
        area_map[update_mask.astype(bool)] = annarea
        foreground_map = cat_map != 0

    if not np.any(np.bitwise_and(cat_map >= 0, cat_map <= 80)):
        raise('Class label should be between 0 and 80')

    misc.imsave(label_dir + '/' + img['file_name'][:-3] + 'png',
                cat_map.astype(np.uint8))