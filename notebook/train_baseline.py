import numpy as np
import lmdb
import sys
caffe_root = '..'
sys.path.insert(0,caffe_root+'/python')
import caffe
import os
import cv2
from scipy import misc
from os.path import expanduser
home_dir = expanduser("~")
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from IPython.display import clear_output

from __future__ import division

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver(caffe_root+'/models/coco-baseline/solver.prototxt')

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)

solver.net.copy_from(caffe_root+'/models/vgg/vgg16_fully_conv.caffemodel')

niter = 80000
train_loss = np.zeros(niter)
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    if it % 100 == 0:
        clear_output()
        print 'iter %d, loss=%f' % (it, train_loss[it])
        plt.plot(train_loss[:it+1])
        plt.show()
        plt.savefig(caffe_root+'/models/coco-baseline/loss_curve.png')
        np.save(caffe_root+'/models/coco-baseline/train_loss.npy',train_loss)
    if it % 2000 == 0: 
        solver.net.save(caffe_root+'/models/coco-baseline/base_{}.caffemodel'.format(it))
print 'done'
