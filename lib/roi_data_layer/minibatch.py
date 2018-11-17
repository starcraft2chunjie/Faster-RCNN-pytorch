"""Compute minibatch blobs for training a Faster R-CNN network."""

import numpy as np 
import numpy.random as npr
from matplotlib.pyplot import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob

import pdb
def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it"""
    num_images = len(roidb)
    random_scale_ids = npr.randint(0, high=len(cfg.TRAIN.SCALES))

    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE({})if len(im.shape) == 2:
            im = im[:,:]
        
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffeif len(im.shape) == 2:
            im = im[:,:]
        
    im_blob, im_scales = _get_image_blob(roidb, random_scale_ids)

    blobs = {'data' : im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scale.
    """
    num_images = len(roidb)

    processed_ims = []
    im_scales = []

    for i in range(num_images):
        im = imread(roidb[i]['image'])

        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2
        # rgb -> bgr
        im = im[:, :, ::-1]

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scales = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                            cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scales)
        processed_ims.append(im)

    # Create a blob to hold the input
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
