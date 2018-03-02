from collections import OrderedDict
from time import perf_counter as pc 

import matplotlib
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torchvision import transforms
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import os 
import random
import tensorflow as tf  

from dataset.VOC_data import VOCDetection, detection_collate, AnnotationTransform
from utils_.utils import make_name_string
from utils_.anchor_utils import get_anchors, generate_anchors

from model_easy import *
from utils_.utils import *
from rpn.proposal_layer import ProposalLayer 
from rpn.anchor_target import rpn_target
from frcnn.proposal_target_layer import frcnn_targets
from loss_easy import rpn_loss, frcnn_loss
from utils_.box_utils import RandomHorizontalFlip, Maxsizescale
from vgg import VGG16


def train(args):

    hyparam_list = [("model", args.model_name),
                    ("bk", args.backbone),
                    ("train", (args.pre_nms_topn, args.nms_thresh, args.post_nms_topn)),
                    ("test", (args.test_pre_nms_topn, args.test_nms_thresh, args.test_post_nms_topn)),
                    ("pos_th", args.pos_threshold),
                    ("bg_th", args.bg_threshold),
                    ("last_nms", args.frcnn_nms),
                    ("include_gt", args.include_gt),
                    ("lr", args.lr)]
    
    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    name_param = "/" + make_name_string(hyparam_dict)
    print(name_param)

    #for args.use_tensorboard
    if args.use_tensorboard:
        import tensorflow as tf 

        summary_writer = tf.summary.FileWriter(args.output_dir + args.log_dir + name_param)

        def inject_summary(summary_writer, tag, value, step):
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                summary_writer.add_summary(summary, global_step=step)

        inject_summary = inject_summary

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = VOCDetection(root=args.input_dir + "/VOCdevkit", image_set="trainval",
                            transform = transform, target_transform=AnnotationTransform())
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True,
                                              num_workers = 1, collate_fn = detection_collate)

    # model define
    t0 = pc()

    class Model(nn.Module):
        """
        this Model class is used for simple model saving and loading
        """
        def __init__(self, args);
            super(Model, self).__init__()
            