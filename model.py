import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from rpn.proposal_layer import ProposalLayer as proposal_layer_py
from rpn.anchor_target import rpn_target as anchor_target_layer_py
from frcnn.proposal_target_layer import frcnn_targets as proposal_target_layer_py
from utils_.box_utils import bbox_transform_inv, clip_boxes
from utils_.box_utils import py_cpu_nms
from utils_.utils import to_var

def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]

class CNN(nn.Module):

    def __init__(self):
        """Load the pretrained vgg11 and delete fc layer."""
        super(CNN, self).__init__()

        #If you call m.modules(), not only produce the changed but also the pretrained model
        #you can use m.modules() for recursive effect.
        """explanation:
           vgg16 itself is not a iterator，we need to use children(), then it's a iterator, we then use list()
           ,here vgg16 consista of a feature sequential and a classifier sequential.
           When we delete the fc layer, the list has only one element, then we need to extract it and use list()
           to divide them so that we can delete the last pooling layer.
        """ 
        vggnet = models.vgg16(pretrained=True)
        modules = list(vggnet.children())[:-1] #delete the last fc layer
        modules = list(modules[0])[:-1] #delete the last pooling layer

        self.vggnet = nn.Sequential(*modules)

        for module in list(self.vggnet.children())[:10]:
            print("fix weight", module)
            for param in module.paramters():
                param.requires_grad = False
            
    def forward(self, images):
        """Extract the image feature vector"""

        # return features in relu5_3
        features = self.vggnet(images)
        return features
    
class RPN(nn.Module):
    
    def __init__(self):
        super(RPN, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride = 1, padding = (1, 1))\
        , nn.Relu())

        # 9 anchor * 2 classifier (object or non-object) each grid
        self.score_conv = nn.Conv2d(512, 2 * 9, kernel_size=1, stride = 1)

        # 9 anchor * 4 coordinate regressor each grids
        self.bbox_conv = nn.Conv2d(512, 4 * 9, kernel_size=1, stride=1)
        self.softmax() = nn.Softmax()
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, features):

        features = self.conv(features)

        rpn_cls_score, rpn_bbox_pred = self.score_conv(features), self.bbox_conv(features)

        height, width = features.size()[-2:]
        rpn_cls_score_reshape = rpn_cls_score.squeeze(0).permute(1, 2, 0).contiguous() # (1, 18, H/16, W/16) => (H/16, W/16, 18)
        rpn_cls_score_reshape = rpn_cls_score_reshape.view(-1, 2) # (H/16, W/16, 18) => (H/16 * W/16 * 9, 2)

        rpn_cls_prob = self.softmax(rpn_cls_score_reshape)
        rpn_cls_prob = rpn_cls_prob.view(height, width, 18)
        rpn_cls_prob = rpn_cls_prob.permute(2, 0, 1).contiguous().unsqueeze(0) #(1, 18, H/16, W/16)

        rois = self.proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, test, args)
        rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info)
        self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

        return features, rois
    
    def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, test, args):
        rpn_cls_prob = rpn_cls_prob.data.cpu().numpy
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy
        x = proposal_layer_py.proposal(rpn_bbox_pred, rpn_cls_prob, im_info, test, args)
        x = to_var(x)
        return x.view(-1, 5)
    
    def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info):
         """
        rpn_cls_score: for pytorch (1, Ax2, H/16, W/16) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info)
        rpn_labels = to_var(rpn_labels)
        rpn_bbox_targets = to_var(rpn_bbox_targets)
        rpn_bbox_inside_weights = to_var(rpn_bbox_inside_weights)
        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights
    
    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
        #classification loss
        rpn_label = rpn_data[0].view(-1)
        
        #select the anchor whose label is not -1
        #ne(-1) return the binary value (the standard is that whether the element of it is equal
        # to -1, if not equal, then apply 1, otherwise apply 0)
        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

        fg_cnt = torch.sum(rpn_label.data.ne(0))

        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

        #box loss
        rpn_bbox_targets, rpn_bbox_inside_weights = rpn_data[1:]
        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return rpn_cross_entropy, rpn_loss_box
    
    def load_from_npz(self, params):
        self.features.load_from_npz(params)

        pairs = {'conv1.conv': 'rpn_conv/3*3', 'score_conv.conv':'rpn_cls_score', 'bbox_conv.conv':'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(3, 2, 0, 1)
            own_dict[key].copy_(param)
        

class ROIpooling(nn.Module):
    
    def __init__(self, size=(7, 7), spatial_scale = 1.0 / 16.0):
        super(ROIpooling, self).__init__()
        self.adapmax2d = nn.AdaptiveAvgPool2d(size)
        self.spatial_scale = spatial_scale
    
    def forward(self, features, rois_boxes):
        # rois_boxes : [x, y, x', y']
        if type(rois_boxes) == np.ndarray:
            rois_boxes = rois_boxes.data.float().clone()
            rois_boxes.mul_(self.spatial_scale)
            rois_boxes = rois_boxes.long()

            output = []

            for i in range(rois_boxes.size(0)):
                roi = rois_boxes[i]

                try:
                    roi_feature = features[:, :, roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
                except Exception as e:
                    print(e, roi)

                pool_feature = self.adapmax2d(roi_feature)
                output.append(pool_feature)

        return torch.cat(output, 0)

class FasterRcnn(nn.Module):
    num_classes = 21
    classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (600,)
    MAX_SIZE = 1000

    def __init__(self):
        super(FasterRcnn, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)

        self.fc1 = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                 nn.ReLU(),
                                 nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout())
        # 20 classes + 1 backround classifier each roi
        self.classifier = nn.Linear(4096, num_classes)
        self.softmax = nn.Softmax()

        # 21 classes * 4 coordinate regressor each roi
        self.regressor = nn.Linear(4096, num_classes * 4)
        
        #loss
        self.cross_entropy = None
        self.loss_box = None

    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, features, rois):
        roi_data = self.proposal_target_layer(rois, gt_boxes, args, self.n_classes)
        rois = roi_data[0]

        features = features.view(-1, 512 * 7 * 7)
        features = self.fc1(features)
        features = self.fc2(features)


        try:
            cls_score = self.classifier(features)
            cls_prob = self.softmax(logits)
            bbox_delta = self.regressor(features)
        
        except Exception as e:
            print(e, logits)

        self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)

        return cls_prob, bbox_delta, rois

    def proposal_target_layer(rpn_rois, gt_boxes, args, num_classes):
        rpn_rois = rpn_rois.data.cpu().numpy
        labels, rois, bbox_targets, bbox_inside_weights = \
        proposal_target_layer_py(rpn_rois, gt_boxes, args, num_classes)
        rois = to_var(rois)
        labels = to_var(labels)
        bbox_targets = to_var(bbox_targets)
        bbox_inside_weights = to_var(bbox_inside_weights)
        return rois, labels, bbox_targets, bbox_inside_weights
        

    def build_loss(cls_score, bbox_pred, roi_data):
        # classification loss
        label = roi_data[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt

        # for log
        if self.debug:
            maxv, predict = cls_score.data.max(1)
            self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
            self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt

        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt) / bg_cnt
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)

        # bounding box regression L1 loss
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return cross_entropy, loss_box
    
    def get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
            im(ndarray): a color image in BGR order
        Returns:
            blob(ndarray): a data blob holding a image pyramid
            im_scale_factors(list): list of image scales (relative to im)
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        im_shape = im_orig.shape 
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.min(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            #Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.MAX_SIZE
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation = cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)
        
        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)
        return blob, np.array(im_scale_factors)

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        processed_ims = [im]
        im_scale_factors = [1.0]

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)
    
    def detect(self, image, thr=0.3):
        im_data, im_scales = self.get_image_blob(image)
        num_channel = 3
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], num_channel, im_scales[0]]],
            dtype=np.float32)
        
        cls_prob, bbox_pred, rois = self(im_data, im_info)
        pred_boxes, scores, classes = \
            self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
            return pred_boxes, scores, classes


    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4) : (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)
        
        #nms
        pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

        return pred_boxes, scores, self.classes[inds]
    
    def load_from_npz(self, params):
        self.rpn.load_from_npz(params)

        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)
    




        
        
    
