import torch
import numpy as np 

#torch tensors
def bbox_overlaps(boxes, gt_boxes):
    """
    paramaters:
        boxes: the prop box: [x1, y1, x2, y2] (1, 4)
        gt_boxes: ground truth box: [x1', y1', x2', y2'] (1, 4)
    """
    from utils import to_tensor

    if isinstance(boxes, np.ndarray):
        boxes = to_tensor(boxes)
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = to_tensor(gt_boxes)

    oo = []

    for b in gt_boxes:
        x1 = boxes.select(1, 0).clone()
        y1 = boxes.select(1, 1).clone()
        x2 = boxes.select(1, 2).clone()
        y2 = boxes.select(1, 3).clone()

        x1[x1.lt(b[0])] = b[0]
        y1[y1.lt(b[1])] = b[1]
        x2[x2.gt(b[2])] = b[2]
        y2[y2.gt(b[3])] = b[3]

        w = x2 - x1 + 1
        h = y2 - y1 + 1

        #intersection area
        inter = torch.mul(w, h).float()

        #total area
        aarea = torch.mul((boxes.select(1, 2) - boxes.select(1, 0) + 1), (boxes.select(1, 3)\
        - boxes.select(1, 1) + 1)).float()
        barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

        #intersection over union overlap

        overlap =  torch.div(inter, (aarea + barea - inter))
        # set invalid entries to 0 overlap
        overlap[w.lt(0)] = 0
        overlap[h.lt(0)] = 0

        oo += [overlap]
    return torch.cat([overlap.view(-1, 1) for overlap in oo], 1)

def _unmap(data, count, inds, fill = 0):
    """ Unmap boxes subset of item (data) back to the original set of
        items (of size count)"""
    
    # make label data
    # map up to original set of anchors
    # labels = _unmap(labels, total_anchors, inds_inside, fill = -1)
    # bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill = 0)
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def bbox_transform(boxes, gt_boxes):
    """return: the target, equvalent to the label(you need to be close to this data)"""

    # boxes is anchor or pred_rois
    # boxes : (num, 4)
    # gt_boxes : (num, 4)
    # x` - x + 1 = width
    # y` - y  +1 = height
    # x + 0.5 * width = ctr_x
    # y + 0.5 * height = ctr_y

    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + ex_widths * 0.5
    ex_ctr_y = boxes[:, 1] + ex_heights * 0.5

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    # x : predicted box, x_a : anchor box, x* : ground truth box

    # faster R-Cnn paper
    # p_dx = (x - x_a)/w_a
    # p_dy = (y - y_a)/h_a
    # p_dw = log(w/w_a)
    # p_dh = log(h/h_a)

    # t_dx* = (x* - x_a)/w_a
    # t_dy* = (y* - y_a)/h_a
    # t_dw* = log(w*/w_a)
    # t_dh* = log(h*/h_a)

    targets_dx = (gt_ctr_x - ex_ctr_x)/ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y)/ex_heights
    targets_dw = np.log(gt_widths/ex_widths)
    targets_dh = np.log(gt_heights/ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    # boxes is anchor or pred_box
    # if feature map is (H/16 * W/16) then boxes are (H/16 * W/16, 4)
    # deltas are also (H/16 * W/16, 4)
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0 #(W/16 * H/16, )
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # (1764, 1)
    #if only use deltas[:, 0] only producing the shape (1764, )
    #here 4 means the stride, and will save another axis
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x'
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y'
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes



def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    im_shape : (H, W)
    """
    if boxes.shape[0] == 0:
        return boxes

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes



def filter_boxes(boxes, min_size):
    """
    Remove all boxes with any side smaller than min_size
    """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1

    keep = np.where((ws >= min_size) & (hs >= min_size))[0]

    return keep



def py_cpu_nms(proposals_boxes_c, thresh):
    """Pure Python NMS baseline"""
    # proposals_boxes_c (?, 5)[x, y, x', y', score]
    scores = proposals_boxes_c[:, -1]
    x1 = proposals_boxes_c[:, 0]
    y1 = proposals_boxes_c[:, 1]
    x2 = proposals_boxes_c[:, 2]
    y2 = proposals_boxes_c[:, 3]

    areas = (x2 - x1) * (y2 - y1) #width * height
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]

    return keep


from PIL import Image
import collections

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __init__(self, random_number):
        self.random_number = random_number


    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.random_number < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class Maxsizescale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, maxsize, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.maxsize = maxsize
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size and h <= self.maxsize) or (h <= w and h == self.size and w <= self.maxsize):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)

                if oh <= self.maxsize:
                    return img.resize((ow, oh), self.interpolation)
                else:
                    oh = self.maxsize
                    ow = int(self.size * w / h)

                    return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)

                if ow <= self.maxsize:
                    return img.resize((ow, oh), self.interpolation)
                else:
                    ow = self.maxsize
                    oh = int(self.size * h / w)
                    return img.resize((ow, oh), self.interpolation)

        else:
            raise Exception("size is not int")


def _jitter_gt_boxes(gt_boxes, jitter=0.05):
    """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
    gt_boxes: (G, 4) [x1 ,y1 ,x2, y2] int
    """
    jittered_boxes = gt_boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    jittered_boxes[jittered_boxes < 0] = 0

    return jittered_boxes

    

        


