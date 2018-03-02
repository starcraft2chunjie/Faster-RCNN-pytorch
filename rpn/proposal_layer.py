import numpy as np 
from utils_.box_utils import bbox_transform_inv, py_cpu_nms, clip_boxes, filter_boxes
from utils_.anchor_utils import generate_anchors, get_anchor

"""
ProposalLayer is used to utilize all the [dx, dy, dw, dh] and the foreground anchors
to caculate the precise proposal, in order to send to the Roi Pooling layer
"""


class ProposalLayer:
    def __init__(self, args):
        self.args = args
    
    def _get_pos_score(self, rpn_cls_prob): #rpn_cls_score: [1, 18, H/16, W/16]

        pos_scores = rpn_cls_prob[:, :9]
        pos_scores = pos_scores.squeeze(0).permute(1, 2, 0).contiguous()  #(H/16, W/16, 9)
        pos_scores = pos_scores.view(-1, 1) #(H/16, W/16, 9) => (H/16 * W/16 * 9, 1)

        return pos_scores
    
    def _get_bbox_deltas(self, rpn_bbox_pred):
        
        bbox_deltas = rpn_bbox_pred.squeeze(0).permute(1, 2, 0).contiguous() # (1, 36, H/16, W/16) => (H/16 ,W/16, 36)
        bbox_deltas = bbox_deltas.view(-1, 4)  # (H/16 ,W/16, 36) => (H/16 * W/16 * 9, 4)

        return bbox_deltas

    def proposal(self, rpn_bbox_pred, rpn_cls_prob, im_info, test, args):
        """
        Arguments:
            rpn_bbox_pred (Tensor) : (1, 4*9, H/16, W/16)
            rpn_cls_prob (Tensor) : (1, 2*9, H/16, W/16)
            im_info (Tuple) : (Height, Width, Channel, scale_ratios)
            test (Bool) : True or False
            args (argparse.Namespace) : global arguments
        Return:
            # in each minibatch number of proposal boxes is variable
            proposals_boxes (Ndarray) : ( # proposal boxes, 4)
            scores (Ndarray) :  ( # proposal boxes, )
        """
        """
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        #layer_params = yaml.load(self.param_str_)
        """


        anchors = generate_anchors()
        _num_anchors = anchors.shape[0]

        all_anchors = get_anchor(rpn_cls_prob, anchors)   # [H * W * 9, 4]

        pre_nms_topn = args.pre_nms_topn if test == False else args.test_pre_nms_topn
        nms_thresh = args.nms_thresh if test == False else args.test_nms_thresh
        post_nms_topn = args.post_nms_topn if test == False else args.test_post_nms_topn

        """It's directly from anchor_target_layer, essentially from training the RPN"""
        bbox_deltas = self._get_bbox_deltas(rpn_bbox_pred).data.cpu().numpy()

        # 1. Convert anchors into proposal via bbox transformation
        """Here we need to generate the precise proposal location for the later operation"""
        proposals_boxes = bbox_transform_inv(all_anchors, bbox_deltas)  # (H/16 * W/16 * 9, 4) all proposal boxes
        scores = self._get_pos_score(rpn_cls_prob).data.cpu().numpy()


        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals_boxes, im_info[:2])

         # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[3])
        keep = filter_boxes(proposals_boxes, self.args.min_size * max(im_info[3]))
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topn (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topn > 0:
            order = order[:pre_nms_topn]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)

        keep = py_cpu_nms(np.hstack((proposals, scores)), nms_thresh)

        # 7. take after_nms_topN (e.g. 300)
        if post_nms_topn > 0:
            keep = keep[:post_nms_topn]
        
        # 8. return the top proposals (-> RoIs top)
        proposals = proposals[keep, :]
        scores = scores[keep]
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy = False)))
        return blob









