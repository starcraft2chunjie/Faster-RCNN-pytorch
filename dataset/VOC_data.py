import torch
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import os.path
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class AnnotationTransform(object):
    
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        channels (int): number of channels
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind, keep_difficult=False):
        self.keep_difficult = keep_difficult
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        
    def __call__(self, target, scale=None):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a Tensor containing [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            # [xmin, ymin, xmax, ymax]
            bndbox = []
            for i, cur_bb in enumerate(bbox):
                bb_sz = int(cur_bb.text) - 1

                if scale is not None:
                    # scale height or width
                    bb_sz = bb_sz / scale[0] if i % 2 == 0 else bb_sz / scale[1]
                bndbox.append(bb_sz)

            label_ind = self.class_to_ind[name]
            bndbox.append(label_ind)
            res += [bndbox] #[[xmin, ymin, xmax, ymax, ind], ...]

class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2012')
    """
    def __init__(self, root, image_set, transform=None, target_transform=None,
                dataset_name='VOC2012'):
                self.root = root
                self.image_set = image_set
                self.transform = transform
                self.target_transform = target_transform

                # VOCdevkit/VOC2012/Annotations/%s.xml
                self._annopath = os.path.join(
                    self.root, dataset_name, 'Annotations', '%s.xml')
                
                # VOCdevkit/VOC2007/JPEGImages/%s.jpg
                self._imgpath = os.path.join(
                    self.root, dataset_name, 'ImageSets', 'Main', '%s.jpg')
                
                # VOCdevkit/VOC2007/ImageSets/Main/%s.txt
                self._imgsetpath = os.path.join(
                    self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')

                # "%s.txt" % "hi" => hi.txt
                with open(self._imgsetpath % self.image_set) as f:
                    self.ids = f.readlines()
                
                # train / val / test.txt
                self.ids = [x.strip('\n') for x in self.ids]
    
    def __getitem__(self, index):
        img_id = self.ids[index]

        # xml parse
        target = ET.parse(self._annopath % img_id).getroot

        img = Image.open(self._imgpath % img_id).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img.squeeze(0), target
    
    def __len__(self):
        return len(self.ids)
    
    def show(self, index, subparts=False):
        '''Shows an image with its ground truth boxes overlaid optionally
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
            subparts (bool, optional): whether or not to display subpart
            bboxes of ground truths
                (default: False)
        '''
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255,0,0))
            draw.text(obj[0:2], obj[4], fill=(0, 255, 0))
        img.show()
        return img


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type([])):
                annos = [torch.Tensor(a) for a in tup]
                targets.append(torch.stack(annos, 0))

    targets = torch.cat(targets, dim=0)
    return (torch.stack(imgs, 0), targets)
    



                

               