# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
#
#
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
#
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster.

# In[3]:


import sys
sys.path.append('cocoapi/PythonAPI/')

from pycocotools.coco import COCO
import os
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from config import Config
import utils
import pose_all_model as modellib
import visualize
from pose_model import log

# get_ipython().magic('matplotlib inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

train_dataset = '/home/liang/Downloads/train2014/'
val_dataset = '/home/liang/Downloads/val2014/'
train_annotation = '/home/liang/Downloads/annotations/person_keypoints_train2014.json'
val_annotation = '/home/liang/Downloads/annotations/person_keypoints_val2014.json'


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2
    NUM_KEYPOINTS = 17

    # Number of classes (including background)


    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = ShapesConfig()


# config.display()

# ## Notebook Preferences

# In[51]:

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# ## Dataset
#
# Create a synthetic dataset
#
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
#
# * load_image()
# * load_mask()
# * image_reference()

# In[155]:
coco = COCO(train_annotation)
cats = coco.loadCats(coco.getCatIds())
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
jointall = []
import cv2 as cv
imagesNum = 100
max_dim = 640


def padding(image, top_pad=0, left_pad=0, person=True):
    # Get new height and width
    h, w = image.shape[:2]
    bottom_pad = max_dim - h - top_pad
    right_pad = max_dim - w - left_pad
    # print(bottom_pad, right_pad)
    if person:
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    else:
        padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
    paddingimage = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return (paddingimage, window)


# annFile = '/data/datasets/COCO/person_keypoints_trainval2014/person_keypoints_train2014.json' # keypoint file
# trainimagepath = '/data/guest_users/liangdong/liangdong/practice_demo/train2014/'             # train image path

def compute_mask_and_label(anns):
    n_rois = len(anns)
    mask_target = np.zeros((max_dim, max_dim, n_rois), dtype=np.int8)
    mask_label = []

    for n in range(n_rois):
        mask = coco.annToMask(anns[n])
        # print(mask.shape)
        paddingmask, window = padding(mask, 0, 0, False)
        # mask = cv.resize(mask, (28, 28), interpolation=cv.INTER_NEAREST)
        mask_target[:, :, n] = paddingmask
        mask_label.append(1)
    return mask_target, mask_label


class Cocodata(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "person")
        self.add_class("shapes", 2, "other")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(imagesNum):
            img = coco.loadImgs(imgIds[i])[0]
            self.add_image("shapes", image_id=i, path=val_dataset + img['file_name'],
                           width=640, height=640, shapes=['person'])

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        '''
        img=cv2.imread(trainimagepath+imagenames[image_id])
        return img
        '''
        # train_dataset
        img = coco.loadImgs(imgIds[image_id])[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # I = io.imread(img['coco_url'])
        image = cv.imread(train_dataset + img['file_name'])
        print('image shape: ', image.shape)
        paddingI, window = padding(image)
        return paddingI
        '''
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image
        '''

    def load_mask_keypoint(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        img = coco.loadImgs(imgIds[image_id])[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # print(len(anns))
        # I = io.imread(img['coco_url'])
        n_rois = len(anns)

        keypoint_target = np.zeros((max_dim, max_dim, 17, n_rois), dtype=np.float)
        mask_target = np.zeros((max_dim, max_dim, n_rois), dtype=np.int8)
        mask_label = []
        keypoint_label = []

        for n in range(n_rois):
            mask = coco.annToMask(anns[n])
            # print(mask.shape)
            paddingmask, window = padding(mask, 0, 0, False)
            # mask = cv.resize(mask, (28, 28), interpolation=cv.INTER_NEAREST)
            mask_target[:, :, n] = paddingmask
            mask_label.append(1)


            for i in range(17):
                if anns[n]['area'] > 2 * 2:
                    keypointx = anns[n]['keypoints'][3 * i]
                    keypointy = anns[n]['keypoints'][3 * i + 1]
                    if anns[n]['keypoints'][3 * i + 2] > 0:
                        keypoint_target[keypointy, keypointx, i, n] = 1
                        sigma = 7
                        for k in range(-10,10):
                            for j in range(-10,10):
                                if keypointy+k > 0 and keypointy+k < 640 and keypointx+j > 0 and keypointx+j < 640:

                                    d2 = k * k + j * j
                                    exponent = d2 / 2.0 / sigma / sigma

                                    if (exponent > 4.6052):
                                        continue
                                    x = math.exp(-exponent)
                                    keypoint_target[keypointy + k, keypointx + j, i, n] = x

                        '''
                        for k in range(-3, 3):
                            for j in range(-3, 3):
                                sigma = 4
                                d2 = k * k + j * j
                                exponent = d2 / 2.0 / sigma / sigma
                                #print(exponent)
                                if (exponent > 4.6052):
                                    continue
                                x = math.exp(-exponent)
                                keypoint_target[keypointy + k, keypointx + j, n] = x
                        '''
                        '''
                        for k in range(-20, 20):
                            for j in range(-20, 20):
                                keypoint_target[keypointy+k, keypointx+j, n] = 1
                                keypoint_target[keypointy+k, keypointx+j, n] = 1
                        '''

            #keypoint_target[:, :, n] = paddingmask
            keypoint_label.append(1)

        return mask_target, np.array(mask_label), keypoint_target, np.array(keypoint_label)
# In[156]:

print('load train and validation dataset')

# Training dataset
dataset_train = Cocodata()
dataset_train.load_shapes(imagesNum, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = Cocodata()
dataset_val.load_shapes(10, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# In[157]:

print('load mask')
mask, class_ids, keypoint_target, keypoint_label = dataset_train.load_mask_keypoint(0)

# Load and display random samples
'''
image_ids = [0,1,2,4]#np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, 1)
'''
# ## Ceate Model
# In[161]:
# Create model in training mode
print('define model')
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

print('load params')
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# ## Training
#
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
#
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# In[163]:

print('model training')
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')

# In[164]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=10,
            layers="all")

# In[166]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_humn.h5")
model.keras_model.save_weights(model_path)

# ## Detection

# In[167]:




