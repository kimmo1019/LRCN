from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random

DATA_PATH = '/home/liuqiao/PR_project/data'
f_img2cap_train = open('%s/image2cap_train.txt'%DATA_PATH,'w')
f_img2cap_valid = open('%s/image2cap_valid.txt'%DATA_PATH,'w')
annFile = '%s/captions_train.json'%DATA_PATH
caps=COCO(annFile)
img2cap_train = []
img2cap_valid = []
for img_id in range(8000):
	annIds = caps.getAnnIds(imgIds=img_id+1)
	for each_annId in annIds:
		image2cap_train.append([img_id,each_annId])
for img_id in range(8000,9000):
	annIds = caps.getAnnIds(imgIds=img_id+1)
	for each_annId in annIds:
		image2cap_valid.append([img_id,each_annId])

img2cap_train = random.shuffle(img2cap_train)
img2cap_valid = random.shuffle(img2cap_valid)
for each in img2cap_train:
	line = each[0]+'\t'+each[1]+'\n'
	f_img2cap_train.write(line)
for each in img2cap_valid:
	line = each[0]+'\t'+each[1]+'\n'
	f_img2cap_valid.write(line)
f_img2cap_train.close()
f_img2cap_valid.close()