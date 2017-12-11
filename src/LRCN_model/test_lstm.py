#!/usr/bin/env python

import numpy as np
import sys
#import cv2
import os
CAFFE_ROOT = '/home/liuqiao/caffe/'
sys.path.append(CAFFE_ROOT + 'python')
os.environ['GLOG_minloglevel'] = '2' # to suppress printing information
sys.path.append(CAFFE_ROOT + 'examples/coco_caption')
from captioner_from_img_feat import Captioner
import caffe
import h5py


#weights_path = CAFFE_ROOT +'models/zh_caption/lstm_lm_iter_16000.caffemodel'
weights_path = CAFFE_ROOT +'models/zh_caption/lstm_lm2_iter_16000.caffemodel'
#lstm_net_proto = CAFFE_ROOT + 'examples/zh_caption/deploy_lstm.prototxt'
lstm_net_proto = CAFFE_ROOT + 'examples/zh_caption/deploy_lstm2.prototxt'
DATA_ROOT = '/home/liuqiao/PR_project/data/'
test_h5_path = DATA_ROOT + 'test_fc2_pca1k_norm1.h5'
vocab_path = DATA_ROOT + 'vocabulary_runtime2.txt'
output_file = DATA_ROOT + 'result2_iter_16000.txt'

vocabulary = [line.strip() for line in open(vocab_path).readlines()]
h5file = h5py.File(test_h5_path, 'r')
h5test = h5file['feature']
device_id = 0
fout = open(output_file, 'w')
for i in range(h5test.shape[0]):
    cap = Captioner(weights_path, lstm_net_proto, vocab_path, device_id)
    #probs = cap.predict_single_word(descriptor,0)
    #top_preds = np.argsort(-1*probs)
    #print [vocabulary[index] for index in top_preds[:10]]
    feature = h5test[i, :]
    beams, beam_probs = cap.predict_caption(feature,strategy={'type': 'beam', 'beam_size': 2})
    sentence = [vocabulary[index] for index in beams[0][:]]
    caption = "".join(sentence[0:-1])
    print caption
    fout.write('%d %s\n' %(i+9000, caption))
    if np.mod(i, 100) == 0:
        print '#%d' %i
fout.close()

    #print [index for index in beams[0][:]]
#im = cv2.imread(image_file)
#cv2.imshow('image',im)
#cv2.waitKey()

