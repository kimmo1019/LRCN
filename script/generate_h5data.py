from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import h5py
import pylab
#generate training h5 data
annFile = '/home/liuqiao/PR_project/data/captions_train.json'
caps=COCO(annFile)
f_train = h5py.File("/home/liuqiao/PR_project/data/captions_train.h5", "w")
b_size = f_train.create_dataset(u'buffer_size', (1,), dtype='i8')
b_size[...] = 100
input_sentence = -np.ones((7700,100)) #(38445/100+1)*20
for i in range(385):
    input_sentence[20*i] = np.zeros(100)
image_cap_list = []
for line in open('/home/liuqiao/PR_project/data/image2cap_train.txt').readlines():
    img_id, cap_id = line.rstrip('\n').split('\t')
    image_cap_list.append([img_id,cap_id])
for i in range(len(image_cap_list)):
    img_id, cap_id = image_cap_list[i]
    annIds = caps.getAnnIds(imgIds=int(img_id))
    #if cap_id==25918:
     #   print img_id
    caption = caps.loadAnns(annIds[annIds.index(int(cap_id))])[0]['caption']
    cap_list = caption.split(' ')
    #print cap_list
    #sys.exit()
    
    batch_th = i/100
    start_row = 20*batch_th
    start_col = i-batch_th*100
    
    for j in range(np.min([len(cap_list),19])):
        try:
            input_sentence[start_row+j+1][start_col] = float(cap_list[j])#list out of range
        except IndexError:
            print start_row,start_col,i,j,len(cap_list)
            sys.exit()
input_seq = f_train.create_dataset(u'input_sentence', (7700,100), dtype='f8')
input_seq[...] = input_sentence

target_sentence = -np.ones((7700,100))
for i in range(385):
    for j in range(100):
        for k in range(19):
            if input_sentence[20*i+k+1][j]!=-1:
                target_sentence[20*i+k][j] = input_sentence[20*i+k+1][j]
            else:
                target_sentence[20*i+k][j] = 0
                break  
target_seq = f_train.create_dataset(u'target_sentence', (7700,100), dtype='f8')
target_seq[...] = target_sentence
cont_sentence = np.zeros([7700,100])
for i in range(7700):
    for j in range(100):
        if input_sentence[i][j] != -1 and input_sentence[i][j]!=0:
            cont_sentence[i][j]=1
con_seq = f_train.create_dataset(u'cont_sentence', (7700,100), dtype='f8')
con_seq[...] = cont_sentence
f_train.close()

#generate testing h5 data
annFile = '/home/liuqiao/PR_project/data/captions_valid.json'
caps=COCO(annFile)
f_valid = h5py.File("/home/liuqiao/PR_project/data/captions_valid.h5", "w")
b_s = f_valid.create_dataset(u'buffer_size', (1,), dtype='i8')
b_s[...] = 100
input_sentence = -np.ones((980,100)) #(4811/100+1)*20
for i in range(49):
    input_sentence[20*i] = np.zeros(100)
image_cap_list = []
for line in open('/home/liuqiao/PR_project/data/image2cap_valid.txt').readlines():
    img_id, cap_id = line.rstrip('\n').split('\t')
    image_cap_list.append([img_id,cap_id])
for i in range(len(image_cap_list)):
    img_id, cap_id = image_cap_list[i]
    annIds = caps.getAnnIds(imgIds=int(img_id))
    #if cap_id==25918:
     #   print img_id
    caption = caps.loadAnns(annIds[annIds.index(int(cap_id))])[0]['caption']
    cap_list = caption.split(' ')
    batch_th = i/100
    start_row = 20*batch_th
    start_col = i-batch_th*100
    for j in range(np.min([len(cap_list),19])):
        try:
            input_sentence[start_row+j+1][start_col] = float(cap_list[j])#list out of range
        except IndexError:
            print start_row,start_col,i,j,len(cap_list)
            sys.exit()
input_seq = f_valid.create_dataset(u'input_sentence', (980,100), dtype='f8')
input_seq[...] = input_sentence

target_sentence = -np.ones((980,100)) #(4811/100+1)*20
for i in range(49):
    for j in range(100):
        for k in range(19):
            if input_sentence[20*i+k+1][j]!=-1:
                target_sentence[20*i+k][j] = input_sentence[20*i+k+1][j]
            else:
                target_sentence[20*i+k][j] = 0
                break 
target_seq = f_valid.create_dataset(u'target_sentence', (980,100), dtype='f8')
target_seq[...] = target_sentence
cont_sentence = np.zeros([980,100])
for i in range(980):
    for j in range(100):
        if input_sentence[i][j] != -1 and input_sentence[i][j] != 0:
            cont_sentence[i][j]=1
con_seq = f_valid.create_dataset(u'cont_sentence', (980,100), dtype='f8')
con_seq[...] = cont_sentence
f_valid.close()

#combine the training and testing h5 data
f_h51=h5py.File('/home/liuqiao/PR_project/data/captions_train.h5','r')
f_h52=h5py.File('/home/liuqiao/PR_project/data/captions_valid.h5','r')
f_train_val = h5py.File("/home/liuqiao/PR_project/data/captions_train_val.h5", "w")
b_s = f_train_val.create_dataset(u'buffer_size', (1,), dtype='i8')
b_s[...] = 100
input_seq1 = f_h51['input_sentence']
input_seq2 = f_h52['input_sentence']
input_seq3 = np.concatenate((input_seq1, input_seq2), axis=0)
input_seq = f_train_val.create_dataset(u'input_sentence', (8680,100), dtype='f8')
input_seq[...]=input_seq3
input_target1 = f_h51['target_sentence']
input_target2 = f_h52['target_sentence']
input_target3 = np.concatenate((input_target1, input_target2), axis=0)
target_seq = f_train_val.create_dataset(u'target_sentence', (8680,100), dtype='f8')
target_seq[...]=input_target3
input_cont1 = f_h51['cont_sentence']
input_cont2 = f_h52['cont_sentence']
input_cont3 = np.concatenate((input_cont1, input_cont2), axis=0)
input_cont3.shape
cont_seq = f_train_val.create_dataset(u'cont_sentence', (8680,100), dtype='f8')
cont_seq[...]=input_cont3
f_train_val.close()