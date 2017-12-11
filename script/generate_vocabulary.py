#generate the vocabulary for all the words for Chinese captions

import thulac
import json
from collections import Counter

DATA_PATH = '/home/liuqiao/PR_project/data'
#split Chinese sentenses into words
spliter = thulac.thulac(seg_only=True, model_path="/home/liuqiao/THULAC-Python/models")
#store all the words as the vocabulary
vocab = []
lines_train = open('%s/train.txt'%DATA_PATH).readlines()
first_line = lines_train[0]
for i in range(len(lines_train)):
    line = lines_train[i]
    if line.decode('UTF-8').rstrip('\n').rstrip('\r').isdigit() or i==0:
        pass
    else:
        cap = line.rstrip('\n').rstrip('\r')
        for each in spliter.cut(cap): 
            vocab.append(each[0])
lines_valid = open('%s/valid.txt'%DATA_PATH).readlines()
for i in range(len(lines_valid)):
    line = lines_valid[i]
    if line.decode('UTF-8').rstrip('\n').rstrip('\r').isdigit() or i==0:
        pass
    else:
        cap = line.rstrip('\n').rstrip('\r')
        for each in spliter.cut(cap): 
            vocab.append(each[0])
vocab_uniq = set(vocab)
list_freq = []
for each in vocab_uniq:
    list_freq.append([each,vocab.count(each)])
list_freq = sorted(list_freq, key=lambda k: k[1],reverse=True)
dic_voc = {}
#write to a txt file
f_vocab = open('%s/vocabulary1.txt'%DATA_PATH,'w')
index = 1
for each in list_freq:
    string = '%05d\t%s\t%d\n'%((index+1),each[0],each[1])
    if index <= 5735:
        dic_voc[each[0].decode('UTF-8')] = float(index+1)
    f_vocab.write(string)
    index += 1
f_vocab.close()

#generate a dictionary for the vocabulary words
f_dic=open('%s/dic_voc.json'%DATA_PATH,'w')
json.dump(dic_voc,f_dic)
f_dic.close()





