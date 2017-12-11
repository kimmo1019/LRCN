#generate the coco format captions file (.json)
import thulac
import json
import sys
DATA_PATH = '/home/liuqiao/PR_project/data'
#load vocabulary dictionary
f_dic = open('%s/dic_voc.json'%DATA_PATH,'r')
dic_voc = json.load(f_dic)
#split Chinese sentenses into words
spliter = thulac.thulac(seg_only=True, model_path="/home/liuqiao/THULAC-Python/models")

lines_train = open('%s/train.txt'%DATA_PATH).readlines()
anno_train = {}
info={}
annotations={}
info[u'contributor'] = u'WWT&LQ'
info[u'date_created'] = u'2017-04-25'
info[u'description'] = u'This is the 2017 Chinese image caption training dataset from PR class'
anno_train[u'info'] = info
anno_train[u'images']=[]
anno_train[u'annotations']=[]
image_id = 1
anno_id = 1
for i in range(len(lines_train)):
    line = lines_train[i]
    if line.decode('UTF-8').rstrip('\n').rstrip('\r').isdigit() or i==0:
        images={}
        if i==0:
            image_id = 1
        else:
            image_id = int(line.decode('UTF-8').rstrip('\n').rstrip('\r'))
        images[u'id'] = image_id
        anno_train[u'images'].append(images)
    else:
        anno={}
        cap = line.rstrip('\n').rstrip('\r')
        sentence = []
        for each in spliter.cut(cap):
            #print each[0]
            try:
                dic_voc[each[0].decode('UTF-8')]
            except KeyError:
                sentence.append(str(1.0))
            else:
                sentence.append(str(dic_voc[each[0].decode('UTF-8')]))
        cap_encode = ' '.join(sentence)
        anno[u'image_id'] = image_id
        anno[u'id'] = anno_id
        anno[u'caption'] = u'%s'%cap_encode
        anno_train[u'annotations'].append(anno)
        anno_id += 1

f_anno_train = open('%s/captions_train.json'%DATA_PATH,'w')
json.dump(anno_train,f_anno_train)
f_anno_train.close()
        
lines_valid = open('%s/valid.txt'%DATA_PATH).readlines()
anno_valid = {}
info={}
annotations={}
info[u'contributor'] = u'WWT&LQ'
info[u'date_created'] = u'2017-04-25'
info[u'description'] = u'This is the 2017 Chinese image caption validation dataset from PR class'
anno_valid[u'info'] = info
anno_valid[u'images']=[]
anno_valid[u'annotations']=[]
image_id = 8001

for i in range(len(lines_valid)):
    line = lines_valid[i]
    if line.decode('UTF-8').rstrip('\n').rstrip('\r').isdigit() or i==0:
        images={}
        if i==0:
            image_id = 8001
        else:
            image_id = int(line.decode('UTF-8').rstrip('\n').rstrip('\r'))
        images[u'id'] = image_id
        anno_valid[u'images'].append(images)
    else:
        anno={}
        cap = line.rstrip('\n').rstrip('\r')
        sentence = []
        for each in spliter.cut(cap):
            try:
                dic_voc[each[0].decode('UTF-8')]
            except KeyError:
                sentence.append(str(1.0))
            else:
                sentence.append(str(dic_voc[each[0].decode('UTF-8')]))
        cap_encode = ' '.join(sentence)
        anno[u'image_id'] = image_id
        anno[u'id'] = anno_id
        anno[u'caption'] = u'%s'%cap_encode
        anno_valid[u'annotations'].append(anno)
        anno_id += 1
        

f_anno_valid = open('%s/captions_valid.json'%DATA_PATH,'w')
json.dump(anno_valid,f_anno_valid)
f_anno_valid.close()