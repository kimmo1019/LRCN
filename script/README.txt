generate_vocabulary.py: 通过提供的标注数据来生成所需的语料库(按词划分则需要使用THU-LAC分词工具)
generate_captions_encode_json.py: 按照COCO caption的格式生成.json的标注格式(方便利用COCO API快速提取任一图片的标注，展示图片标注等功能)
shuffle_captions.py: 将图片的标注随机打乱，并将（img_id，caption_id）记录至文本，方便训练数据的存储以及模型训练时数据的抽取
generate_h5data.py: 按照图片标注打乱后的顺序，生成相应的input_mat,target_mat等，数据格式为hdf5，作为整个模型的输入数据
generate_words.ipynb: 用于展示整个数据预处理的过程（以上均为按词生成的数据，而此展示的为基于单个字的数据预处理过程）

数据预处理过程中使用的主要python library如下：
h5py: 用于生成h5文件
json: 用于生成json文件
COCO: 用于快速提取任一图片的所有标注，展示任一图片标注等
THU-LAC: 用于词模型中对中文的分词


