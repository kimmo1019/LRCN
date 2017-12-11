模型的训练：
	train.sh: LRCN caffe模型的训练shell脚本,可以选择使用GPU来加速训练。
	lstm_lm_solver.prototxt: LRCN模型的参数，包括迭代次数，学习率等。
	train.prototxt: LRCN网络的参数，包括输入层，LSTM层，输出层等。
模型的测试：
	test_lstm.py: 模型测试的Python脚本，会调用训练过程中生成的caffemodel、模型的配置文件deply_lstm.prototxt，直接生成最终的测试图片的caption文本。


Note: 当从基于词的模型变成基于字的模型，训练模型输入的h5文件得改动，测试模型调用的caffemodel得相应的改动。模型的配置文件中需要改动的地方为vocabulary size以及输出层向量的维度。其他地方均不变。


