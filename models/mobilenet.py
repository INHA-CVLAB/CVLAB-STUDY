from builtins import super

import tensorflow as tf
from data_loader.cifar10 import cifar10, normalize
from data_loader.tiny_imagenet import tiny_imagenet
# from keras.optimizers import *
# from keras.models import Model #
# from keras.layer import Conv2D, Max
# from keras.layer import Activation, BatchNormalization
from keras.activations import * 
from keras.callbacks import *  #
'''
http://github.com/arthurdouillard/keras-mobilenet
'''

# class MobileNet(object):
# 	def __init__(self, x):
# 		self.model = self.MobileNet(x)

def MobileNet(input_x):
	# Input Size(224 x 224 x 3)
	x = tf.layers.conv2d(input_x, 32, kernel_size=3, strides=(2,2), padding='SAME', name='Conv0/s2')

	# Input Size(112 x 112 x 32)
	x1 = tf.layers.separable_conv2d(x, 32, kernel_size=3, padding='SAME', strides=(1, 1)) #(112, 112, 32)
	x1 = tf.layers.batch_normalization(x1)
	x1 = tf.nn.relu(x1)
	x1 = tf.layers.conv2d(x1,64,kernel_size=1, strides=(1,1)) #(112, 112, 64)
	x1 = tf.layers.batch_normalization(x)
	x1 = tf.nn.relu(x1)

	# Input Size(112 x 112 x 64)
	x2 = tf.layers.separable_conv2d(x1, 64, kernel_size=3, padding='SAME', strides=(2, 2)) #(56, 56, 64)
	x2 = tf.layers.batch_normalization(x2)
	x2 = tf.nn.relu(x2)
	x2 = tf.layers.conv2d(x2, 128, kernel_size=1, padding='SAME', strides=(1, 1)) #(56, 56, 128)
	x2 = tf.layers.batch_normalization(x2)
	x2 = tf.nn.relu(x2)

	# # Input Size(56 x 56 x 128)
	# x3 = tf.layers.separable_conv2d(x2, 128, kernel_size=3, padding='SAME', strides=(1, 1)) #(56, 56, 128)
	# x3 = tf.layers.batch_normalization(x3)
	# x3 = tf.nn.relu(x3)
	# x3 = tf.layers.conv2d(x3, 128, kernel_size=1, padding='SAME', strides=(1, 1)) #(56, 56, 128)
	# x3 = tf.layers.batch_normalization(x3)
	# x3 = tf.nn.relu(x3)
	#
	# # Input Size(56 x 56 x 128)
	# x4 = tf.layers.separable_conv2d(x3, 128, kernel_size=3, padding='SAME', strides=(2, 2)) #(28, 28, 128)
	# x4 = tf.layers.batch_normalization(x4)
	# x4 = tf.nn.relu(x4)
	# x4 = tf.layers.conv2d(x4, 256, kernel_size=1, strides=(1, 1)) #(28, 28, 128)
	# x4 = tf.layers.batch_normalization(x4)
	# x4 = tf.nn.relu(x4)

	# # Input Size(28 x 28 x 256)
	# x5 = tf.layers.separable_conv2d(x4, 256, kernel_size=3, padding='SAME', strides=(1, 1)) #(28, 28, 256)
	# x5 = tf.layers.batch_normalization(x5)
	# x5 = tf.nn.relu(x5)
	# x5 = tf.layers.conv2d(x5, 256, kernel_size=1, strides=(1, 1)) #(28, 28, 256)
	# x5 = tf.layers.batch_normalization(x5)
	# x5 = tf.nn.relu(x5)
	#
	# # Input Size(28 x 28 x 256)
	# x6 = tf.layers.separable_conv2d(x5, 256, kernel_size=3, padding='SAME', strides=(2, 2)) #(14, 14, 256)
	# x6 = tf.layers.batch_normalization(x6)
	# x6 = tf.nn.relu(x6)
	# x6 = tf.layers.conv2d(x6, 512, kernel_size=1, padding='SAME', strides=(1, 1)) #(14,14,512)
	# x6 = tf.layers.batch_normalization(x6)
	# x6 = tf.nn.relu(x6)

	##########################################################################################
	# Input Size(14 x 14 x 512)
	x7 = tf.layers.separable_conv2d(x2, 256, kernel_size=3, padding='SAME', strides=(1, 1)) #(14, 14, 512)
	x7 = tf.layers.batch_normalization(x7)
	x7 = tf.nn.relu(x7)
	x7 = tf.layers.conv2d(x7, 512, kernel_size=1, padding='SAME', strides=(1, 1)) #(14,14,512)
	x7 = tf.layers.batch_normalization(x7)
	x7 = tf.nn.relu(x7)

	x8 = tf.layers.separable_conv2d(x7, 256, kernel_size=3, padding='SAME', strides=(1, 1)) #(14, 14, 512)
	x8 = tf.layers.batch_normalization(x8)
	x8 = tf.nn.relu(x8)
	x8 = tf.layers.conv2d(x8, 512, kernel_size=1, padding='SAME', strides=(1, 1)) #(14,14,512)
	x8 = tf.layers.batch_normalization(x8)
	x8 = tf.nn.relu(x8)

	x9 = tf.layers.separable_conv2d(x8, 256, kernel_size=3, padding='SAME', strides=(1, 1)) #(14, 14, 512)
	x9 = tf.layers.batch_normalization(x9)
	x9 = tf.nn.relu(x9)
	x9 = tf.layers.conv2d(x9, 512, kernel_size=1, padding='SAME', strides=(1, 1)) #(14,14,512)
	x9 = tf.layers.batch_normalization(x9)
	x9 = tf.nn.relu(x9)

	x10 = tf.layers.separable_conv2d(x9, 256, kernel_size=3, padding='SAME', strides=(1, 1)) #(14, 14, 512)
	x10 = tf.layers.batch_normalization(x10)
	x10 = tf.nn.relu(x10)
	x10 = tf.layers.conv2d(x10, 512, kernel_size=1, padding='SAME', strides=(1, 1)) #(14,14,512)
	x10 = tf.layers.batch_normalization(x10)
	x10 = tf.nn.relu(x10)

	x11 = tf.layers.separable_conv2d(x10, 256, kernel_size=3, padding='SAME', strides=(1, 1)) #(14, 14, 512)
	x11 = tf.layers.batch_normalization(x11)
	x11 = tf.nn.relu(x11)
	x11 = tf.layers.conv2d(x11, 512, kernel_size=1, padding='SAME', strides=(1, 1)) #(14,14,512)
	x11 = tf.layers.batch_normalization(x11)
	x11 = tf.nn.relu(x11)
	##########################################################################################
	# Input Size(14 x 14 x 512)
	x12 = tf.layers.separable_conv2d(x11, 512, kernel_size=3, padding='SAME', strides=(2, 2)) #(7, 7, 512)
	x12 = tf.layers.batch_normalization(x12)
	x12 = tf.nn.relu(x12)
	x12 = tf.layers.conv2d(x12, 1024, kernel_size=1, padding='SAME', strides=(1, 1)) #(7, 7, 1024)
	x12 = tf.layers.batch_normalization(x12)
	x12 = tf.nn.relu(x12)

	# # Input Size(7 x 7 x 1024)
	# x13 = tf.layers.separable_conv2d(x12, 512, kernel_size=3, padding='SAME', strides=(2, 2)) #(7, 7, 1024)
	# x13 = tf.layers.batch_normalization(x13)
	# x13 = tf.nn.relu(x13)
	# x13 = tf.layers.conv2d(x13, 1024, kernel_size=1, padding='SAME', strides=(1, 1)) #(3, 3, 1024)
	# x13 = tf.layers.batch_normalization(x13)
	# x13 = tf.nn.relu(x13)

	x14 = tf.layers.average_pooling2d(x12, pool_size=(4,4), strides=(1, 1))
	x14 = tf.layers.dense(x14, units=200) #In case tiny-ImageNet 'unit=200'

	return tf.reshape(x14, [-1, 200]) #In case tiny-ImageNet 'unit=200'

iteration = 782 * 2 # data_number = batch_size * iteration
test_iteration = 10

batch_size = 64 # 50048 = 782 * 64
total_epoch = 10
init_learning_rate = 1e-4

global_step = tf.Variable(0, trainable=False, name = 'global_step')

train_x, train_y, test_x, test_y = tiny_imagenet()
train_x, test_x = normalize(train_x, test_x)

# image_size = 32, img_channels = 3, class_num = 10 in cifar10
# X = tf.placeholder(tf.uint8 shape=[None, 32, 32, 3]) # <-------------Input
# label = tf.placeholder(tf.int32, shape=[None, 10])    # <-------------Input
X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3]) # <-------------Input
label = tf.placeholder(tf.float32, shape=[None, 200])    # <-------------Input

training_flag = tf.placeholder(tf.bool)					# <-------------Input
learning_rate = tf.placeholder(tf.float32, name='learning_rate') # <-------------Input

#logits = MobileNet(X).model
logits = MobileNet(X)

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))

with tf.name_scope('optimizer'):
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	train = optimizer.minimize(loss, global_step = global_step)

with tf.name_scope('Accuracy'):
	print(logits.shape)
	print(label.shape)
	correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(label, axis=1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())

#######
# 신경망 모델 학습
######
def Evaluate(sess):
	test_acc = 0.0
	test_loss = 0.0
	test_pre_index = 0
	add = 1000

	for it in range(test_iteration):
		test_batch_x = test_x[test_pre_index: test_pre_index + add]
		test_batch_y = test_y[test_pre_index: test_pre_index + add]
		test_pre_index = test_pre_index + add

		test_feed_dict = {
			X: test_batch_x,
			label: test_batch_y,
			learning_rate: epoch_learning_rate,
			training_flag: False
		}

		loss_, acc_ = sess.run([loss, accuracy], feed_dict=test_feed_dict)

		test_loss += loss_ /10.0
		test_acc += acc_ / 10.0

	summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
								tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

	return test_acc, test_loss, summary

with tf.Session() as sess:
	ckpt = tf.train.get_checkpoint_state('./model')
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())

	# tf.summary 참조
	#tf.summary.scalar('loss', loss)  # 값이 하나인 텐서를 수집할 때 사용. 손실값을 수정.
	# tf.summary.histogram("Weights",W1)
	##########################################################################################
	#merged = tf.summary.merge_all()  # 모아둔 텐서 값들을 계산하여 수집
	writer = tf.summary.FileWriter('./logs', sess.graph)  # 그래프와 텐서 값들을 저장할 장소 설정
	##########################################################################################

	epoch_learning_rate = init_learning_rate
	for epoch in range(1, total_epoch+1):
		if epoch == (total_epoch * 0.5) or epoch == (total_epoch * 0.75):
			epoch_learning_rate = epoch_learning_rate/10

		pre_index = 0
		train_acc = 0
		train_loss = 0

		for step in range(1, iteration+1):
			if pre_index + batch_size < 100000: #In case tiny-ImageNet 100,000
				batch_x = train_x[pre_index : pre_index + batch_size]
				batch_y = train_y[pre_index: pre_index + batch_size]
			else:
				batch_x = train_x[pre_index:] #When batch is out of data border
				batch_y = train_y[pre_index:]

			train_feed_dict = {
				X: batch_x,
				label: batch_y,
				learning_rate: epoch_learning_rate
			}
			_, batch_loss = sess.run([train, loss],feed_dict=train_feed_dict)
			batch_acc = accuracy.eval(feed_dict=train_feed_dict)

			train_loss += batch_loss
			train_acc += batch_acc
			pre_index += batch_size

			if step == iteration:
				train_loss /= iteration # average_loss
				train_acc /= iteration # average accuracy

				train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
												 tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

				test_acc, test_loss, test_summary = Evaluate(sess)

				writer.add_summary(summary=train_summary, global_step=epoch) #train_loss, train_accuracy
				writer.add_summary(summary=test_summary, global_step=epoch) #test_loss, test_accuracy
				writer.flush()

				line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
					epoch, total_epoch, train_loss, train_acc, test_loss, test_acc)
				print(line)

				with open('logs.txt', 'a') as f:
					f.write(line)

	saver.save(sess, './model/dnn.ckpt', global_step=global_step)


#
# def depthwise_conv_block(self, num_pwc_filters, width_multiplier, strides, name):
# 	'''
# 		3*3 Depthwise Conv
# 		BN + Relu
# 		1*1 Conv
# 		BN + Relu
# 	'''
#
# 	with tf.variable_scope(name):
# 		num_pwc_filters = round(num_pwc_filters * width_multiplier)
#
#
# with tf.name_scope('layer1'):
#
#
#
# def get_conv_block(tensor, channels, strides, alpha=1.0, name=''):
#
# 	channels = int(channels * alpha) #
# 	x = Conv2D(channels,
# 			kernel_size=(3,3),
# 			strides=strides,
# 			use_bias=Flase,
# 			padding='same',
# 			name='{}_conv'.format(name))(tensor)
# 	x = BatchNormalization(name='{}_bn'.format(name))(x)
# 	x = Ativations('relu', name='{}_act'.format(name))(x)
# 	return x
#
# def get_dw_sep_block(tensor, channels, strides, alpha=1.0, name=''):
# 	channels = int(channels * alpha)
#
# 	#Depthwise
# 	x = DepthwiseCon2D(kernel_size=(3,3),
# 					strides=strides,
# 					use_bias=False,
# 					padding='same',
# 					name={}_dw,format(name))(tensor)
# 	x = BatchNormalization(name='{}_bn1'.format(name))(x)
# 	x = Activations('relu', name='{}_act'.format(name))(x)
#
# 	#Pointwise
# 	x = Conv2D(channels,
# 			kernel_size=(1,1),
# 			strides=(1,1),
# 			use_bias=False,
# 			padding='same',
# 			name='{}_pw'.format(name))(tensor)
# 	x = BatchNormalization(name='{}_bn2'.format(name))(x)
# 	x = Ativations('relu', name='{}_act2'.format(name))(x)
# 	return x
#
# def MobileNet(shape, num_classes, alpha=1.0, include_top=Ture, weight=None):
# 	'''
# 	Arguments
# 		include_top:
# 	'''
# 	x_in = Input(shape=shape)
#
# 	x = get_conv_block(x_in, 32, (2,2), alpha=alpha, name='initial')
#
# 	layer = [
# 		(64, (1,1)),
# 		(128, (2,2)),
# 		(128, (1,1)),
# 		(256, (2,2)),
# 		(256, (1,1)),
# 		(512, (2,2)),
# 		*[(512, (1,1)) for _ in range(5)],
# 		(1024, (2,2)),
# 		(1024, (2,2))
# 	]
#
# 	for i, (channels, strides) in enumerate(layers):
# 		x =  get_dw_sep_block(x, channels, strides, alpha=alpha, name='block{}'.format(i))
#
# 	if include_top:
# 		x = GlobalAvgPool2D(name='global_avg')(x)
# 		x = Dense(num_classes, activations='softmax', name='softmax')(x)
#
# 	model = Model(input=x_in, outputs=x)
#
# 	if weight is not None:
# 		model.load_weights(weight, by_name=True)
#
# 	return model
#
# def MobileNetV2(shape, alpha, depth_multiplier=1):
#
#
#
# 	return