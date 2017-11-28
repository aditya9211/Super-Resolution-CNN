# -*- coding: utf-8 -*-
# Imports
import numpy as np
import os
import tensorflow as tf
import scipy.misc as ms
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def cnn_model_fn(features):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, 33, 33, 1]) # Input is 33x33 Grayscale Image

  # Convolutional Layer #1	Low Dimensional Feature
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[9, 9],
      padding="valid",
      activation=tf.nn.relu)


  # Convolutional Layer #2 	Non-Linear Mapping (ReLU)
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[1, 1],
      padding="valid",
      activation=tf.nn.relu)
  
    # Convolutional Layer #3 	High Dimensional Feature
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=1,
      kernel_size=[5, 5],
      padding="valid")

  return conv3


def get_batch(X, y, batch_size): # Exracts Batches In SGD
    for i in np.arange(0, X.shape[0], batch_size):
        yield(X[i:i+batch_size,:,:],y[i:i+batch_size,:,:])


# Our Input Values
x = tf.placeholder(tf.float32, shape = [None, 33, 33])
y = tf.placeholder(tf.float32, shape = [None, 21, 21])


os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # To suprress the Tf Warning

# Load training and eval data
train_data = np.load("/tmp/SRCNN/data.npy") # Returns np.array
train_labels = np.load("/tmp/SRCNN/label.npy")

# Uncomment the code for Loading Eval Data
'''
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
'''

max_epochs = 20
num_examples = train_data.shape[0]

prediction = cnn_model_fn(x) # Outputs an 21x21xBatch_size Image
#print (prediction.shape)

# MSE Error Cost Function with ADAM Optimizer
loss = tf.reduce_sum(tf.square(prediction[:,:,:,0]-y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(max_epochs):
			epoch_loss = 0
			count = 0
			print ()
			print (epoch+1 , ': ........ ')
			for (epoch_x , epoch_y) in get_batch(train_data, train_labels, 50):
					_, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
					count = count + 1
					epoch_loss += (c)/(21.0*21.0)
			epoch_loss = (epoch_loss*1.0)/count
			print ('Epoch', epoch+1, 'completed out of ', max_epochs, 'loss', epoch_loss)
			print ()


	#prediction = cnn_model_fn(tf.cast(train_data[101,:,:],dtype=tf.float32))
	#ms.imshow(prediction[:,:,0])
	#ms.imshow(train_labels[101,:,:])
	
	sr = prediction.eval({x:train_data[101:103,:,:]})
	#ms.imshow(train_labels[101,:,:])
	#ms.imshow(sr[0,:,:,0])
	#print (sr[0,:,:,0], train_labels[101,:,:])

	mse = tf.reduce_sum(tf.square(sr[0,:,:,0]-train_labels[101,:,:]))
	mse = mse /( 21.0 * 21.0 )
	mse_val = sess.run(mse)
	print ('MSE: ', mse_val)
	psnr = 20*np.log10(255.0) - 10*np.log10(mse_val)
	print ('psnr value: ', psnr)

"""
	prediction = cnn_model_fn(train_data[101,:,:])

	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

	accu = 0.0
	#accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	for (epoch_x , epoch_y) in get_batch(eval_data, onehot_labels2, 100):
		accu += correct.eval({x:epoch_x, y:epoch_y})

	accuracy = np.sum(accu)/float(eval_data.shape[0])
	print('Accuracy :' , accuracy, '%')
"""


