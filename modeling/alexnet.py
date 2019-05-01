# reference: https://github.com/zhixuhao/unet/
#!/usr/bin/python
#     file: alexnet.py
#   author: Ziyi Cui
#  created: April 28th, 2019
#  purpose: xxx


# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
from keras.layers import *
from keras.models import *
from keras.optimizers import *

from modeling.metrics import get_metrics


# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
def alexnet(pretrained_weights=None, lr=1e-3):
	model = Sequential()

	# 1st Convolutional Layer
	model.add(Conv2D(filters=96, input_shape=(256,256,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
	model.add(Activation('relu'))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

	# 2nd Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

	# 3rd Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))

	# 4th Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))

	# 5th Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

	# Passing it to a Fully Connected layer
	model.add(Flatten())
	# 1st Fully Connected Layer
	model.add(Dense(4096, input_shape=(256*256*3,)))
	model.add(Activation('relu'))
	# Add Dropout to prevent overfitting
	model.add(Dropout(0.4))

	# 2nd Fully Connected Layer
	model.add(Dense(4096))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))

	# 3rd Fully Connected Layer
	model.add(Dense(1000))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))

	# Output Layer
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.summary()

	# Compile the model
	
	model.compile(
        optimizer = Adam(lr = lr), loss = 'binary_crossentropy',
        metrics=get_metrics())

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model


