# reference: https://github.com/zhixuhao/unet/
#!/usr/bin/python
#     file: unet.py
#   author: Ziyi Cui
#  created: April 28th, 2019
#  purpose: xxx


# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras.backend as K


# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
from modeling.metrics import mean_pred, false_pos_rate, false_neg_rate, \
    accuracy, logloss, get_metrics


def unet(pretrained_weights = None,input_size = (256,256,3), lr=1e-4, **kwargs):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([conv4,up6], axis = 3)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3,up7], axis = 3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    # conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    # conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    # up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2,up8], axis = 3)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    # conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    # up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1,up9], axis = 3)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    # conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv9 = BatchNormalization()(conv9)

    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    conv10 = Lambda(lambda x: K.squeeze(x, axis=3))(conv10)

    model = Model(input = inputs, output = conv10)

    model.compile(
        optimizer = Adam(lr = lr), loss = 'binary_crossentropy',
        metrics=get_metrics())
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
