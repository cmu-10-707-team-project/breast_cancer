# reference: https://github.com/zhixuhao/unet/
#!/usr/bin/python
#     file: unet.py
#   author: Ziyi Cui
#  created: April 28th, 2019
#  purpose: xxx

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #

# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
import keras.backend as K
from keras import Input, Model
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, concatenate, \
    UpSampling2D, Lambda, Dropout, Conv2DTranspose, Concatenate
from keras.optimizers import Adam

from modeling.metrics import get_metrics


def unet(pretrained_weights = None,input_size = (256,256,3), lr=1e-4, **kwargs):
    inputs = Input((256, 256, 3))

    c1 = Conv2D(filters=16,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(normed)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(filters=16,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(filters=32,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(filters=32,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(filters=64,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(filters=64,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(filters=128,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(filters=128,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(filters=256,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(filters=256,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)

    c_b = Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='elu',
                 kernel_initializer='he_normal',
                 padding='same')(p5)
    c_b = Dropout(0.2)(c_b)
    c_b = Conv2D(filters=512,
                 kernel_size=(3, 3),
                 activation='elu',
                 kernel_initializer='he_normal',
                 padding='same')(c_b)

    u_b = Conv2DTranspose(filters=256,
                          kernel_size=(2, 2),
                          strides=(2, 2),
                          padding='same')(c_b)
    u_b = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 padding='same',
                 kernel_initializer='he_normal',
                 activation='elu')(Concatenate()([u_b, c5]))
    u_b = Dropout(0.2)(u_b)
    u_b = Conv2D(filters=256,
                 kernel_size=(3, 3),
                 padding='same',
                 kernel_initializer='he_normal',
                 activation='elu')(u_b)

    u6 = Conv2DTranspose(filters=128,
                         kernel_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(u_b)
    u6 = Conv2D(filters=128,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(Concatenate()([u6, c4]))
    u6 = Dropout(0.2)(u6)
    u6 = Conv2D(filters=128,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(u6)

    u7 = Conv2DTranspose(filters=64,
                         kernel_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(u6)
    u7 = Conv2D(filters=64,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(Concatenate()([u7, c3]))
    u7 = Dropout(0.2)(u7)
    u7 = Conv2D(filters=64,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(u7)

    u8 = Conv2DTranspose(filters=32,
                         kernel_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(u7)
    u8 = Conv2D(filters=32,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(Concatenate()([u8, c2]))
    u8 = Dropout(0.1)(u8)
    u8 = Conv2D(filters=32,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(u8)

    u9 = Conv2DTranspose(filters=16,
                         kernel_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(u8)
    u9 = Conv2D(filters=16,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(Concatenate()([u9, c1]))
    u9 = Dropout(0.1)(u9)
    u9 = Conv2D(filters=16,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(u9)

    output = Conv2D(filters=1,
                    kernel_size=(1, 1),
                    activation='sigmoid')(u9)
    output = Lambda(lambda x: K.squeeze(output, axis=3))

    model = Model(input = inputs, output = output)

    model.compile(
        optimizer = Adam(lr = lr), loss = 'binary_crossentropy',
        metrics=get_metrics())
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
