from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam

from modeling import unet
from modeling import alexnet
from modeling.metrics import mean_pred, false_pos_rate, false_neg_rate, \
accuracy, logloss


def get_model(model_name, lr, **kwargs):
    ########################
    
    if model_name == 'unet':
        model = unet.unet(lr=lr)

    ########################

    elif model_name == 'alexnet':
        model = alexnet.alexnet(lr=lr)
    
    ########################

    elif model_name == 'resnet50':
        from keras.applications.resnet50 import ResNet50
        base_model = ResNet50(weights='imagenet',include_top=False)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1,activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer = Adam(lr = lr), loss = 'binary_crossentropy',
            metrics=[accuracy, mean_pred, false_pos_rate, false_neg_rate, logloss])
    
    ########################
    
    elif model_name == 'vgg16':
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights='imagenet', include_top=False)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=Adam(lr=lr), loss='binary_crossentropy',
            metrics=[accuracy, mean_pred, false_pos_rate, false_neg_rate,
                     logloss])
    
    ########################
    
    elif model_name == 'vgg19':
        from keras.applications.vgg19 import VGG19
        base_model = VGG19(weights='imagenet',include_top=False)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1,activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer = Adam(lr = lr), loss = 'binary_crossentropy',
            metrics=[accuracy, mean_pred, false_pos_rate, false_neg_rate, logloss])

    ########################
    else:
        raise ValueError('unknown model')

    return model
