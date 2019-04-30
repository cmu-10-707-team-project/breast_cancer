import keras.backend as K
from tensorflow.python.estimator import keras


def mean_pred(_, y_pred):
    return K.mean(y_pred)


def false_pos_rate(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true = K.max(y_true, axis=(1, 2))
        y_pred = K.max(y_pred, axis=(1, 2))

    y_true = K.cast(K.less(0.5, y_true), 'float32')
    y_pred = K.cast(K.less(0.5, y_pred), 'float32')

    fp = ((y_true * (1 - y_pred)) + (1 - y_true) * y_pred) * y_pred
    return K.sum(fp) / K.cast(K.shape(y_true)[0], 'float32')


def false_neg_rate(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true = K.max(y_true, axis=(1, 2))
        y_pred = K.max(y_pred, axis=(1, 2))

    y_true = K.cast(K.less(0.5, y_true), 'float32')
    y_pred = K.cast(K.less(0.5, y_pred), 'float32')

    fn = ((y_true * (1 - y_pred)) + (1 - y_true) * y_pred) * (1 - y_pred)
    return K.sum(fn) / K.cast(K.shape(y_true)[0], 'float32')


def accuracy(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true = K.max(y_true, axis=(1, 2))
        y_pred = K.max(y_pred, axis=(1, 2))

    y_true = K.cast(K.less(0.5, y_true), 'float32')
    y_pred = K.cast(K.less(0.5, y_pred), 'float32')

    return keras.metrics.binary_accuracy(y_true, y_pred)


def logloss(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true = K.mean(y_true, axis=(1, 2))
        y_pred = K.mean(y_pred, axis=(1, 2))

    return keras.metrics.binary_crossentropy(y_true, y_pred)


# def patch_mean_iou(y_true, y_pred):
#     y_true = K.cast(K.less(0.5, y_true), 'float32')
#     y_pred = K.cast(K.less(0.5, y_pred), 'float32')
#
#     intersect = K.sum(y_true * y_pred, axis=(1, 2))
#     union = K.sum(1 K.cast(K.less(1., y_true + y_pred), 'float32'), axis=(1, 2))
# b
#     return K.mean(intersect / union)
