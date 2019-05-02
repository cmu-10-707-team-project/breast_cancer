import keras.backend as K
from tensorflow.python.estimator import keras


def get_metrics():
    return [mean_pred, false_pos_rate, false_neg_rate, accuracy, logloss, neg_mean, pos_mean, neg_std, pos_std]


def mean_pred(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true, y_pred = _mask_to_prob(y_true, y_pred)

    return K.mean(y_pred)


def false_pos_rate(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true, y_pred = _mask_to_prob(y_true, y_pred)

    y_true = K.cast(K.less(0.5, y_true), 'float32')
    y_pred = K.cast(K.less(0.5, y_pred), 'float32')

    fp = ((y_true * (1 - y_pred)) + (1 - y_true) * y_pred) * y_pred
    return K.sum(fp) / K.cast(K.shape(y_true)[0], 'float32')


def false_neg_rate(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true, y_pred = _mask_to_prob(y_true, y_pred)

    y_true = K.cast(K.less(0.5, y_true), 'float32')
    y_pred = K.cast(K.less(0.5, y_pred), 'float32')

    fn = ((y_true * (1 - y_pred)) + (1 - y_true) * y_pred) * (1 - y_pred)
    return K.sum(fn) / K.cast(K.shape(y_true)[0], 'float32')


def accuracy(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true, y_pred = _mask_to_prob(y_true, y_pred)

    y_true = K.cast(K.less(0.5, y_true), 'float32')
    y_pred = K.cast(K.less(0.5, y_pred), 'float32')

    return keras.metrics.binary_accuracy(y_true, y_pred)


def logloss(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true, y_pred = _mask_to_prob(y_true, y_pred)

    return keras.metrics.binary_crossentropy(y_true, y_pred)


def neg_mean(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true, y_pred = _mask_to_prob(y_true, y_pred)

    mask = 1 - y_true
    return K.sum(y_pred * mask) / K.sum(mask + K.epsilon())


def pos_mean(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true, y_pred = _mask_to_prob(y_true, y_pred)
    return K.sum(y_true * y_pred) / K.sum(y_true + K.epsilon())


def neg_std(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true, y_pred = _mask_to_prob(y_true, y_pred)
    mask = 1 - y_true
    mean = K.sum(y_pred * mask) / K.sum(mask)

    return K.sqrt(K.sum(K.square(y_pred - mean) * mask)
                  / (K.sum(mask) + K.epsilon()))


def pos_std(y_true, y_pred):
    if len(y_true.shape) == 3:
        y_true, y_pred = _mask_to_prob(y_true, y_pred)
    mean = K.sum(y_true * y_pred) / (K.sum(y_true) + K.epsilon())

    return K.sqrt(K.sum(K.square(y_pred - mean) * y_true)
                  / (K.sum(y_true) + K.epsilon()))


def _mask_to_prob(y_true, y_pred):
    y_true = K.mean(y_true, axis=(1, 2))
    y_pred = K.mean(y_pred, axis=(1, 2))
    return y_true, y_pred
