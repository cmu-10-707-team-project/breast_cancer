# Code built based on :
# https://github.com/hujunxianligong/Tensorflow-CNN-Tutorial/blob/master/cnn.py
# https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    num = 0
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname != ".DS_Store":
            fpaths.append(fpath)
            image = Image.open(fpath)
            data = np.array(image) / 255.0
            #print(data.shape)
            label = int(fname.split("_")[0])
            datas.append(data)
            labels.append(label)
            num += 1

    datas = np.array(datas)
    labels = np.array(labels)
    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels, num


def network(labels):
    num_classes = len(set(labels))

    datas_phr = tf.placeholder(tf.float32, [None, 1024, 450, 3])
    labels_phr = tf.placeholder(tf.int32, [None])
    dropout_phr = tf.placeholder(tf.float32)

    conv0 = tf.layers.conv2d(datas_phr, 20, 5, activation=tf.nn.relu)
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

    conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

    flatten = tf.layers.flatten(pool1)

    fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

    dropout_fc = tf.layers.dropout(fc, dropout_phr)

    logits = tf.layers.dense(dropout_fc, num_classes)

    predicted_labels = tf.arg_max(logits, 1)

    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(labels_phr, num_classes),
        logits=logits
    )
    mean_loss = tf.reduce_mean(losses)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(losses)

    saver = tf.train.Saver()
    return (datas_phr, labels_phr, dropout_phr), (predicted_labels,logits, saver), (losses, mean_loss, optimizer, saver)


def polt_loss(train_loss_list):
    plt.plot(list(range(len(train_loss_list))), train_loss_list)
    plt.grid(axis='x')
    plt.ylabel("cross-entropy error")
    plt.xlabel("number of epochs")
    plt.title("Average cross-entropy loss against number of epochs")
    plt.show()
    return


def train(datas, labels, model_path, placeholders, params, epochs):
    datas_placeholder, labels_placeholder, dropout_placeholdr = placeholders
    losses, mean_loss, optimizer, saver = params
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            # dropout
            dropout_placeholdr: 0
        }
        print("Training")
        mean_loss_list = []
        for step in range(epochs):
            print("step={}".format(step))
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
            mean_loss_list.append(mean_loss_val)
            print("step = {}\tmean loss = {}".format(step, mean_loss_val))
        saver.save(sess, model_path)
        print("Model saved to {}".format(model_path))
        polt_loss(mean_loss_list)
        return sess


def test(num, model_path, fpaths, datas, labels, placeholders, params):
    datas_placeholder, labels_placeholder, dropout_placeholdr = placeholders
    predicted_labels,logits, saver = params
    with tf.Session() as sess:
        print("Testing")
        saver.restore(sess, model_path)
        label_name_dict = {
            0: "normal",
            1: "tumor",
        }
        test_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        logits_val = sess.run(logits, feed_dict=test_feed_dict)


        correct_n = 0
        final_preds = []
        truth = []

        for fpath, real_label, predicted_label, logits in zip(fpaths, labels, predicted_labels_val, logits_val):

            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            final_preds.append(int(predicted_label))
            truth.append(int(real_label))
            if predicted_label_name == real_label_name:
                correct_n += 1
            print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))
            #print("softmax : {}".format(logits))

        final_preds = np.asarray(final_preds)
        truth = np.asarray(truth)
        TP = tf.count_nonzero(np.multiply(final_preds, truth))
        TN = tf.count_nonzero(np.multiply((final_preds - 1), (truth - 1)))
        FP = tf.count_nonzero(np.multiply(final_preds, (truth - 1)))
        FN = tf.count_nonzero(np.multiply((final_preds - 1), truth))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        print("Test accuracy : {}".format(correct_n/num))
        print("F1 socre : {}".format(f1.eval()))
        print("FP socre : {}".format(FP.eval()))
        print("FN socre : {}".format(FN.eval()))
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', dest="in_dir", help='Folder contains input images', required=True)
    parser.add_argument('-m',  dest="model_dir", help='Folder contains trained model', required=True)
    parser.add_argument('-e', dest = "epoch", help='Number of epoches', required=False, type=int)
    parser.add_argument('--train', dest='indicator', action='store_true')
    parser.add_argument('--test', dest='indicator', action='store_false')
    parser.set_defaults(indicator=True)

    args = parser.parse_args()
    in_dir = args.in_dir
    model_path = args.model_dir
    indicator = args.indicator
    epochs = args.epoch


    fpaths, datas, labels, num = read_data(in_dir)
    placeholders, test_params, train_params = network(labels)
    if indicator:
        # train model
        model = train(datas, labels, model_path, placeholders, train_params, epochs)
    else:
        # test mode
        test(num, model_path, fpaths, datas, labels, placeholders, test_params)
