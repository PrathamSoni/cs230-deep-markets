import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
from data import get_sp, get_news
from models import getModel
from sklearn.metrics import *
import time
import multiprocessing
import argparse
import statistics

def train(multi=False, model_num=0, epochs=0, mode=0, type_final=None):
    train_sp, y_train, val_sp, y_val, test_sp, y_test = get_sp()
    train_news, val_news, test_news = get_news()

    if mode == 0:
        x_train = train_sp
        x_val = val_sp
        x_test = test_sp
        model = getModel(model_id=model_num)

    elif mode == 1:
        x_train = train_news
        x_val = val_news
        x_test = test_news
        model = getModel(model_id=model_num)

    elif mode == 2:
        x_train = [train_news, train_sp]
        x_val = [val_news, val_sp]
        x_test = [test_news, test_sp]
        model = getModel(model_id=model_num, type_final=type_final)

    opt = tf.keras.optimizers.Adam(learning_rate=.0001)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(), 'accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=.1 * epochs),
    ]
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=2,
                        batch_size=32,
                        validation_data=(x_val, y_val))

    results = model.evaluate(x_test, y_test, verbose=2)

    if not multi:
        predictions = model.predict(x_test)
        output_list = [0, 0]
        label_list = [0, 0]

        for i in range(predictions.shape[0]):
            output_list[int(np.round(predictions[i]))] += 1
            label_list[int(np.round(y_test[i]))] += 1
        print("predictions:\t", output_list)
        print("labels:\t", label_list)
        print(confusion_matrix(y_test, np.round(predictions)))
        print(classification_report(y_test, np.round(predictions)))
    return results


if __name__ == '__main__':
    use_GPU = True
    if not use_GPU:
        tf.config.experimental.set_visible_devices([], 'GPU')
        print('computing on CPU')
    else:
        print('computing on GPU')

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_num", type=int)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--type', type=int, default=None)
    parser.add_argument('--run_id', type=str, default='0')
    args = parser.parse_args()

    start = time.time()
    auc = []
    acc = []

    iters = args.iters

    if iters == 1:
        _, cur_auc, cur_acc = train(multi=False, model_num=args.model_num, epochs=args.epochs, mode=args.mode,
                                    type_final=args.type)
        auc.append(cur_auc)
        acc.append(cur_acc)
    else:
        for i in range(iters):
            _, cur_auc, cur_acc = train(multi=True, model_num=args.model_num, epochs=args.epochs, mode=args.mode,
                                        type_final=args.type)
            auc.append(cur_auc)
            acc.append(cur_acc)

    dur = time.time() - start
    print("Time: {}, iters: {}, average auc: {:.3f}, average acc: {:.3f}".format(dur, iters, statistics.mean(auc), statistics.mean(acc)))
    print("Time: {}, iters: {}, std auc: {:.3f}, std acc: {:.3f}".format(dur, iters, statistics.pstdev(auc), statistics.pstdev(acc)))
    with open('../logs/{}.txt'.format(args.run_id),'a') as f:
        f.write("Time: {}, iters: {}, average auc: {:.3f}, average acc: {:.3f}\n".format(dur, iters, statistics.mean(auc), statistics.mean(acc)))
        f.write("Time: {}, iters: {}, std auc: {:.3f}, std acc: {:.3f}\n".format(dur, iters, statistics.pstdev(auc), statistics.pstdev(acc)))
