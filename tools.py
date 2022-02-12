import numpy as np
import time

# from _bolster import bolstering
# from error_estimator import generalized_resub
import _bolster
import error_estimator

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def get_x_perClass(x, y):
    xtr = np.array(x)
    ytr = np.array(y)
    classes = sorted(set(ytr.reshape(len(ytr),)))
    xx = {y_cls: xtr[ytr == y_cls, :] for y_cls in classes}
    return xx
                    
def create_FC(input_shape, num_classes,
              num_hidden_layers, num_units,
              optimizer_ = 'Adam', dropout_rate= 0.0):    
    clf = Sequential()
    clf.add(keras.Input(shape=input_shape))
    clf.add(layers.Flatten())
    for i in range(num_hidden_layers):
        clf.add(layers.Dense(units=num_units, activation='relu'))
        if (dropout_rate > 0.0):
            clf.add(layers.Dropout(rate=dropout_rate))    
    clf.add(layers.Dense(units=num_classes, activation = 'softmax'))
    
    step = tf.Variable(0, trainable=False)
    if optimizer_ == 'Adam':
        boundaries = [100, 200, 300, 400, 500, 600, 700]
        values = [0.001, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
    elif optimizer_ == 'SGD':
        boundaries = [50, 100, 150, 200]
        values = [0.1, 0.05, 0.01, 0.005, 0.001]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    lr_piecewiseConstant = learning_rate_fn(step)
    
    if optimizer_ == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr_piecewiseConstant)
    elif optimizer_ == 'SGD':
        opt =  tf.keras.optimizers.SGD(learning_rate=lr_piecewiseConstant)
    clf.compile(loss='categorical_crossentropy',
                optimizer= opt,
                metrics=['accuracy']) 
    return clf

def get_err(xtr, ytr, xts, yts, model, factor, mc_samples):
    # sss = time.time()

    test_err, standard_resub = [], []
    bol_err, pp_err = [], []
    bolpp_err = []

    ee = error_estimator.generalized_resub(x=xtr, y=ytr,
                                           classifier=model,
                                           mc_samples=mc_samples)

    pp_mc, psi_mc = ee.compute_mc_pp_psi(mc_samples)

    tst = ee.standard(xts, yts)  # test-set err
    test_err += [tst]
    print('\n\t test_err = ', tst)
    standard_resub += [ee.standard()-tst]  # standard resub
    bol_err += [ee.bolster(mc_samples, psi_mc)-tst]
    pp_err += [ee.pp()-tst]
    bolpp_err += [ee.bolster_pp(mc_samples, pp_mc, psi_mc)-tst]
    print('stanadard={s} \n bol={b} \n pp={p}, \n bolpp={bp} \n '.format(
        s= standard_resub, b=bol_err, p=pp_err, bp=bolpp_err))
    
    return {'test_info': test_err,
            'standard': standard_resub,
            'bolster': bol_err,
            'pp': pp_err,
            'bolster_pp': bolpp_err}
          
def run_exp(xtr, ytr, xts, yts, 
            width_list, depth_list, 
            batch_size_list, dropout_rate_list,
            n_mc_sample=100):
    num_classes = len(set(np.array(ytr).reshape(len(ytr),)))
    y_train_one_hot = keras.utils.to_categorical(ytr, num_classes)
    # y_test_one_hot = keras.utils.to_categorical(yts, num_classes)    
    # n, dim = xtr.shape
    mc = _bolster.bolstering(x=xtr, y=ytr,
                             bolstering_type='original'
                             ).generate_mc_sample(n_mc_sample=n_mc_sample,
                                                  save_to_file=False,
                                                  input_shape=None,
                                                  filename=None)
    callback = [StopOnPoint(0.990)]  # <- set optimal point
    bs = batch_size_list[0]
    dr = dropout_rate_list[0]
    err_hidlay = {}
    for num_hidden_layers in depth_list:
        print ('num H Layers = ', str(num_hidden_layers))
        err_units = []
        for units in width_list:
            print('\t num units = ', str(units))
            clf_FC = create_FC(input_shape=xtr.shape[1:],
                               num_classes=num_classes,
                               num_hidden_layers=num_hidden_layers,
                               num_units=units,
                               optimizer_='Adam', 
                               dropout_rate=dr)

            clf_FC.fit(xtr, y_train_one_hot,
                       batch_size=bs,
                       epochs=800,
                       callbacks=callback,
                       verbose=0)

            training_history = clf_FC.history.history
            err = get_err(xtr, ytr, xts, yts,
                          model=clf_FC, factor=1, mc_samples=mc)

            err['history'] = training_history
            
            err_units += [err]
            del clf_FC
        err_hidlay[str(num_hidden_layers)]= err_units 

    return err_hidlay

class StopOnPoint(tf.keras.callbacks.Callback):
    '''
    source for this class StopOnPoint:
        https://stackoverflow.com/questions/67216419/stop-training-model-when-accuracy-reached-greater-than-0-99/67217202
    '''
    def __init__(self, point):
        super(StopOnPoint, self).__init__()
        self.point = point    
    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["accuracy"]
        if accuracy >= self.point:
            self.model.stop_training = True                    
                    
