import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
import glob
from plotread import *

import tensorflow as tf

def createmodel(opt, output_size):
    model = Sequential()
    # initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=1, seed=10)
    if opt.model == 'LSTM':
        model.add(LSTM(opt.hidden_size, return_sequences=True))
    elif opt.model == 'GRU':
        model.add(GRU(opt.hidden_size, return_sequences=True))
    elif opt.model == 'RNN':
        model.add(SimpleRNN(opt.hidden_size, return_sequences=True))
    # # INIT DEFAULT: 
    model.add(TimeDistributed(Dense(output_size)))
    # # INIT CUSTOM:
    # model.add(TimeDistributed(Dense(output_size, kernel_initializer='ones')))

    return model

def getcallbacks(opt):
    weightspath = opt.savepath + 'weights'
    isExist = os.path.exists(weightspath)
    if not isExist:
        os.makedirs(weightspath)
    filepath= weightspath + "/weights-{epoch:02d}-{val_loss:.6f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='auto', period=1)
    if opt.early_stop == False:
        callbacks_list = [checkpoint]
    elif opt.early_stop == True:
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=opt.patience,
            verbose=1, mode="auto")
        callbacks_list = [checkpoint, early_stop]

    return callbacks_list

def savehistory(history, opt):

    dat = np.array([history.history['loss'], history.history['val_loss']])
    a= np.column_stack((dat))
    np.savetxt(opt.savepath + 'training_history.dat', a, delimiter=' ')

    return

def loadweights(opt, model):
    checkpoint_dir = opt.savepath + "weights/*"
    list_files = glob.glob(checkpoint_dir)
    latest = max(list_files, key=os.path.getctime)
    print(latest)
    model.load_weights(latest)

    return model


def main():
    ###########################################################################
    # Parser Definition
    ###########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model', choices=['LSTM', 'GRU', 'RNN'], default='GRU')
    parser.add_argument('-epochs', type=int, default=10000)
    parser.add_argument('-hidden_size', type=int, default=8)
    parser.add_argument('-learning_rate', type=float, default=0.005)
    parser.add_argument(
        '-optimizer', choices=['sgd', 'rmsprop', 'adam'], default='adam')
    parser.add_argument('-datapath', type=str, default="./Dataset2/")
    parser.add_argument('-savepath', type=str, default="./Experiment1/GRU/8/Run11/")
    parser.add_argument('-extension', type=str, default=".dat")
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-plots', type=bool, default=False)
    parser.add_argument('-early_stop', type=bool, default=False)
    parser.add_argument('-patience', type=int, default=50)
    opt = parser.parse_args()

    ###########################################################################
    # Variables Definition
    ###########################################################################
    nin = ['time', 'Applied voltage']
    nout = ['time', 'Minimum gap']
    neurons = ['time','Applied voltage','Minimum gap']

    ## Decaying learning rate
    # initial_learning_rate * decay_rate ^ (step / decay_steps)
    initial_learning_rate = 0.005
    # lr_schedule = ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=50,
    #     decay_rate=0.9,
    #     staircase=True)
    lr_schedule = initial_learning_rate

    if opt.optimizer == 'adam':
        opt_function = Adam(learning_rate=lr_schedule)
    elif opt.optimizer == 'sgd':
        opt_function = SGD(learning_rate=opt.learning_rate)
    elif opt.optimizer == 'rmsprop':
        opt_function = RMSprop(learning_rate=opt.learning_rate)

    ###########################################################################
    # Read data
    ###########################################################################   
    files = getdata(opt.datapath, opt.extension)
    train, valid, test = splitdata(files)
    trainx, trainy = readdata(opt.datapath, train, neurons, nin, nout)
    validx, validy = readdata(opt.datapath, valid, neurons, nin, nout)
    testx, testy = readdata(opt.datapath, test, neurons, nin, nout)
    if opt.plots:
        plotdata(trainx, '/train_data', '/x', opt.model, opt.savepath)
        plotdata(trainy, '/train_data', '/y', opt.model, opt.savepath)
        plotdata(validx, '/valid_data', '/x', opt.model, opt.savepath)
        plotdata(validy, '/valid_data', '/y', opt.model, opt.savepath)
        plotdata(testx, '/test_data', '/x', opt.model, opt.savepath)
        plotdata(testy, '/test_data', '/y', opt.model, opt.savepath)
    ###########################################################################
    # Create and Train Model
    ###########################################################################
    output_size = 1
    model = createmodel(opt, output_size)

    # mse = tf.keras.losses.MeanSquaredError()
    # def loss_dp(y_true, y_pred):
    #     return mse(y_true,y_pred**2)

    model.compile(optimizer=opt_function, loss='mean_squared_error')
    # model.compile(optimizer=opt_function, loss=loss_dp)
    callbacks_list = getcallbacks(opt)
    history=model.fit(
        np.array(trainx),
        np.array(trainy),
        batch_size=opt.batch_size,
        epochs=opt.epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(np.array(validx), np.array(validy)),
        validation_freq=1,
        workers=12,
        use_multiprocessing=True
    )
    model.summary()
    savehistory(history, opt)
    model = loadweights(opt, model)
    model.save(opt.savepath + 'model.h5')

if __name__ == "__main__":
    main()