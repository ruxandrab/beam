import argparse
import numpy as np
from keras.models import load_model
from plotread import *
import pandas as pd
from keras.layers import Normalization
import numpy as np

import tensorflow as tf

def evaluateresults(model, opt, trainx, trainy, validx, validy, testx, testy, output_size):

    # Evaluate error on training, validation and test data
    train_l = model.evaluate(np.array(trainx), np.array(trainy), batch_size=opt.batch_size,
        verbose=1, workers=12, use_multiprocessing=True)
    valid_l = model.evaluate(np.array(validx), np.array(validy), batch_size=opt.batch_size,
        verbose=1, workers=12, use_multiprocessing=True)
    test_l = model.evaluate(np.array(testx), np.array(testy), batch_size=opt.batch_size,
        verbose=1, workers=12, use_multiprocessing=True)
    
    # Actually predict the signal
    trainPredict = model.predict(np.array(trainx), batch_size=opt.batch_size, verbose=1,
        workers=12, use_multiprocessing=True)
    # plotresponse(opt, trainx, trainy, trainPredict, "train", output_size)
    plotresultstofile(opt, trainx, trainy, trainPredict, opt.model, "/train", opt.savepath, output_size)
    validPredict = model.predict(np.array(validx), batch_size=opt.batch_size, verbose=1,
        workers=12, use_multiprocessing=True)
    # plotresponse(opt, validx, validy, validPredict, "valid", output_size)
    plotresultstofile(opt, validx, validy, validPredict, opt.model, "/valid", opt.savepath, output_size)
    testPredict = model.predict(np.array(testx), batch_size=opt.batch_size, verbose=1,
        workers=12, use_multiprocessing=True)
    # plotresponse(opt, testx, testy, testPredict, "test", output_size)
    plotresultstofile(opt, testx, testy, testPredict, opt.model, "/test", opt.savepath, output_size)

    return train_l, valid_l, test_l

def plotresponse(opt, inputx, real, predicted, type, out_size):
    cols = ['Minimum gap predicted']

    i=0
    for frame in predicted:
        pos = real[i].shape[1]
        for j in range(out_size):
            real[i].insert(
                loc = pos,
                column = cols[j],
                value = frame[:,j]
            )
            pos = pos + 1
        
        ax = real[i][["Minimum gap"]].plot(kind='line', linewidth=2.0)
        real[i][["Minimum gap predicted"]].plot(ax=ax, kind='line', linestyle='dashed', marker='*', ms=4, linewidth=1.0, color='red')
        # inputx[i][["Air viscosity"]].plot(ax=ax, kind='line', linewidth=1.0, color='black')
        plt.xlabel("Time [msec]", fontsize=16)
        plt.ylabel("Minimum gap [um]", fontsize=16)
        plt.title(type + str(i))

        if opt.plots_rt:
            plt.legend(loc='upper right', fontsize=14)
            # plt.legend().remove()
            plt.show()

        i = i + 1

def plotresultstofile(opt, inputx, real, predicted, folder, string, path, out_size):
    cols = ['Minimum gap predicted']
    # colsin = ['Input: Applied voltage']

    i=0
    for frame in predicted:
        # print('i=',str(i))
        pos = real[i].shape[1]  # no of columns of real[i]
        for j in range(out_size):   # for each output of this example; j=0
            # print(frame[:,j])
            # print(type(frame[:,j]))
            # print('pos='+str(pos))  # pos=1
            real[i].insert(
                loc = pos,
                column = cols[j],
                value = frame[:,j]
            )
            pos = pos + 1
        
        # i = i + 1

    
    # i=0
    # for framein in inputx:
    #     posin = real[i].shape[1]
    #     print('pos='+str(pos))  # pos=2
    #     print(framein)
    #     print(type(real[i]))
    #     xx = np.array(framein)
    #     # print(np.array(framein['Applied voltage']))
    #     print(type(xx))
    #     real[i].insert(
    #         loc = posin,
    #         column = colsin,
    #         value = xx
    #     )
        
        ax = real[i][["Minimum gap"]].plot(kind='line', linewidth=2.0)
        real[i][["Minimum gap predicted"]].plot(ax=ax, kind='line', linestyle='dashed', marker='*', ms=4, linewidth=1.0, color='red')
        inputx[i][["Applied voltage"]].plot(ax=ax, kind='line', linewidth=1.0, color='black')
        plt.xlabel("Time [msec]", fontsize=16)
        plt.ylabel("Minimum gap [um]", fontsize=16)

        # print(type(inputx[i][["Air viscosity"]]))
        # print(np.array(inputx[i][["Air viscosity"]]).shape)
        airvisc = np.array(inputx[i][["Air viscosity"]])[1]
        mlength = np.array(inputx[i][["Length"]])[1]
        mwidth  = np.array(inputx[i][["Width"]])[1]

        Path(path + "results_files").mkdir(parents=True, exist_ok=True)
        real[i].to_csv(
            path + "results_files/" + 'data_' + str(mlength) + '_' + str(mwidth) + '_' + str(airvisc) + '_' + str(i) + string.lstrip('/') + '.dat', sep=' ', header=False)
        if opt.plots_out:
            plt.legend(loc='upper right', fontsize=14)
            # plt.legend().remove()
            Path(path + "results_plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(path + "results_plots" + '/response'  + '_' + str(mlength) + '_' + str(mwidth) + '_' + str(airvisc) + '_' + str(i) + string.lstrip('/') + '.pdf')
            plt.close()

        i = i + 1

def main():
    ###########################################################################
    # Parser Definition
    ###########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model', choices=['LSTM', 'GRU', 'RNN'], default='GRU')
    parser.add_argument('-datapath', type=str, default="Dataset3_param/")
    parser.add_argument('-savepath', type=str, default="Experiment2/GRU/16/Run10/")
    parser.add_argument('-extension', type=str, default=".dat")
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-plots_in', type=bool, default=False)
    parser.add_argument('-plots_out', type=bool, default=True)
    parser.add_argument('-plots_rt', type=bool, default=True)
    opt = parser.parse_args()

    ###########################################################################
    # Variables Definition
    ###########################################################################
    nin = ['time', 'Applied voltage', 'Length', 'Width', 'Air viscosity']
    nout = ['time', 'Minimum gap']
    neurons = ['time','Applied voltage', 'Length', 'Width', 'Air viscosity', 'Minimum gap']
    # nin = ['time', 'Applied voltage', 'Air viscosity']
    # nout = ['time', 'Minimum gap']
    # neurons = ['time','Applied voltage', 'Air viscosity', 'Minimum gap']

    ###########################################################################
    # Read data
    ###########################################################################
    files = getdata(opt.datapath, opt.extension)
    train, valid, test = splitdata(files)
    trainx, trainy = readdata(opt.datapath, train, neurons, nin, nout)
    validx, validy = readdata(opt.datapath, valid, neurons, nin, nout)
    testx, testy = readdata(opt.datapath, test, neurons, nin, nout)
    if opt.plots_in:
        plotdata(trainx, '/train_data', '/x', opt.model, opt.savepath)
        plotdata(trainy, '/train_data', '/y', opt.model, opt.savepath)
        plotdata(validx, '/valid_data', '/x', opt.model, opt.savepath)
        plotdata(validy, '/valid_data', '/y', opt.model, opt.savepath)
        plotdata(testx, '/test_data', '/x', opt.model, opt.savepath)
        plotdata(testy, '/test_data', '/y', opt.model, opt.savepath)
    ###########################################################################
    # Load Model and Evaluate
    ###########################################################################
    output_size = 1

    # mse = tf.keras.losses.MeanSquaredError()
    # def loss_dp(y_true, y_pred):
    #     return mse(y_true,y_pred**2)

    ###########################################################################
    # Pre-process data
    ###########################################################################
    # normalizerx = Normalization(axis=None, input_shape=(101,4))
    # normalizerx = Normalization(axis=None, input_shape=(101,2))
    # normalizerx.adapt(np.array(trainx))
    # layernormx = normalizerx(np.array(trainx))


    # model = load_model(opt.savepath + 'model.h5', custom_objects={"loss_dp": loss_dp})
    model = load_model(opt.savepath + 'model.h5')
    model.summary()
    lt, lv, ltt = evaluateresults(
        model, opt, trainx, trainy, validx, validy, testx, testy, output_size)
    # lt, lv, ltt = evaluateresults(
    #     model, opt, normalizerx(np.array(trainx)), trainy, normalizerx(np.array(validx)), validy, normalizerx(np.array(testx)), testy, output_size)
    print("Training Loss: " + str(lt))
    print("Validation Loss: " + str(lv))
    print("Test Loss: " + str(ltt))

if __name__ == "__main__":
    main()