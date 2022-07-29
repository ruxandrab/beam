import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path


def getdata(data_path, ext):
    files_list = []
    files = os.listdir(data_path)
    for file in files:
        if file.endswith(ext):
            files_list.append(file)

    return files_list

def splitdata(data_list):
    train = []
    valid = []
    test = []
    saved_list = []
    ## test_set = [36, 33, 30, 26, 19, 15, 13, 10, 7, 3]
    # test_set = [40, 36, 33, 30, 26, 19, 15, 13, 10, 7]
    # valid_set = [39, 34, 32, 27, 24, 20, 16, 12, 8, 2]
    

    # test_set = [83, 79, 69, 61, 52, 41, 31, 21, 16, 8]
    # valid_set = [80, 71, 64, 58, 46, 36, 25, 17, 14, 7]


    ## indices generated randomly in Matlab
    valid_set = [496, 486, 484, 480, 478, 466, 456, 452, 445, 443, 441, 437, 434, 429, 426, 422,
   415, 413, 408, 400, 397, 394, 389, 387, 383, 377, 372, 366, 362, 358, 351, 347,
   340, 337, 333, 327, 322, 318, 316, 310, 306, 304, 292, 286, 283, 277, 273, 265,
   261, 259, 254, 251, 239, 231, 229, 226, 223, 216, 212, 199, 190, 187, 180, 176,
   171, 168, 164, 162, 159, 157, 151, 146, 137, 129, 127, 122, 116, 112, 106,  98,
    93,  89,  86,  79,  76,  70,  65,  60,  55,  47,  42,  39,  33,  29,  26,  22,  17,  12,  5,  2]

    test_set = [499, 493, 485, 482, 479, 467, 463, 455, 446, 444, 442, 440, 435, 433, 428, 423,
   419, 414, 411, 401, 399, 395, 391, 388, 384, 380, 376, 369, 363, 361, 355, 350,
   342, 339, 334, 330, 324, 321, 317, 311, 308, 305, 294, 291, 284, 278, 275, 270,
   262, 260, 258, 252, 248, 234, 230, 227, 224, 217, 215, 203, 196, 188, 186, 178,
   175, 170, 165, 163, 161, 158, 155, 147, 140, 132, 128, 124, 120, 114, 111, 104,
    94,  92,  88,  82,  78,  74,  67,  64,  59,  52,  43,  41,  36,  30,  28,  25,  21,  15,  11,  3]


    for file in data_list:
        saved_list.append(file)
    for i in range(len(test_set)):
        test.append(data_list.pop(test_set[i]-1))
    for i in range(len(test_set)):
        valid.append(saved_list.pop(valid_set[i]-1))
    train = [x for x in data_list if x in saved_list]

    return train, valid, test


def readdata(data_path, file_list, cols, nin, nout):
    datax = []
    datay = []
    for file in file_list:
        df = pd.read_csv(data_path + file,
            sep = "\s+", #separator whitespace
            names=cols)
        df = df.set_index('time')
        x = df.drop(columns = nout[1:2])
        y = df.drop(columns = nin[1:5])
        datax.append(x)
        datay.append(y)

    return datax, datay

def plotdata(sequences, dtype, var, folder, path):
    
    i=0
    for seq in sequences:
        seq.plot(kind='line')
        plt.legend(loc='upper left')
        Path(path + folder + dtype).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + folder + dtype + var + str(i) + '_plots.pdf')
        plt.close()

        i = i + 1

