import pandas as pd
import matplotlib.pyplot as plt

##################################################################################################
##################################################################################################
# Experiment 1 
##################################################################################################
##################################################################################################
mainfolder1 = "beam/"
folder1_list = ["RNN/16/Run1/results_files/test", "LSTM/8/Run1/results_files/test", "GRU/8/Run1/results_files/test",
    "RNN/64/Run1/results_files/test", "LSTM/32/Run1/results_files/test", "GRU/32/Run1/results_files/test"]
sequences1 = ["1"] #You can choose here more sequences from 0 to 9
end1 = "_data.dat"
cols1 = ["Minimum gap", "Minimum gap predicted"]
cols2 = ["Minimum gap", "Minimum gap predicted"]
color1 = ["blue"]
color2 = ["red"]
save1 = ["16RNN", "8LSTM", "8GRU", "64RNN", "32LSTM", "32GRU"]
head = ["time", "Minimum gap", "Minimum gap predicted"]

i = 0
for f in folder1_list:
    for s in sequences1:
        df = pd.read_csv(mainfolder1 + f + s + end1,
            sep = "\s+",
            names=head)
        df = df.set_index('time')

        fig, ax = plt.subplots()
        df[cols11].plot(ax=ax, kind='line', color=color1)
        df[cols12].plot(ax=ax, kind='line', linestyle='dashed', linewidth=2.0, marker='None', color=color1)
        ax.get_legend().remove()
        plt.xlabel("Time [msec]", fontsize=16)
        plt.ylabel("Minimum gap [um]", fontsize=16)
        plt.savefig(mainfolder1 + 'Seq' + s + '_p1_' + save1[i] + '.pdf',
            bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots()
        df[cols21].plot(ax=ax, kind='line', color=color2)
        df[cols22].plot(ax=ax, kind='line', linestyle='dashed', linewidth=2.0, marker='None', color=color2)
        ax.get_legend().remove()
        plt.xlabel("Time [msec]", fontsize=16)
        plt.ylabel("Minimum gap [um]", fontsize=16)
        plt.savefig(mainfolder1 + 'Seq' + s + '_p2_' + save1[i] + '.pdf',
            bbox_inches='tight')
        plt.close()

    i = i + 1
