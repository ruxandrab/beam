# Recurrent neural networks for the dynamics of the membrane's displacement in the micromachined fixed-fixed beam 

This repository is the official implementation of "Machine learning techniques to model highly nonlinear multi-field dynamics"

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  The available code was developed using Python 3.7.
>📋  To run the code the libraries available in the "requirements.txt" file were used, with the referred version.
>📋  It is recommended to install them with the given command and with the same version of each library, although other versions might still work.

## Training

To train the model(s) in the paper, run this command:

```train
python train.py -model 'Type of Recurrent Layer' -epochs 10000 -hidden_size "Recurrent Layer Size" -learning_rate 0.005 -optimizer 'adam' -datapath <path do the data/> -savepath <path where to save the results to/> -batch_size 32 -plots True/False
```
>📋 The input parameters for the training script are:
    * -model: The model to use from, ['LSTM', 'GRU', 'RNN']
    * -epochs: The number of epochs to train the model
    * -hidden_size: The size of the recurrent neural network layer
    * -learning_rate: The value for the learning rate
    * -optimizer: The optimizer to use from ['sgd', 'rmsprop', 'adam']
    * -datapath: The path to take the data from
    * -savepath: The path where to save the weights, model and training history
    * -batch_size: Size of each batch of data used to train the model
    * -plots: Whether to plot the inputs and outputs from the dataset or not
    * -early_stop: Whether to use early stopping or not
    * -patience: The amount of epochs to wait with no improvement before stopping the model training in case -early_stop is true
>📋 The appropriate hyperparameters are learning_rate = 0.005 and batch_size = 32 for the model without parameters l,w,and air viscosity.
>📋 The rest of the parameters used for training each model depend on the different experiments and settings tried. To check the appropriate settings for each one go to the Results section of this README
>📋 This script only trains the model creating a folder with the weights saved in each iteration where the model improved, a training history data file with training and validation losses, "training_history.dat" and a saved model file "model.h5"

## Evaluation

To evaluate any of the models used run:

```eval
python eval.py -model 'Type of Recurrent Layer' -datapath <path do the data/> -savepath <path to the model.h5 file, where the model was previously saved/> -batch_size 32 -plots_in True/False -plots_out True/False
```

>📋 The input parameters for the training script are:
    * -model: The model to use from, ['LSTM', 'GRU', 'RNN'].
    * -datapath: The path to take the data from.
    * -savepath: The path where to save the model was saved.
    * -batch_size: Size of each batch of data used to evaluate the model.
    * -plots_in: Whether to plot the inputs and outputs from the dataset or not.
    * -plots_out: Whether to plot the results or not.
>📋 An example for each evaluation execution of each model can be found on the results section of this README
>📋 This script only evaluates the model creating a folder with .dat files containing the results and also creating a folder with plots of the results if -plots_out is True.

## Pre-trained Models

For each setting tried in each experiment there is an available model, "model.h5" in the corresponding folder.

## Results

This section contains the results with and without the parameters l,w,air viscosity. 

There are also available Matlab scripts to generate the some of the figures in the paper.


### Experiment 1

Experiment 1 compares the performances of three architectures: simple RNN, LSTM and GRU when predicting the output -- the minimum gap -- for a variety of different inputs -- applied voltages.
To replicate this experiment use the following training and/or evaluation commands:

>📋 RNN - Size 16

```train
python train.py -model 'RNN' -hidden_size 16 -datapath "Dataset2/" -savepath "Experiment1/RNN/16/Run*/"
```
```eval
python eval.py -model 'RNN' -datapath "Dataset2/" -savepath "Experiment1/RNN/16/Run*/"
```

>📋 LSTM - Size 8

```train
python train.py -model 'LSTM' -hidden_size 8 -datapath "Dataset2/" -savepath "Experiment1/LSTM/8/Run*/"
```
```eval
python eval.py -model 'LSTM' -datapath "Dataset2/" -savepath "Experiment1/LSTM/8/Run*/"
```

>📋 GRU - Size 8

```train
python train.py -model 'GRU' -hidden_size 8 -datapath "Dataset2/" -savepath "Experiment1/GRU/8/Run*/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset2/" -savepath "Experiment1/GRU/8/Run*/"
```


>📋 The Root Mean Squared Error [um] obtained during the supervised learning are summarized in the following table:

| Set / Model      |  RNN - 16  | LSTM - 8   |  GRU - 8   |  RNN - 64  | LSTM - 32  |  GRU - 32  | 
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Training Set     |  0.0856    | 0.0376     | 0.0304     |  0.1728    | 0.0243     |  0. 0505   |
| Validation Set   |  0.1335    | 0.0759     | 0.0896     |  0.1928    | 0.1781     |  0.1378    |


### Parameter-aware models

Experiment 2 trains a GRU on input-output data, where the input contains the applied voltage (v(t)), the membrane length l, 
the membrane width w and the air viscosity u. The dataset contains 500 examples with random combinations of input and parameter values. 
To replicate this experiment navigate to beam-param folder and use the following training and/or evaluation commands:   

```

>📋 GRU 

```train
python train.py -model 'GRU' -hidden_size 16 -batch_size 512 -datapath "Dataset3_param/" -savepath "Experiment2/GRU/16/Run*/"
```
```eval
python eval.py -model 'GRU' -batch_size 512 -datapath "Dataset3_param/" -savepath "Experiment1/GRU/16/Run*/"
```


## Contributing

>📋  The License is available in this repository.