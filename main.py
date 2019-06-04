import tensorflow as tf
import numpy as np
from data_model import StockDataModel
import matplotlib.pyplot as plt
from rnn import RNN
import pickle
import datetime

# ticker_list = ['MSFT', 'AAPL', 'AMZN', 'FB', 'BRK.B', 'JNJ', 'JPM', 'GOOG', 'GOOGL', 'XOM', 'V', 'BAC', 'PG', 'CSCO', 'DIS', 'VZ', 'PFE', 'UNH', 'INTC', 'HD']

# ticker_list = ['MSFT', 'AAPL', 'AMZN', 'FB', 'BRK.B', 'SPY']
paras_list = [100, 1, 1, 300, 2, 0.001, 1.0, 50, 100]
ticker_list = ['SPY']
window_sizes = [5, 10, 30, 50, 100]

'''
paras list:
    window_length: length of sequence window
    input_dim
    output_dim
    lstm_size: number of neurons per lstm layer
    num_layers: number of lstm layers
    init_learning_rate: initial learning rate for AdamOptimizer
    keep_prob: probability of keeping a neuron in dropout layer
    batch_size
    num_epochs
    stride: how many steps to move the sliding window each time
'''
# =======================================================
# Section Used for window_size Exploration
# =======================================================
for ticker in ticker_list:
    for w in window_sizes:
        tf.reset_default_graph()
        print('******************************')
        print('Training: ', ticker, 'with window_length ', w)
        sdm = StockDataModel(ticker)
        all_sequence = sdm.create_sequence(window_length=w, stride=1, normalize=True)
        rnn_model = RNN([w, 1, 1, 300, 2, 0.001, 1.0, 50, 100], all_sequence, ticker)

        rnn_model.train()
        rnn_model.evaluate()
        print('******************************')
        print('')
