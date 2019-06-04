import tensorflow as tf
import numpy as np
from data_model import StockDataModel
import matplotlib.pyplot as plt
from rnn import RNN
import pickle
import datetime

# =======================================================
# Section Used for Dropout keep_prob Exploration
# =======================================================
# dropout_list = [1.0, 0.95, 0.9, 0.8, 0.5]
# ticker = 'GOOGL'

# for d in dropout_list:
#     tf.reset_default_graph()
#     print('******************************')
#     print('Training: ', ticker, 'with dropout rate ', d)
#     sdm = StockDataModel(ticker)
#     all_sequence = sdm.create_sequence(window_length=10, stride=1, normalize=True)
#     rnn_model = RNN([10, 1, 1, 300, 2, 0.001, d, 50, 100], all_sequence, ticker)

#     rnn_model.train()
#     rnn_model.evaluate()
#     print('******************************')
#     print('')

# =======================================================
# Section Used for stride Exploration
# =======================================================
# stride_list = [1, 3, 5, 10, 20]
# ticker = 'GOOGL'

# for s in stride_list:
#     tf.reset_default_graph()
#     print('******************************')
#     print('Training: ', ticker, 'with stride ', s)
#     sdm = StockDataModel(ticker)
#     all_sequence = sdm.create_sequence(window_length=30, stride=s, normalize=True)
#     rnn_model = RNN([30, 1, 1, 300, 2, 0.001, 1.0, 50, 100, s], all_sequence, ticker)

#     rnn_model.train()
#     rnn_model.evaluate()
#     print('******************************')
#     print('')

ticker = 'GOOGL'
tf.reset_default_graph()
print('******************************')
print('Training: ', ticker)
sdm = StockDataModel(ticker)
all_sequence = sdm.create_sequence(window_length=30, stride=1, normalize=True)
rnn_model = RNN([30, 1, 1, 300, 2, 0.001, 1.0, 50, 100, 1], all_sequence, ticker)

rnn_model.train()
rnn_model.evaluate()
print('******************************')
print('')
