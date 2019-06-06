from data_model import StockDataModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import datetime


class RNN:
    def __init__(self, paras, data, name):
        '''
        data:
            np arrray, sequence data generated from StockDataModel in data_model.py
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
        name:
            identifier for model, set to be a stock's ticker symbol
        Note: model inspired from https://www.kaggle.com/raoulma/ny-stock-price-prediction-rnn-lstm-gru?scriptVersionId=2395222
        '''
        self.window_length = paras[0]
        self.input_dim = paras[1]
        self.output_dim = paras[2]
        self.lstm_size = paras[3]
        self.num_layers = paras[4]
        self.learning_rate = paras[5]
        self.keep_prob = paras[6]
        self.batch_size = paras[7]
        self.num_epochs = paras[8]
        self.stride = paras[9]
        self.name = name

        self.data = data
        self.split_data()
        self.build_graph()

    def shuffle_index(self):
        np.random.shuffle(self.random_index_array)

    def split_data(self, percent_test=0.1, percent_valid=0.1):
        size_test = int(percent_test * self.data.shape[0])
        size_v = int(percent_valid * self.data.shape[0])
        size_train = self.data.shape[0] - size_test - size_v

        self.train_seq = self.data[:size_train, :-1, :]
        self.train_pred = self.data[:size_train, -1, :]

        self.valid_seq = self.data[size_train:size_train + size_v, :-1, :]
        self.valid_pred = self.data[size_train:size_train + size_v, -1, :]

        self.test_seq = self.data[size_train + size_v:, :-1, :]
        self.test_pred = self.data[size_train + size_v:, -1, :]

        self.iterator_index = 0
        self.random_index_array = np.arange(self.train_seq.shape[0])
        self.shuffle_index()

    def get_next_batch(self):
        if self.iterator_index > self.train_seq.shape[0]:
            # finished one epoch
            self.iterator_index = 0
            start = self.iterator_index
            self.shuffle_index()
        else:
            start = self.iterator_index
            self.iterator_index += self.batch_size
        return self.train_seq[self.random_index_array[start:start + self.batch_size]], self.train_pred[self.random_index_array[start:start + self.batch_size]]

    def individual_cell(self):
        cell = tf.contrib.rnn.LSTMCell(num_units=self.lstm_size, activation=tf.nn.leaky_relu)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.prob)
        return cell

    def build_graph(self):
        tf.reset_default_graph()
        # input sequence length is all but the last one in the window, which is used to predict the last one
        self.X = tf.placeholder(tf.float32, [None, self.window_length - 1, self.input_dim], name='X')
        self.pred = tf.placeholder(tf.float32, [None, self.output_dim], name='pred')
        self.prob = tf.placeholder_with_default(self.keep_prob, shape=())
        self.stacked_rnn = tf.contrib.rnn.MultiRNNCell([self.individual_cell() for _ in range(self.num_layers)])
        # rnn outputs are a sequence of length window x lstm_size
        # rnn output dimension [batch_size, window_length, lstm_size]
        self.rnn_outputs, self.rnn_states = tf.nn.dynamic_rnn(self.stacked_rnn, self.X, dtype=tf.float32)
        # dynamic rnn output dimension [batch_size, window_length, lstm_size]
        # pass to fully-connected layer to map to output dimension
        self.stacked_outputs = tf.reshape(self.rnn_outputs, [-1, self.lstm_size])
        self.dense_outputs = tf.layers.dense(self.stacked_outputs, self.output_dim)
        self.all_pred = tf.reshape(self.dense_outputs, [-1, self.window_length - 1, self.output_dim])
        self.final_pred = self.all_pred[:, self.window_length - 2, :]

        self.loss = tf.reduce_mean(tf.square(self.final_pred - self.pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def train(self):
        # best_loss = float('inf')  # used for early stopping
        with tf.Session() as sess:
            sess.run(self.init)
            for e in range(self.num_epochs):
                # iteration for number of batches in one epoch
                for batch_i in range(math.ceil(self.train_seq.shape[0] / self.batch_size)):
                    x_batch, y_batch = self.get_next_batch()
                    sess.run(self.train_op, feed_dict={self.X: x_batch, self.pred: y_batch})

                if e % 5 == 0:
                    train_loss = self.loss.eval(feed_dict={self.X: self.train_seq, self.pred: self.train_pred, self.prob: 1.0})
                    valid_loss = self.loss.eval(feed_dict={self.X: self.valid_seq, self.pred: self.valid_pred, self.prob: 1.0})
                    print('%d epoch: train_loss: %.6f, valid_loss: %.6f' % (e, train_loss, valid_loss))
                    # implements early stopping
                    # stop if no progress in 10 epochs
                    # if e % 10 == 0:
                    #     if valid_loss <= best_loss:
                    #         best_loss = valid_loss
                    #     else:
                    #         break
                    # if valid_loss < 0.00020:
                    #     break

            try:
                os.makedirs('models/' + self.name + '/')
            except:
                print('Directory exists')
            # =======================================================
            # Different saving path for different purpose (window_length, keep_prob, stride)
            # =======================================================
            # save_path = self.saver.save(sess, './models/demo_model.ckpt')
            # save_path = self.saver.save(sess, './models/' + self.name + '/' + self.name + '-' + str(self.window_length) + '_model/model.ckpt')
            save_path = self.saver.save(sess, './models/' + self.name + '/' + self.name + '-' + str(self.keep_prob) + '_model/model.ckpt')
            # save_path = self.saver.save(sess, './models/' + self.name + '/' + self.name + '-stride-' + str(self.stride) + '_model/model.ckpt')
            print('Model saved in path: %s' % (save_path))

    def evaluate(self, test_seq=None, test_pred=None):
        if test_seq == None or test_pred == None:
            test_seq = self.test_seq
            test_pred = self.test_pred
        self.build_graph()
        with tf.Session() as sess:
            # =======================================================
            # Different saving path for different purpose (window_length, keep_prob, stride)
            # =======================================================
            # self.saver.restore(sess, './models/demo_model.ckpt')
            # self.saver.restore(sess, './models/' + self.name + '/' + self.name + '-' + str(self.window_length) + '_model/model.ckpt')
            self.saver.restore(sess, './models/' + self.name + '/' + self.name + '-' + str(self.keep_prob) + '_model/model.ckpt')
            # self.saver.restore(sess, './models/' + self.name + '/' + self.name + '-stride-' + str(self.stride) + '_model/model.ckpt')
            print('Model restored')
            actual_test_pred = sess.run(self.final_pred, feed_dict={self.X: test_seq, self.prob: 1.0})
            test_loss = self.loss.eval(feed_dict={self.X: test_seq, self.pred: test_pred, self.prob: 1.0})
            print('final test loss: %.6f' % (test_loss))
            self.visualize(actual_test_pred)

    def predict(self, model_path, latest_seq=None):
        if latest_seq == None:
            latest_seq = self.data[-1, 1:, :].reshape([1, 49, 1])
            print("latest_seq: ", latest_seq.shape)
        self.build_graph()
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            print('Model restored for predictions')
            pred = self.final_pred.eval(feed_dict={self.X: latest_seq, self.prob: 1.0})
            print('*************************************')
            print('Latest Predictions: ')
            print('Given Sequence: ', latest_seq)
            print('Next Value PREDICTION: ', pred)

    def visualize(self, actual_test_pred):
        plt.figure(num=None, dpi=300)
        plt.plot(actual_test_pred[:, 0], color='red', label='model predictions')
        plt.plot(self.test_pred[:, 0], color='black', label='actual price')
        plt.title(self.name + ' Stock Price Predictions vs Actual')
        plt.xlabel('time in days')
        plt.ylabel('stock price normalized')
        plt.legend(loc='best')
        # plt.show()
        try:
            os.makedirs('figures/' + self.name + '/')
        except:
            print('Directory exists')
        plt.show()
        # =======================================================
        # Different saving path for different purpose (window_length, keep_prob, stride)
        # =======================================================
        # plt.savefig('./figures/' + self.name + '/' + self.name + '-' + str(self.window_length) + '.png')
        # plt.savefig('./figures/' + self.name + '/' + self.name + '-' + str(self.keep_prob) + '.png')
        # plt.savefig('./figures/' + self.name + '/' + self.name + '-stride-' + str(self.stride) + '.png')

    def print_params(self):
        '''
        self.window_length = paras[0]
        self.input_dim = paras[1]
        self.output_dim = paras[2]
        self.lstm_size = paras[3]
        self.num_layers = paras[4]
        self.learning_rate = paras[5]
        self.keep_prob = paras[6]
        self.batch_size = paras[7]
        self.num_epochs = paras[8]
        self.stride = paras[9]
        self.name = name
        '''
        print('=======Model Info=======')
        print('name: ', self.name)
        print('lstm_size: ', self.lstm_size)
        print('num_layers: ', self.num_layers)
        print('keep_prob: ', self.keep_prob)
        print('========================')
