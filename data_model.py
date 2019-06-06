import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
import fix_yahoo_finance as yf


class StockDataModel:
    def __init__(self, ticker):
        self.name = ticker
        # Date is index column
        self.data = yf.download(ticker, period='max', actions=False)
        # self.data = pd.read_csv('SPY.csv', index_col=0)
        # self.data.info()

        self.data.drop(['High'], 1, inplace=True)
        self.data.drop(['Low'], 1, inplace=True)
        self.data.drop(['Open'], 1, inplace=True)
        self.data.drop(['Volume'], 1, inplace=True)
        self.data.drop(['Close'], 1, inplace=True)
        self.normalized_data = self.normalize()

    def data_info():
        self.data.info()

    def plot_prices(self, normlized=False):
        print_frequency = len(self.data.index) // 10
        # plt.figure(num=None, dpi=300)
        if normlized:
            plt.plot(self.normalized_data['Adj Close'].values, color='red', label='Adj Close')
            plt.ylabel('price normalized')
        else:
            plt.plot(self.data['Adj Close'].values, color='red', label='Adj Close')
            plt.ylabel('price')
        plt.title(self.name + ' stock price')
        plt.xlabel('time')
        plt.legend(loc='best')
        return plt

    def normalize(self):
        # drops volume, dividends & stock splits info
        normalized_data = self.data.copy()
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_data['Adj Close'] = min_max_scaler.fit_transform(self.data['Adj Close'].values.reshape(-1, 1))
        print('=======================')
        print('Normalized data info: ')
        print("data min: ", min_max_scaler.data_min_)
        print("data max: ", min_max_scaler.data_max_)
        print("data range: ", min_max_scaler.data_range_)
        normalized_data.info()
        return normalized_data.values

    def create_sequence(self, window_length=None, stride=1, normalize=True):
        all_sequences = []
        if normalize:
            for i in range((len(self.data) - window_length) // stride + 1):
                all_sequences.append(self.normalized_data[i * stride: i * stride + window_length])
        else:
            for i in range((len(self.data) - window_length) // stride + 1):
                all_sequences.append(self.data.values[i * stride: i * stride + window_length])

        print('Sequence shape is: ', np.array(all_sequences).shape)
        return np.array(all_sequences)
