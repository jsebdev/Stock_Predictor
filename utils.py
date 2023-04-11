import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yfin
import os

from huggingface_hub import from_pretrained_keras


def get_data(ticker='AAPL', start=None, end=None):
    if end is None:
        end = dt.date.today()
    if start is None:
        start = end - dt.timedelta(days=800)

    yfin.pdr_override()
    data = web.data.get_data_yahoo(ticker, start, end)
    # data = pd.read_csv('train_data.csv', index_col='Date')
    return data


def get_last_candle_value(data, column):
    val = data.iloc[-1][column]
    return "{:.2f}".format(val)


# Preprocessing functions copied from notebook where model was trained
def create_remove_columns(data):
    # create jump column
    data = pd.DataFrame.copy(data)
    data['Jump'] = data['Open'] - data['Close'].shift(1)
    data['Jump'].fillna(0, inplace=True)
    data.insert(0,'Jump', data.pop('Jump'))
    return data

def normalize_data(data):
    # Returns a tuple with the normalized data, the scaler and the decoder
    # The normalized data is a dataframe with the following columns:
    # ['Jump', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    the_data = pd.DataFrame.copy(data)
    # substract the open value to all columns but the first one and the last one which are "Jump" and "Volume"
    the_data.iloc[:, 1:-1] = the_data.iloc[:,1:-1] - the_data['Open'].values[:, np.newaxis]
    # print('the_data')
    # print(the_data)

    the_data.pop('Open')
    # Create the scaler
    max_value = float(os.getenv('SCALER_MAX_VALUE'))
    max_volume = float(os.getenv('SCALER_MAX_VOLUME'))
    def scaler(d):
        data = pd.DataFrame.copy(d)
        print('max_value: ', max_value)
        print('max_volume: ', max_volume)
        data.iloc[:, :-1] = data.iloc[:,:-1].apply(lambda x: x/max_value)
        data.iloc[:, -1] = data.iloc[:,-1].apply(lambda x: x/max_volume)
        return data
    def decoder(values):
        decoded_values = values * max_value
        return decoded_values
    
    normalized_data = scaler(the_data)

    return normalized_data, scaler, decoder

def preprocessing(data):
    # print(data.head(3))
    data_0 = create_remove_columns(data)
    # print(data_0.head(3))
    #todo: save the_scaler somehow to use in new runtimes
    norm_data, scaler, decoder = normalize_data(data_0)
    # print(norm_data.head(3))
    # print(x_train.shape, y_train.shape)
    norm_data_array = np.array(norm_data)
    return np.expand_dims(norm_data_array, axis=0), decoder


# Model prediction
model = from_pretrained_keras("jsebdev/apple_stock_predictor")
def predict(data):
    input, decoder = preprocessing(data)
    print("input")
    print(input.shape)
    result = decoder(model.predict(input))
    last_close = data.iloc[-1]['Close']
    next_candle = result[0, -1]
    print('next_candle')
    print(next_candle)
    jump = next_candle[0]
    next_candle = next_candle + last_close
    return (jump, next_candle[0], next_candle[1], next_candle[2], next_candle[3])

def predict_mock(data):
    return (0,1,2,3,4)