import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.callbacks import TensorBoard
from pipaek.util import *


class Model:
    def __init__(self):
        pass

    def train(self, train_data, val_data, epochs, verbose):
        pass

    def save(self, filename):
        self.net.save_weights(filename) #overwrite=True

    def load(self, filename):
        self.net.load_weights(filename)

    def predict(self, obs):
        obs_batch = np.expand_dims(obs, 0)
        act_batch = self.net.predict_on_batch(obs_batch)
        return np.ndarray.flatten(act_batch)


    def getModelName(self):
        return type(self).__name__


#standard Dense
class DenseModel(Model):
    def __init__(self, env, layer_units=(128, 128, 64, 64), activation='relu', loss='mse', optimizer='sgd', dropout=0.2):
        input_len, output_len = env_dims(env)

        self.net = Sequential()

        for idx in range(len(layer_units)):
            if idx==0:
                self.net.add(Dense(units=layer_units[idx], input_dim=input_len, activation=activation))
            else:
                self.net.add(Dense(units=layer_units[idx], activation=activation))
            if dropout > 0.:
                self.net.add(Dropout(dropout))
        self.net.add(Dense(units=output_len))

        #optimizer.__setattr__('lr', 0.001)

        self.net.compile(loss=loss, optimizer=optimizer)
        #self.net.optimizer.__setattr__('lr', 0.001)

    def train(self, train_data, val_data, epochs, verbose):
        _train_data = self.rollout_to_train_data(train_data)
        _val_data = self.rollout_to_train_data(val_data)

        self.net.fit(_train_data[0], _train_data[1],
                       batch_size=128,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=_val_data)

    def rollout_to_train_data(self, rollout):

        rollout_arr = np.array(rollout)
        print(rollout_arr.shape)

        unzip_list = list(zip(*rollout))
        unzip_list_arr = np.array(unzip_list)
        print(unzip_list_arr.shape)

        return (np.array(list(np.concatenate(unzip_list_arr[0]))), np.array(list(np.concatenate(unzip_list_arr[1]))))


#same, but bigger than DenseModel
class DenseModelBigger(DenseModel):
    def __init__(self, env, layer_units=(256, 256, 128, 128), activation='relu', loss='mse', optimizer='sgd', dropout=0.2):
        super(DenseModelBigger, self).__init__(env, layer_units=layer_units, activation=activation, loss=loss, optimizer=optimizer, dropout=dropout)

# same, but smaller than DenseModel
class DenseModelSmaller(DenseModel):
    def __init__(self, env, layer_units=(64, 64), activation='relu', loss='mse', optimizer='sgd', dropout=0.2):
        super(DenseModelSmaller, self).__init__(env, layer_units=layer_units, activation=activation, loss=loss,
                                               optimizer=optimizer, dropout=dropout)

# same, but much smaller than DenseModel
class DenseModelTiny(DenseModel):
    def __init__(self, env, layer_units=(64,), activation='relu', loss='mse', optimizer='sgd', dropout=0):
        super(DenseModelTiny, self).__init__(env, layer_units=layer_units, activation=activation, loss=loss,
                                                optimizer=optimizer, dropout=dropout)


#same, but bigger than DenseModel
class DenseModelNoDropout(DenseModel):
    def __init__(self, env, layer_units=(128, 128, 64, 64), activation='relu', loss='mse', optimizer='sgd', dropout=0.0):
        super(DenseModelNoDropout, self).__init__(env, layer_units=layer_units, activation=activation, loss=loss, optimizer=optimizer, dropout=dropout)



class RecurrentModel(Model):
    def __init__(self, env, hidden_size=64, activation='tanh', loss='mse', optimizer='adam', dropout=0.2):
        input_len, output_len = env_dims(env)
        timesteps = 5  #hyperparam

        self.net = Sequential()
        #self.net.add(LSTM(units=hidden_size, activation=activation, batch_input_shape=(128, input_len, ), return_sequences=False))
        self.net.add(LSTM(units=hidden_size, input_shape=(None, input_len), return_sequences=False))
        #self.net.add(LSTM(units=hidden_size, return_sequences=False))
        #self.net.add(LSTM(input_len, hidden_size, return_sequences=False))
        if dropout > 0.:
            self.net.add(Dropout(dropout)),
        self.net.add(Dense(output_len, input_dim=hidden_size))
        self.net.add(Activation("linear"))
        self.net.compile(loss=loss, optimizer=optimizer)

        #self.net.summary()

    def train(self, train_data, val_data, epochs, verbose):
        self.net.fit(train_data[0], train_data[1],
                     batch_size=128,
                     epochs=epochs,
                     verbose=verbose,
                     validation_data=val_data)

        '''model.summary()

        for idx in range(len(layer_units)):
            if idx==0:
                self.net.add(LSTM(units=layer_units[idx], input_shape=(timesteps, input_len), return_sequences=True))
            #elif idx==len(layer_units)-1:
            #    self.net.add(LSTM(units=layer_units[idx], return_sequences=False))
            else:
                self.net.add(LSTM(units=layer_units[idx], return_sequences=True))


        self.net.add(LSTM(5, input_shape=(timesteps, input_len), return_sequences=True))
        self.net.add(Dropout(0.2))
        self.net.add(LSTM(10, return_sequences=False))
        self.net.add(Dense(1))
        self.net.add(Activation('linear'))'''



