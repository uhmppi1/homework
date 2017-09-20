import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from pipaek.util import *




class Model:
    def __init__(self):
        pass #self.net = None

    def train(self, train_data, val_data, epochs, verbose):
        self.net.fit(train_data[0], train_data[1],
                       batch_size=128,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=val_data)

    def save(self, filename):
        self.net.save_weights(filename) #overwrite=True

    def load(self, filename):
        self.net.load_weights(filename)

    #def net(self):
    #    return self.net
    def predict(self, obs):
        obs_batch = np.expand_dims(obs, 0)
        act_batch = self.net.predict_on_batch(obs_batch)
        return np.ndarray.flatten(act_batch)


    def getModelName(self):
        #return type(self).__class__.__name__
        return type(self).__name__



class DenseModel(Model):
    def __init__(self, env, layer_units=(128, 128, 64, 64), activation='relu', loss='mse', optimizer='sgd'):
        input_len, output_len = env_dims(env)

        self.net = Sequential()

        for idx in range(len(layer_units)):
            if idx==0:
                self.net.add(Dense(units=layer_units[idx], input_dim=input_len, activation=activation))
            else:
                self.net.add(Dense(units=layer_units[idx], activation=activation))
        self.net.add(Dense(units=output_len))

        self.net.compile(loss=loss, optimizer=optimizer)