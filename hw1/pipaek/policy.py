import numpy as np
import abc
import random
from pipaek.util import *


class Policy:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractclassmethod
    def act(self, obs):
        pass


class TrainablePolicy(Policy):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model):
        pass

    @abc.abstractclassmethod
    def act(self, obs):
        pass

    @abc.abstractclassmethod
    def train(self, train_data):
        pass

    @abc.abstractclassmethod
    def save_weights(self, filepath):
        pass

    @abc.abstractclassmethod
    def load_weights(self, filepath):
        pass


# A neural network that learns a mapping from observations to actions.
class ImmitationPolicy(TrainablePolicy):
    def __init__(self, model):
        self.model = model

    def act(self, obs):
        return self.model.predict(obs)

    def train(self, train_data, val_data, epochs, verbose):
        self.model.train(train_data, val_data, epochs, verbose)

    def save_weights(self, filepath):
        self.model.save(filepath)

    def load_weights(self, filepath):
        self.model.load(filepath)


class TeacherActionPolicy(Policy):
    def __init__(self, teacher):
        self.teacher = teacher

    def act(self, obs):
        return self.teacher.act(obs)


# A policy that may use the student's action but, with probability
# fraction_assist, uses the teacher's action instead. In either case, it
# remembers the teacher's action, which can be used for supervised learning.
class DaggerActionPolicy(Policy):
    def __init__(self, env, teacher, student):
        self.CAPACITY = 100000
        self.teacher = teacher
        self.student = student
        self.teacher_act_ratio = 1.
        self.next_idx = 0
        self.size = 0

        input_len, output_len = env_dims(env)
        self.obs_data = np.empty([self.CAPACITY, input_len])
        self.act_data = np.empty([self.CAPACITY, output_len])

    def act(self, obs):
        teacher_act = self.teacher.policy.act(obs)
        self.obs_data[self.next_idx] = obs
        self.act_data[self.next_idx] = teacher_act
        self.next_idx = (self.next_idx + 1) % self.CAPACITY
        self.size = min(self.size + 1, self.CAPACITY)

        if random.random() < self.teacher_act_ratio:
            return teacher_act
        else:
            return self.student.myact(obs)

    def teacher_data(self):
        return (self.obs_data[:self.size], self.act_data[:self.size])