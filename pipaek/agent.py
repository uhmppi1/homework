import tensorflow as tf
import numpy as np
import gym
import importlib
import random

import os
import abc

from pipaek.policy import *
from pipaek.util import *
#import roboschool

class Agent:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractclassmethod
    def act(self, obs):
        pass


class OffPolicyAgent(Agent):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractclassmethod
    def policy_to_act(self):
        pass

    @abc.abstractclassmethod
    def policy_to_train(self):
        pass

    @abc.abstractclassmethod
    def act(self, obs):
        pass


class Teacher(Agent):
    def __init__(self, env, module_name):

        debug(log_level_info, 'Loading Teacher Policy module_name : %s' % module_name)

        if module_name.endswith('.py'):
            module_name = module_name[:-3]

        policy_module = importlib.import_module(module_name)

        debug(log_level_info, 'Teacher Policy loaded OK')

        self.cls = getattr(policy_module, 'ZooPolicyTensorflow')
        if self.cls is None:
            self.cls = getattr(policy_module, 'SmallReactivePolicy')

        assert self.cls is not None

        self.env = env
        self.instance = self.cls("Teacher", self.env.observation_space, self.env.action_space)
        self.policy = TeacherActionPolicy(self)

    def act(self, obs):
        return self.instance.act(obs, None)



class Student(OffPolicyAgent):
    def __init__(self):
        self.policy_to_train = None
        self.policy_to_act = None

    def set_policy(self, policy_to_train, policy_to_act):
        self.policy_to_train = policy_to_train
        self.policy_to_act = policy_to_act

    def policy_to_act(self):
        return self.policy_to_act

    def policy_to_train(self):
        return self.policy_to_train

    def act(self, obs):
        return self.policy_to_act.act(obs)

    def myact(self, obs):
        return self.policy_to_train.act(obs)
