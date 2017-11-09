import tensorflow as tf
import numpy as np
import gym
import importlib
import random

import os

#import roboschool

from pipaek.agent import *
from pipaek.config import *


verbose = 2

ALGORITHMS={'behavior_cloning', 'DAgger'}

IMITATION_ITERATION = 200


class ImitationSchool:
    def __init__(self, game, module_name, algorithm):
        self.env = game.env
        self.env_name = game.env_name
        self.module_name = module_name
        self.teacher = Teacher(self.env, module_name)
        self.algorithm = algorithm

        if self.algorithm not in ALGORITHMS:
            raise ValueError("Unknown Algorithm %s"%self.algorithm)

    def imitation_learning(self, student, policy_to_train):
        max_steps = self.env.spec.timestep_limit
        debug(log_level_info, 'imitation_learning algorithm:%s START!!' % self.algorithm)

        if self.algorithm == 'behavior_cloning':
            policy_to_act = self.teacher.policy
            student.set_policy(policy_to_train, policy_to_act)
            self.behavior_cloning(self.env, student, self.teacher, max_steps)
        elif self.algorithm == 'DAgger':
            policy_to_act = DaggerActionPolicy(self.env, self.teacher, student)
            student.set_policy(policy_to_train, policy_to_act)
            self.dagger(self.env, student, self.teacher, max_steps)

        debug(log_level_info, 'imitation_learning algorithm:%s FINISH!!' % self.algorithm)

        filepath = self.weight_file_path(policy_to_train.model)
        policy_to_train.save_weights(filepath)

        debug(log_level_info, 'Weight File:%s Saved!!' % filepath)

    # Trains the student network using Behavior Cloning.
    def behavior_cloning(self, env, student, teacher, max_steps):
        policy_to_act = teacher.policy
        policy_to_train = student.policy_to_train

        debug(log_level_info, 'behavior_cloning generate_rollouts START')
        #train_data = self.generate_rollouts(env, policy_to_act, max_steps, 100, False, 0)
        train_data = self.generate_rollouts(env, policy_to_act, max_steps, 3, False, 0)
        debug(log_level_info, 'behavior_cloning generate_rollouts OK')
        val_data = self.make_validation(env, teacher.policy, max_steps)
        debug(log_level_info, 'behavior_cloning make_validation OK')

        filepath = self.weight_file_path(policy_to_train.model)

        for i in range(IMITATION_ITERATION+1):
            debug(log_level_info, 'Iteration : %d' % i)
            if i == 0:
                #epochs = 100
                epochs = 10
            else:
                epochs = 4
            policy_to_train.train(train_data, val_data, epochs, verbose)

            self.generate_rollouts(env, policy_to_train, max_steps, 1, False, verbose)
            policy_to_train.save_weights(filepath)
            debug(log_level_info, 'Weight File:%s Saved!!' % filepath)


    # Trains the student network using DAgger.
    def dagger(self, env, student, teacher, max_steps):
        policy_to_act = student.policy_to_act   # DaggerActionPolicy!!
        policy_to_train = student.policy_to_train

        debug(log_level_info, 'DAgger START')
        val_data = self.make_validation(env, teacher.policy, max_steps)
        debug(log_level_info, 'DAgger make_validation OK')

        filepath = self.weight_file_path(policy_to_train.model)

        for i in range(IMITATION_ITERATION+1):
            debug(log_level_info, 'Iteration : %d' % i)
            if i == 0:   # same as behavior_cloning
                #rollouts = 100
                #epochs = 100
                rollouts = 3
                epochs = 3
                train_data = self.generate_rollouts(env, teacher.policy, max_steps, rollouts, False, 0)
                policy_to_train.train(train_data, val_data, epochs, verbose)
            else:
                rollouts = 1
                epochs = 4
                #policy_to_act.teacher_act_ratio -= 1/IMITATION_ITERATION
                debug(log_level_info, 'DAgger Iter %d, teacher_act_ratio = %f' % (i, policy_to_act.teacher_act_ratio))
                policy_to_act.init_rollout_data(i-1)
                self.generate_rollouts(env, policy_to_act, max_steps, rollouts, False, 0)
                policy_to_train.train(policy_to_act.teacher_data(), val_data, epochs, verbose)

            self.generate_rollouts(env, policy_to_train, max_steps, 1, False, verbose)
            policy_to_train.save_weights(filepath)
            debug(log_level_info, 'Weight File:%s Saved!!' % filepath)


    def student_perform(self, student, policy, num_rollouts):
        if self.algorithm == 'behavior_cloning':
            student.set_policy(policy, None)
        elif self.algorithm == 'DAgger':
            student.set_policy(policy, None)

        filepath = self.weight_file_path(policy.model)
        policy.load_weights(filepath)

        debug(log_level_info, 'Weight File:%s Loaded!!' % filepath)

        max_steps = self.env.spec.timestep_limit
        self.generate_rollouts(self.env, policy, max_steps, num_rollouts, render=True, verbose=0)


    def teacher_demonstration(self, num_rollouts):
        max_steps = self.env.spec.timestep_limit
        self.generate_rollouts(self.env, self.teacher.policy, max_steps, num_rollouts, render=True, verbose=0)

    ''' pipaek : generate_rollouts only for Dense..
    # Generates rollouts of the policy on the environment, prints the mean & std of
    # the rewards, and returns the observations and actions.
    def generate_rollouts(self, env, policy, max_steps, num_rollouts, render, verbose):
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy.act(obs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0 and verbose >= 2:
                    debug(log_level_trace, "%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            if verbose >= 1:
                debug(log_level_info, 'rollout %i/%i return=%f' % (i + 1, num_rollouts, totalr))
            returns.append(totalr)

        debug(log_level_info, 'Return summary: mean=%f, std=%f' % (np.mean(returns), np.std(returns)))

        return (np.array(observations), np.array(actions))'''

    # pipaek : To adapt for both Recurrence and Dense...
    # Generates rollouts of the policy on the environment, prints the mean & std of
    # the rewards, and returns the observations and actions.
    def generate_rollouts(self, env, policy, max_steps, num_rollouts, render, verbose):
        returns = []
        rollouts = []
        for i in range(num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            observations = []
            actions = []
            while not done:
                action = policy.act(obs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0 and verbose >= 2:
                    debug(log_level_trace, "%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            rollouts.append((observations, actions))
            if verbose >= 1:
                debug(log_level_info, 'rollout %i/%i return=%f' % (i + 1, num_rollouts, totalr))
            returns.append(totalr)

        debug(log_level_info, 'Return summary: mean=%f, std=%f' % (np.mean(returns), np.std(returns)))

        return rollouts

    # Make a small but low-variance validation test by subsampling across many episodes.
    ''' pipaek : make_validation only for Dense..
    def make_validation(self, env, policy, max_steps):
        val_data = self.generate_rollouts(env, policy, max_steps, 50, False, 0)
        val_data = (val_data[0][::10], val_data[1][::10])
        return val_data'''

    # pipaek : To adapt for both Recurrence and Dense...
    #rollouts.append((observations, actions))
    def make_validation(self, env, policy, max_steps):
        val_data = self.generate_rollouts(env, policy, max_steps, 50, False, 0)
        #val_data = (val_data[0][::10], val_data[1][::10])
        return val_data[::10]

    def weight_file_path(self, model):
        dir = work_dir + '/games/' + self.env_name
        path = dir + '/weight_' + self.algorithm + '_' + model.getModelName()
        if not tf.gfile.IsDirectory(dir):
            tf.gfile.MakeDirs(dir)
        return path