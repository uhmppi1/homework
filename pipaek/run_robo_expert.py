import tensorflow as tf
import numpy as np
import gym
import importlib
import roboschool     #important for making roboschool envs.

from pipaek.config import *
from pipaek.school import *
from pipaek.model import *


class Game:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)

def run_imitation_learning(game, module_name, algorithm, model, mode, num_rollouts=5):

    with tf.Session() as sess:

        school = ImitationSchool(game, module_name, algorithm)
        student = Student()

        if mode == 'train':
            policy_to_train = ImmitationPolicy(model)
            school.imitation_learning(student, policy_to_train)
        elif mode == 'perform':
            policy_to_train = ImmitationPolicy(model)
            school.student_perform(student, policy_to_train, num_rollouts)
        elif mode == 'demo':
            school.teacher_demonstration(num_rollouts)



#game = Game('RoboschoolHopper-v1')
game = Game('RoboschoolAnt-v1')
model=DenseModelTiny(game.env)
#model=RecurrentModel(game.env)
#model=DenseModelBigger(game.env)
#model=RecurrentModel(game.env)
#run_imitation_learning('RoboschoolHopper-v1', 'RoboschoolHopper_v1_2017jul', 'behavior_cloning', None, mode='demo', num_rollouts=5)
#run_imitation_learning(game, 'RoboschoolHopper_v1_2017jul', 'behavior_cloning', model=None, mode='demo', num_rollouts=5)
#run_imitation_learning(game.env_name, 'RoboschoolHopper_v1_2017jul', 'behavior_cloning', model=SequentialModel(game.env), mode='train', num_rollouts=5)

#run_imitation_learning(game, 'RoboschoolHopper_v1_2017jul', 'behavior_cloning', model=SequentialModel(game.env), mode='train')
#run_imitation_learning(game, 'RoboschoolHopper_v1_2017jul', 'behavior_cloning', model=model, mode='train')
#run_imitation_learning(game, 'RoboschoolHopper_v1_2017jul', 'behavior_cloning', model=model, mode='perform', num_rollouts=5)

#run_imitation_learning(game, 'RoboschoolAnt_v1_2017jul', 'DAgger', model=model, mode='train')
run_imitation_learning(game, 'RoboschoolAnt_v1_2017jul', 'DAgger', model=model, mode='demo')
#run_imitation_learning(game, 'RoboschoolHopper_v1_2017jul', 'behavior_cloning', model=model, mode='train')
#run_imitation_learning(game, 'RoboschoolHopper_v1_2017jul', 'behavior_cloning', model=model, mode='perform', num_rollouts=20)


'''game = Game('RoboschoolAnt-v1')
model=DenseModel(game.env)
run_imitation_learning(game, 'RoboschoolAnt_v1_2017jul', 'behavior_cloning', model=model, mode='train')
run_imitation_learning(game, 'RoboschoolAnt_v1_2017jul', 'DAgger', model=model, mode='train')
model=DenseModelBigger(game.env)
run_imitation_learning(game, 'RoboschoolAnt_v1_2017jul', 'behavior_cloning', model=model, mode='train')
run_imitation_learning(game, 'RoboschoolAnt_v1_2017jul', 'DAgger', model=model, mode='train')

#run_imitation_learning(game, 'RoboschoolAnt_v1_2017jul', 'behavior_cloning', model=model, mode='perform')
#run_imitation_learning(game, 'RoboschoolAnt_v1_2017jul', 'DAgger', model=model, mode='train')
run_imitation_learning(game, 'RoboschoolAnt_v1_2017jul', 'DAgger', model=model, mode='perform', num_rollouts=5)'''




'''roll1 = (np.array([1, 2, 3]), np.array(['a', 'b', 'c']))
roll2 = (np.array([4, 5]), np.array(['d', 'e']))
roll3 = (np.array([6, 7, 8, 9]), np.array(['f', 'g', 'h', 'i']))

roll_arr = np.array([roll1, roll2, roll3])

print (roll_arr)

roll_list = roll_arr.tolist()

print(roll_list)'''





#roll_arr = np.array([roll1, roll2, roll3])

#print (roll_arr)

#roll_list = roll_arr.tolist()

#print(roll_list)











        #if algorithm and save_weight_file:
        #    student.save(save_weight_file)
        #print('Final Rollouts')
        #generate_rollouts(env, student, max_steps, num_rollouts, render, verbose)


'''def load_file_path(env_name, algorithm):
    dir = 'games/' + env_name + '/' + algorithm
    if not tf.gfile.IsDirectory(dir):
        tf.gfile.MakeDirs(dir)

    print('In load_file_path')
    #LOAD_FILES = "data/"+env_name+"/"+algorithm+"/weight-?????"
    LOAD_FILES = dir+"/weight_?????"

    match = tf.gfile.Glob(LOAD_FILES)
    if not match:
        print('not match')
        return -1, None
        #raise ValueError("Found no files matching %s" % pattern)
    #input_files.extend(match)

    print('matched!!')
    print(match)

    #sorted(match, key=os.path.getctime, reverse=True)
    match.sort()
    print('load file : %s' % match[-1])
    print('load file idx : %d' % int(match[-1][-5:]))
    #print('load file idx : %d' % int(match[-1][-5]))

    return int(match[-1][-5:]), match[-1]

    #count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))



def save_file_path(env_name, algorithm, cur_idx):
    #dir = 'data/'+ env_name
    #os.makedirs(os.path.dirname(dir), exist_ok=True)
    #if not tf.gfile.IsDirectory(dir):
    #    tf.gfile.MakeDirs(dir)
    dir = 'games/' + env_name+ '/' + algorithm
    if not tf.gfile.IsDirectory(dir):
        tf.gfile.MakeDirs(dir)
    return dir + '/weight_%05d'%(cur_idx+1)'''




'''run_imitation_learning(env_name='RoboschoolHalfCheetah-v1', module_name='RoboschoolHalfCheetah_v1_2017jul',
    algorithm = 'dagger', render = True,  # max_timesteps=10,     # pipaek max_timesteps:10  temporal
    load_weight_file = None, save_weight_file = None)'''


'''run_imitation_learning(env_name='RoboschoolHumanoid-v1', module_name='RoboschoolHumanoid_v1_2017jul',
    algorithm = 'dagger', render = True,  # max_timesteps=10,     # pipaek max_timesteps:10  temporal
    load_weight_file = None, save_weight_file = None)'''

#run_imitation_learning(env_name='RoboschoolAnt-v1', module_name='RoboschoolAnt_v1_2017jul',
#    algorithm = ['behavior_cloning','dagger'], render = True,  # max_timesteps=10,     # pipaek max_timesteps:10  temporal
#    )

#print (os.environ['ROBOSCHOOL_PATH'])
#print (os.environ)
#import sys
#print(sys.getprofile())
#ret = os.system('echo getenv("ROBOSCHOOL_PATH");')
#print(ret)

#import subprocess
#cmd = 'echo $ROBOSCHOOL_PATH'
#subprocess.call(cmd)

#ret = os.popen('echo ${ROBOSCHOOL_PATH}').read()
#ret = os.popen('echo 123').read()
#print(ret)
#print(os.environ['HOME'])
#from envparse import env as penv

#aaa=penv.list("ROBOSCHOOL_PATH")


#print('OK')


#run_imitation_learning(env_name='RoboschoolHalfCheetah_v1', module_name='RoboschoolHalfCheetah_v1_2017jul',
#    algorithm = 'behavior_cloning', render = True,  # max_timesteps=10,     # pipaek max_timesteps:10  temporal
#    load_weight_file = None, save_weight_file = None)

#for _ in range(10):
#    print (random.random())
