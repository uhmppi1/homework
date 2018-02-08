import sys

sys.path.append("../pipaek")

#from pipaek.util import *
from util import *


testSqlite()

#conditions = { 'ValueType':['mean reward'], 'game':['PongNoFrameskip-v3'] }
#conditions = { 'ValueType':['mean reward'], 'game':['BreakoutNoFrameskip-v3'] }
#conditions = { 'ValueType':['mean reward'], 'Model':['baseline'] }

#conditions = { 'ValueType':['mean reward'], 'game':['BeamRiderNoFrameskip-v3', 'BeamRider-ram-v0'], 'Model':['baseline'] }
#conditions = { 'ValueType':['mean reward'],#conditions = { 'ValueType':['mean reward'], 'game':['BeamRiderNoFrameskip-v3', 'BeamRider-ram-v0'] }
#               'game':['BeamRiderNoFrameskip-v3', 'BeamRider-ram-v0'], 'Model':['baseline', 'exp_2m'] }
#conditions = { 'ValueType':['mean reward'], 'game':['BeamRiderNoFrameskip-v3', 'BeamRider-ram-v0'] }
conditions = { 'ValueType':['mean reward'],#conditions = { 'ValueType':['mean reward'], 'game':['BeamRiderNoFrameskip-v3'] }
               'game':['BeamRiderNoFrameskip-v3'], 'Model':['baseline', 'rep_2m', 'exp_2m'] }
selectdata = select_log_data(conditions)
#print(selectdata)

data_x, data_y = get_plot_data(selectdata)
#print(data_x)
#print(data_y)

draw_plot(selectdata, x_units=10000)