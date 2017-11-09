import sys

sys.path.append("../pipaek")

#from pipaek.util import *
from util import *


testSqlite()

conditions = { 'ValueType':['mean reward'] }
#conditions = { 'ValueType':['mean reward'], 'Model':['baseline'] }
#conditions = { 'ValueType':['mean reward'], 'game':['BeamRiderNoFrameskip-v3', 'BeamRider-ram-v0'] }
selectdata = select_log_data(conditions)
#print(selectdata)

data_x, data_y = get_plot_data(selectdata)
#print(data_x)
#print(data_y)

draw_plot(selectdata, x_units=10000)