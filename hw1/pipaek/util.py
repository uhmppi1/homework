import time

log_level_error = 1
log_level_warning = 2
log_level_info = 3
log_level_debug = 4
log_level_trace = 5

log_level = 5

def debug(level, logstr):
    if(log_level >= level):
        print("[" + getCurrentTime() + "] "+str(logstr))

def getCurrentTime():
    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (
    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

# Extracts the number of input and output units from an OpenAI Gym environment.
def env_dims(env):
    return (env.observation_space.shape[0], env.action_space.shape[0])