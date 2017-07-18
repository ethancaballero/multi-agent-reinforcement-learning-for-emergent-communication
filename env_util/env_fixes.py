
# certain environments require agents to jump through hoops to start
def init_actions(env_name):
    action_init = None
    if env_name=='flashgames.RetroRunner-v0':
        action_init = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,2,0,0,0,0,1,2]
    return action_init

def state_stuck(env_name, state):
    if env_name=='flashgames.RetroRunner-v0':
        #if state[0][0][-1] > 0.00000000:
        if state[1][0][0][-1] > 0.00000000:
            return True
        else:
            return False
    else:
        return False 
