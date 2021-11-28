import numpy as np
import gym
from gym import spaces
import utils
import hyperparameterss


class CircleOfDeath(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(CircleOfDeath, self).__init__()

        # action space
        self.actions_list = ["stay", "right", "left", "up", "down", "right_up", "right_down", "left_up", "left_down"]
        self.action_space = spaces.Discrete(len(self.actions_list))

        # state space: [position, velocity]
        self.state = dict()
        self.circle_size = (6,6)

        # rewards_dict
        self.rewards_dict = {
            "crash": -500,
            "restricted_zone": -50,
            "wrong_direction": -20,
            "missed_exit": -5,
            "reg_no_crash": 10,
            "success": 300,
            "delay_penalty": -10,
            "out_of_bounds": -1000
        }

        # median + sidewalk corners
        self.restricted_zones = [0, 5, 14, 15, 20, 21, 30, 35]

        self.exit_zoneNorth = [1,2,3,4] #4 for each side
        self.exit_zoneWest = [6,12,18,24]
        self.exit_zoneSouth = [31,32,33,34]
        self.exit_zoneEast = [11,17,23,29]
        self.exit_zones = [1,2,3,4,6,12,18,24,11,17,23,29,31,32,33,34]
        self.zone_num_to_dir = {0: self.exit_zoneNorth, 1: self.exit_zoneWest, 2: self.exit_zoneSouth, 3: self.exit_zoneEast}

        self.right_dirs = utils.get_right_dirs()

        self.circle = np.full(self.circle_size, '*')
        for i in self.restricted_zones:
            block_coord = np.unravel_index(i, self.circle_size)
            self.circle[block_coord[0], block_coord[1]] = 'X'

        self.state = self._new_state()
        self.prev_state = self.state

    def reset(self):
        self.state = self._new_state()
        return self.state

    def _new_state(self):
        #"sensor_reads"}
        state = dict()
        zone_num = np.random.choice(np.arange(4), size=2, replace=False)
        state["start_loc"] = np.random.choice(self.zone_num_to_dir[zone_num[0]])
        state["exit_goal"] = self.zone_num_to_dir[zone_num[1]]
        state["cur_loc"] = state["start_loc"]
        state["action"] = "None"
        return state 
        # sensor_reads = 

    def _get_reward(self, action, out_of_bounds):
        # return reward, done
        reward = 0
        done = False
        no_faults = True

        if self.state["cur_loc"] in self.restricted_zones:
            reward += self.rewards_dict["restricted_zone"]
            no_faults = False

        if action=="stay" or self.state['cur_loc']==self.prev_state['cur_loc']:
            reward += self.rewards_dict["delay_penalty"]
            no_faults = False

        if self.state["cur_loc"] in self.state["exit_goal"]:
            done=True
            reward += self.rewards_dict["success"]

        if action not in self.right_dirs[self.prev_state['cur_loc']]:
            reward += self.rewards_dict["wrong_direction"]
            no_faults = False

        if out_of_bounds:
            reward += self.rewards_dict["out_of_bounds"]
            no_faults = False

        if no_faults:
            reward += self.rewards_dict["reg_no_crash"]

        # self.rewards_dict = {
        #     "crash": -100,
        #}

        return reward, done

    def step(self, actionIndex):
        self.prev_state = self.state
        action = self.actions_list[actionIndex]
        self.state["action"] = action

        if action=="None":
            return

        out_of_bounds = False

        cur_loc_coord = np.unravel_index(self.state["cur_loc"], self.circle_size)
        new_loc_coord = utils.execute_action(cur_loc_coord, action)

        if new_loc_coord[0]>(self.circle_size[0]-1) or new_loc_coord[1]>(self.circle_size[1]-1) or new_loc_coord[0]<0 or new_loc_coord[1]<0:
            new_loc_coord = cur_loc_coord
            out_of_bounds = True
            # print("Reached boundary of map")

        # print(new_loc_coord)
        new_loc =  np.ravel_multi_index(new_loc_coord, self.circle_size)
        self.state['cur_loc'] = new_loc

        reward, done = self._get_reward(action, out_of_bounds)

        return self.state, reward, done, {}

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        
        cur_loc_coord = np.unravel_index(self.state["cur_loc"], self.circle_size)
        orig_value = self.circle[cur_loc_coord[0], cur_loc_coord[1]]
        self.circle[cur_loc_coord[0], cur_loc_coord[1]] = 'B'
        
        print(self.state)
        print(self.circle)

        self.circle[cur_loc_coord[0], cur_loc_coord[1]] = orig_value

    def close(self):
        print("finished")