import numpy as np
import gym
from gym import spaces


class CircleOfDeath(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        """
        HighwayETC Environment
        :param config: see config.py
        """
        super(CircleOfDeath, self).__init__()

        # action space
        # ["no_change", "speed_up", "speed_up_up", "slow_down", "slow_down_down"]
        self.actions_list = ["stay", "right", "left", "up", "down", "right_up", "right_down", "left_up", "left_down"]
        #self.action2delta = {0: 0., 1: 1., 2: 2., 3: -1., 4: -2.}
        self.action_space = spaces.Discrete(len(self.actions_list))

        # state space: [position, velocity]
        self.state = dict()
        self.circle_size = (6,6)
        # self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            # high=np.array([self.max_position ]),
                                            # shape=(6,6),
                                            # dtype=np.float32)

        # rewards_dict
        self.rewards_dict = {
            "crash": -50,
            "restricted_zone": -30,
            "wrong_direction": -10,
            "missed_exit": -5,
            "reg_no_crash": 10,
            "success": 300,
            "delay_penalty": -10,
            "out_of_bounds": -100
        }

        # median + sidewalk corners
        self.restricted_zones = [0, 5, 14, 15, 20, 21, 30, 35]

        self.exit_zoneNorth = [1,2,3,4] #4 for each side
        self.exit_zoneWest = [6,12,18,24]
        self.exit_zoneSouth = [31,32,33,34]
        self.exit_zoneEast = [11,17,23,29]
        self.exit_zones = [1,2,3,4,6,12,18,24,11,17,23,29,31,32,33,34]
        self.zone_num_to_dir = {0: self.exit_zoneNorth, 1: self.exit_zoneWest, 2: self.exit_zoneSouth, 3: self.exit_zoneEast}

        self.right_dirs = {
            0: ["down", "right_down"],
            1: ["left_down", "left", "down"],
            2: ["left_down", "left", "down"],
            3: ["left_down", "left", "down"],
            4: ["left_down", "left", "down"],
            5: ["left_down", "left"],
            6: ["right", "down", "right_down"],
            7: ["left", "down", "left_down" ,"up"],
            8: ["left", "up", "left_up", "left_down"],
            9: ["left", "up", "left_up"],
            10: ["left", "up", "left_up", "right"],
            11: ["left", "left_up"],
            12: ["right", "down", "right_down"],
            13: ["left", "down"],
            14: ["left", "left_down"],
            15: ["up", "left_up"],
            16: ["right", "up", "right_up", "left_up"],
            17: ["left", "up", "left_up"],
            18: ["right", "down", "right_down"],
            19: ["left", "down", "right_down"],
            20: ["left", "left_down", "down", "right_down"],
            21: ["right", "right_up"],
            22: ["right", "up", "right_up"],
            23: ["up", "left_up", 'left'],
            24: ["right", "down", "right_down"],
            25: ["left", "down", "right", "right_down"],
            26: ["right", "down", "right_down"],
            27: ["right", "down", "right_down"],
            28: ["right", "down", "up", "right_up"],
            29: ["up", "left_up", "left"],
            30: ["right", "up", "right_up"],
            31: ["right", "up", "right_up"],
            32: ["right", "up", "right_up"],
            33: ["right", "up", "right_up"],
            34: ["right", "up", "right_up"],
            35: ["right", "up"]
        }


        self.circle = np.full(self.circle_size, '*')
        for i in self.restricted_zones:
            block_coord = np.unravel_index(i, self.circle_size)
            self.circle[block_coord[0], block_coord[1]] = 'X'

        self.state = self._new_state()
        self.prev_state = self.state

    def reset(self):
        """
        Return observation as np.array
        :return: observation (np.array)
        """
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
        #     "missed_exit": -30,
        # }

        return reward, done

    def step(self, actionIndex):
        # print(self.state)
        # 0=no_change, 1=speed_up, 2=speed_up_up, 3=slow_down, 4=slow_down_down

        # remember prev state
        self.prev_state = self.state
        action = self.actions_list[actionIndex]
        self.state["action"] = action
        out_of_bounds = False

        cur_loc_coord = np.unravel_index(self.state["cur_loc"], self.circle_size)
        new_loc_coord = None

        #["stay", "right", "left", "up", "down", "right_up", "right_down", "left_up", "left_down"]
        if action=="stay":
            new_loc_coord = cur_loc_coord
        elif action=="right":
            new_loc_coord = (cur_loc_coord[0], cur_loc_coord[1] + 1)
        elif action=="left":
            new_loc_coord = (cur_loc_coord[0], cur_loc_coord[1] - 1)
        elif action=="up":
            new_loc_coord = (cur_loc_coord[0] - 1, cur_loc_coord[1])
        elif action=="down":
            new_loc_coord = (cur_loc_coord[0] + 1, cur_loc_coord[1])
        elif action=="right_up":
            new_loc_coord = (cur_loc_coord[0] - 1, cur_loc_coord[1] + 1)
        elif action=="right_down":
            new_loc_coord = (cur_loc_coord[0] + 1, cur_loc_coord[1] + 1)
        elif action=="left_up":
            new_loc_coord = (cur_loc_coord[0] - 1, cur_loc_coord[1] - 1)
        elif action=="left_down":
            new_loc_coord = (cur_loc_coord[0] + 1, cur_loc_coord[1] - 1)
        elif action=="None":
            return
        else:
            raise KeyError("Unknown Action")

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

# Instantiate the env