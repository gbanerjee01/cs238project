import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(raw_rewardsperep, save_path):
    plt.grid(linestyle='-.')
    returns_smoothed = pd.Series(raw_rewardsperep).rolling(10, min_periods=10).mean()
    plt.plot(raw_rewardsperep, linewidth=0.5, label='reward per episode')
    plt.plot(returns_smoothed, linewidth=2.0, label='smoothed reward (over window size=10)')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward v.s. Training Episode")
    plt.savefig(save_path)

def get_right_dirs():
    right_dirs = {
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

    return right_dirs

def execute_action(cur_loc_coord, action):
    new_loc_coord = None
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
    else:
        raise KeyError("Unknown Action")

    return new_loc_coord


def uncertainty(adv_locs):
    all_locs = np.arange(36)

    difference = set(all_locs).symmetric_difference(set(adv_locs))
    no_advs = list(difference)

    ret = []

    for i in no_advs: #false positives explicitly being added
        if np.random.normal(0.05, 0.01) > 0.8:
            ret.append[i]

    for i in adv_locs: #false negative introduction by implicit ignoring
        if np.random.normal(0.85, 0.5) > 0.9:
            ret.append[i]

    return ret