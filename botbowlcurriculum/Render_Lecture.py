#!/usr/bin/env python3

# import gym
import numpy as np
import ffai
from ffai.ai.env_wrappers import GotebotWrapper
from pdb import set_trace

import gym

from botbowlcurriculum import all_lectures

def main():
    env = GotebotWrapper(gym.make("FFAI-v3"), all_lectures)
    env.lecture = [lect for lect in all_lectures if lect.name == "Pickup"][0]
    reset = True
    action = None

    while True:
        if reset:
            reset = False
            spatial_obs, non_spatial_obs, action_mask = env.gen_observation(env.env_reset_with_lecture())
            print(f"reset to '{env.lecture.name}' lvl {env.lecture.get_level()}")
        else:
            _, done, _, _, action_mask = env.step(action)
            if done:
                reset = True
                continue

        action = None
        env.env.render()

        c = input()
        if c == "n":  # next level
            env.lecture.increase_level()
            print(f"level: {env.lecture.get_level()}")
            reset = True
        elif c == "r":  # reset at same diff
            reset = True

        elif c == "s" or c == "":  # take a random step
            action = np.random.choice(action_mask.nonzero()[0])

        elif c == "t":  # set trace
            set_trace()
        elif c == "a":  # print actions
            actions = env.game.get_available_actions()
            for a in actions:
                print(a.action_type)
        elif c == "exit" or c == "e" or c == "x":
            break


if __name__ == "__main__":
    main()