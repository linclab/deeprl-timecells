import numpy as np
import random
import copy
from gym import spaces

class IntervalDiscrimination_2D(object):
    """
    Interval discrimination task in 2D touch chamber environement.
    After an initiation signal appears at one corner of the arena, the agent has
    to navigate to the initiation cue and poke it to start stimulus presentation phase.
    Stimulus 1 is presented at the left corner and its duration is chosen randomly from a list of possible durations.
    Stimulus 1 presentation is immediately followed by a delay phase, then stimulus 2
    presentation phase. After stimulus 2 presentation (at the right corner), the agent has to run toward the
    location where the stimulus with a longer delay was presented.
    """
    def __init__(self, len_delay, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, seed=1):
        assert len_edge % 2 == 1
        assert len_edge >= 5
        self.h = (len_edge + 1) // 2 + 2
        self.w = len_edge + 2

        assert rwd >= 0
        assert inc_rwd <= 0
        assert step_rwd <= 0
        assert poke_rwd >= 0

        self.walls = np.ones((self.h, self.w))  # 1 = wall
        i = 1
        while i < len_edge + 2 - i:
            self.walls[i, i:self.w - i] = 0  # 0 = empty arena
            i += 1

        # Six possible actions:
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        # 4: Remain at the same location
        # 5: Poke stimulus
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.h, self.w, 3), dtype=np.uint8)
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1)), np.array((0, 0))]

        # Random number generator
        self.rng = np.random.RandomState(seed)

        # tuple locations for possible signals
        self.initiation_loc = (self.h - 2, (self.w - 1) // 2)
        self.left_loc = (1, 1)
        self.right_loc = (1, self.w - 2)

        # dict - key = loc, value = RGB values
        self.color = {
            "current_loc": [0, 0, 255],  # blue
            "initiation_loc": [255, 0, 0],  # red
            "touchscreen_loc": [0, 255, 0],  # green
            "walls": [255, 255, 255]  # white
        }

        # possible initial cells
        self.init_row = np.where(self.walls == 0)[0]
        self.init_col = np.where(self.walls == 0)[1]

        self.reward = 0
        self.nav_reward = 0  # keep a record of total navigation reward (movement & poking signals)
        self.dist_to_init = None
        self.done = False
        self.len_delay = len_delay
        self.rwd = rwd
        self.inc_rwd = inc_rwd
        self.step_rwd = step_rwd
        self.poke_rwd = poke_rwd
        self.indelay = False
        self.t = 0  # time since delay
        self.current_loc = None
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]
        self.correction_trial = False
        self.phase = None
        self.dist_to_init = None

        self.stimulus_set = [10,15,20,25,30,35,40]
        self.stim_1_len = None
        self.stim_2_len = None
        self.stim_1_loc = None
        self.stim_2_loc = None
        self.correct_loc = None

    def reset(self):
        init_idx = self.rng.choice(np.arange(len(self.init_row)))
        self.current_loc = (self.init_row[init_idx], self.init_col[init_idx])  # random initial location
        self.dist_to_init = abs(self.current_loc[0] - self.initiation_loc[0]) + abs(
            self.current_loc[1] - self.initiation_loc[1])
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]
        self.observation[self.current_loc] += self.color["current_loc"]  # show current location in observation
        self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
        self.indelay = False
        self.t = 0  # time since delay
        self.reward = 0
        self.nav_reward = 0  # keep a record of total navigation reward (movement & poking signals)
        self.done = False
        if not self.correction_trial:   # the length of the stim_1 and stim_2 are kept the same as the previous episode
            self.stim_1_len = np.random.choice(self.stimulus_set)                               # in correction trials
            self.stim_2_len = self.select_second_stim()
        #self.stim_1_loc, self.stim_2_loc = random.sample((self.left_loc, self.right_loc), 2)
        self.stim_1_loc, self.stim_2_loc = [self.left_loc, self.right_loc]
        self.correct_loc = self.find_correct_loc()
        self.phase = "initiation"
        self.indelay = False

    def select_second_stim(self):
        stim_set_copy = copy.deepcopy(self.stimulus_set)
        stim_set_copy.remove(self.stim_1_len)
        return np.random.choice(stim_set_copy)

    def find_correct_loc(self):
        if self.stim_1_len > self.stim_2_len:
            return self.stim_1_loc
        else:
            return self.stim_2_loc

    def dist_init_choice(self):
        left_dist = np.sum(np.abs(np.array(self.left_loc) - np.array(self.initiation_loc)))
        right_dist = np.sum(np.abs(np.array(self.right_loc) - np.array(self.initiation_loc)))
        return (left_dist + right_dist) / 2

    def step(self, action):
        assert self.action_space.contains(action)
        if action in [0, 1, 2, 3, 4]:  # movement
            if self.phase != "delay":
                self.reward = self.step_rwd  # lightly punish all step actions except during delay
                self.nav_reward += self.reward
            next_loc = tuple(self.current_loc + self.directions[action])
            if not self.walls[next_loc]:  # if next location is not walls
                self.observation[self.current_loc] -= self.color["current_loc"]  # leave current location
                self.current_loc = next_loc  # update location
                self.observation[self.current_loc] += self.color["current_loc"]  # show on observation
        else: # poke
            if self.phase == 'initiation' and self.current_loc == self.initiation_loc:
                self.observation[self.current_loc] -= self.color["initiation_loc"]  # turn signal off
                self.reward = self.poke_rwd
                self.nav_reward += self.reward
                self.observation[self.stim_1_loc] += self.color["touchscreen_loc"]  # turn on stim 1 touchscreen
                self.phase = "stim_1"
            elif self.phase == "choice" and (self.current_loc == self.left_loc or self.current_loc == self.right_loc):
                if self.current_loc == self.correct_loc:
                    self.reward = self.rwd
                    self.correction_trial = False
                else:
                    self.reward = self.inc_rwd
                    self.correction_trial = True
                self.done = True
            else:
                if not self.indelay:
                    self.reward = self.step_rwd
                    self.nav_reward += self.reward

        if self.phase == 'stim_1':
            if self.t < self.stim_1_len:
                self.t += 1
            else:
                self.observation[self.stim_1_loc] -= self.color["touchscreen_loc"]  # turn off stim 1 touchscreen
                self.phase = "delay"
                self.indelay = True
                self.t = 0
        elif self.phase == 'delay':
            if self.t < self.len_delay:
                self.t += 1
            else:
                self.observation[self.stim_2_loc] += self.color["touchscreen_loc"]  # turn on initiation signal
                self.phase = "stim_2"
                self.indelay = False
                self.t = 0
        elif self.phase == 'stim_2':
            if self.t < self.stim_2_len:
                self.t += 1
            else:
                self.observation[self.left_loc] += self.color["touchscreen_loc"]
                self.observation[self.right_loc] += self.color["touchscreen_loc"]  # turn on choice signals
                self.phase = "choice"
                self.t = 0

        return self.observation, self.reward, self.done, {}

    def calc_reward_without_stepping(self, action):
        assert self.action_space.contains(action)
        reward = 0
        if action in [0, 1, 2, 3, 4]:  # movement
            if self.phase != "delay":
                reward = self.step_rwd  # lightly punish all step actions except during delay
        else: # poke
            if self.phase == 'initiation' and self.current_loc == self.initiation_loc:
                reward = self.poke_rwd
            elif self.phase == "choice" and (self.current_loc == self.left_loc or self.current_loc == self.right_loc):
                if self.current_loc == self.correct_loc:
                    reward = self.rwd
                else:
                    reward = self.inc_rwd
            else:
                if not self.indelay:
                    reward = self.step_rwd

        return reward




