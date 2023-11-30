import numpy as np
import random
from gym import spaces

'''
Trial-Unique, Delayed Nonmatch-to-Location (TUNL) task in a triangular arena. 
Including the no-memory verion.
'''
class run_to_reward_port(object):
    def __init__(self, len_edge, incentive_probability, incentive_magnitude, poke_reward, step_reward, rng_seed=1234):
        """
        In each episode, the agent starts from a random location and sees either left or right stimulus.
        Agent must poke the stimulus to turn it off and run to the reward port at the back, which will administer a
        reward of incentive_magnitude with incentive_probability.
        :param len_edge: int -- length of long edge in triangle (minimum = 5)
        :param incentive_probability: float -- probability of receiving incentive reward when reaching the reward port
        :param incentive_magnitude: float -- size of incentive reward
        :param poke_reward: float -- size of reward when poke correctly (eg. 1)
        :param step_reward: float -- size of reward for each step (eg. -0.5)
        :param rng_seed: int -- seed for rng that generates initial location (default=1234)
        """
        assert len_edge % 2 == 1
        assert len_edge >= 5
        self.h = (len_edge + 1) // 2 + 2
        self.w = len_edge + 2

        assert incentive_probability >= 0 and incentive_probability <= 1
        assert incentive_magnitude >= 0
        assert poke_reward >= 0
        assert step_reward <= 0

        self.walls = np.ones((self.h, self.w))  # 1 = wall
        i = 1
        while i < len_edge + 2 - i:
            self.walls[i, i:self.w - i] = 0
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
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1)),
                           np.array((0, 0))]

        # Random number generator
        self.rng = np.random.RandomState(rng_seed)

        # tuple locations for possible signals
        self.incentive_loc = (self.h - 2, (self.w - 1) // 2)
        self.left_loc = (1, 1)
        self.right_loc = (1, self.w - 2)

        # dict - key = loc, value = RGB values
        self.color = {
            "current_loc": [0, 0, 255],  # blue
            "touchscreen_loc": [0, 255, 0],  # green
            "walls": [255, 255, 255]  # white
        }

        # possible initial cells
        self.init_row = np.where(self.walls == 0)[0]
        self.init_col = np.where(self.walls == 0)[1]

        self.reward = 0
        self.nav_reward = 0  # keep a record of total navigation reward (movement & poking signals)
        self.dist_to_sample = None
        self.done = False
        self.incentive_probability = incentive_probability
        self.incentive_magnitude = incentive_magnitude
        self.poke_reward = poke_reward
        self.step_reward = step_reward
        self.sample_loc = None
        self.current_loc = None
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]

    def reset(self):
        init_idx = self.rng.choice(np.arange(len(self.init_row)))
        self.current_loc = (self.init_row[init_idx], self.init_col[init_idx])
        self.sample_loc = random.choices((self.left_loc, self.right_loc))[0]
        self.dist_to_sample = abs(self.current_loc[0] - self.sample_loc[0]) + abs(
            self.current_loc[1] - self.sample_loc[1])
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]
        self.observation[self.current_loc] += self.color["current_loc"]  # show current location in observation
        self.observation[self.sample_loc] += self.color["touchscreen_loc"]  # turn on sample touchscreen
        self.reward = 0
        self.nav_reward = 0  # keep a record of total navigation reward (movement & poking signals)
        self.done = False

    def step(self, action):
        """
        :param action: one of the six actions
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if action in [0, 1, 2, 3, 4]:  # movement
            self.reward = self.step_reward
            self.nav_reward += self.reward
            next_loc = tuple(self.current_loc + self.directions[action])
            if not self.walls[next_loc]:  # if next location is not walls
                self.observation[self.current_loc] -= self.color["current_loc"]  # leave current location
                self.current_loc = next_loc  # update location
                self.observation[self.current_loc] += self.color["current_loc"]  # show on observation
            # if next location is incentive location, give incentive reward with probability incentive_probability
            if self.current_loc == self.incentive_loc and np.all(self.observation[self.sample_loc] == [0,0,0]):
                if self.rng.rand() < self.incentive_probability:
                    self.reward = self.incentive_magnitude
                    self.done = True
                else:
                    self.reward = 0
                    self.done = True
        else:  # poke
            if np.sum(self.observation[self.current_loc]) > 255:  # currently at a signal location
                if self.current_loc == self.sample_loc:  # currently at sample location
                    self.observation[self.current_loc] -= self.color["touchscreen_loc"]  # turn signal off
                    self.reward = self.poke_reward
                    self.nav_reward += self.reward
                else:
                    self.reward = self.step_reward
                    self.nav_reward += self.reward
            else:
                self.reward = self.step_reward
                self.nav_reward += self.reward
        return self.observation, self.reward, self.done, {}



class Tunl(object):
    def __init__(self, len_delay, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, rng_seed=1234):
        """
        :param len_delay: int
        :param len_edge: odd int -- length of long edge in triangle (minimum = 5)
        :param rwd: int -- size of reward when make a correct choice (eg. 100)
        :param inc_rwd: int  -- size of reward when make an incorrect choice (eg. -10)
        :param step_rwd: float -- size of reward for each step (eg. -0.5)
        :param poke_rwd: float -- size of reward when poke correctly (eg. 1)
        :param rng_seed: int -- seed for rng that generates initial location (default=1234)
        """
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
        self.rng = np.random.RandomState(rng_seed)

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
        self.delay_t = 0  # time since delay
        self.sample = "undefined"
        self.sample_loc = None
        self.current_loc = None
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]
        self.correction_trial = False

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
        self.delay_t = 0  # time since delay
        self.reward = 0
        self.nav_reward = 0  # keep a record of total navigation reward (movement & poking signals)
        self.done = False
        self.sample = "undefined"
        if not self.correction_trial:
            self.sample_loc = random.choices((self.left_loc, self.right_loc))[0]

    def step(self, action):
        """
        :param action
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if action in [0, 1, 2, 3, 4]:  # movement
            if not self.indelay:
                self.reward = self.step_rwd  # lightly punish all step actions except during delay
                self.nav_reward += self.reward
            next_loc = tuple(self.current_loc + self.directions[action])
            if not self.walls[next_loc]:  # if next location is not walls
                self.observation[self.current_loc] -= self.color["current_loc"]  # leave current location
                self.current_loc = next_loc  # update location
                self.observation[self.current_loc] += self.color["current_loc"]  # show on observation
        else:  # poke
            if np.sum(self.observation[self.current_loc]) > 255:  # currently at a signal location
                if self.current_loc == self.initiation_loc:  # currently at initiation signal location
                    self.observation[self.current_loc] -= self.color["initiation_loc"]  # turn signal off
                    self.reward = self.poke_rwd
                    self.nav_reward += self.reward
                    if self.sample == "undefined":  # initiate sample phase
                        self.observation[self.sample_loc] += self.color["touchscreen_loc"]  # turn on sample touchscreen
                    else:  # initiate choice phase
                        self.observation[self.left_loc] += self.color["touchscreen_loc"]
                        self.observation[self.right_loc] += self.color["touchscreen_loc"]  # turn on choice signals
                elif self.current_loc == self.left_loc:  # currently at left touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "L"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "L": # poked incorrectly at match location
                        self.reward = self.inc_rwd
                        self.correction_trial = True  # set the flag so the next trial gets the same sample_loc
                        self.done = True
                    elif self.sample == "R":  # poked correctly at nonmatch location
                        self.reward = self.rwd
                        self.correction_trial = False  # reset correction_trial Flag
                        self.done = True
                elif self.current_loc == self.right_loc:  # currently at the right touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "R"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "R":  # poked incorrectly at match location
                        self.reward = self.inc_rwd
                        self.correction_trial = True  # set the flag so the next trial gets the same sample_loc
                        self.done = True
                    elif self.sample == "L":  # poked correctly at nonmatch location
                        self.reward = self.rwd
                        self.correction_trial = False  # reset correction_trial Flag
                        self.done = True
            else:
                if not self.indelay:
                    self.reward = self.step_rwd  # lightly punish unnecessary poke actions unless during delay
                    self.nav_reward += self.reward
        if self.indelay:  # delay period
            if self.delay_t < self.len_delay:
                self.delay_t += 1
            else:
                self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
                self.indelay = False
            if not (
                    self.delay_t == 1 and self.indelay is True):  # unless just poked sample, in which case self.reward=5
                self.reward = 0

        return self.observation, self.reward, self.done, {}

    def calc_reward_without_stepping(self, action):
        """
        Calculate the hypothetical reward from taking this action, without actually changing the environment.
        :param action
        :return: reward
        """
        assert self.action_space.contains(action)
        reward = 0
        if action in [0, 1, 2, 3, 4]:  # movement
            if not self.indelay:
                reward = self.step_rwd  # lightly punish all step actions except during delay
        else:  # poke
            if np.sum(self.observation[self.current_loc]) > 255:  # currently at a signal location
                if self.current_loc == self.initiation_loc:  # currently at initiation signal location
                    reward = self.poke_rwd
                elif self.current_loc == self.left_loc:  # currently at left touchscreen
                    if self.sample == "undefined":  # enter delay
                        reward = self.poke_rwd
                    elif self.sample == "L": # poked incorrectly at match location
                        reward = self.inc_rwd
                    elif self.sample == "R":  # poked correctly at nonmatch location
                        reward = self.rwd
                elif self.current_loc == self.right_loc:  # currently at the right touchscreen
                    if self.sample == "undefined":  # enter delay
                        reward = self.poke_rwd
                    elif self.sample == "R":  # poked incorrectly at match location
                        reward = self.inc_rwd
                    elif self.sample == "L":  # poked correctly at nonmatch location
                        reward = self.rwd
            else:
                if not self.indelay:
                    reward = self.step_rwd  # lightly punish unnecessary poke actions unless during delay
        if self.indelay:  # delay period
            if not (self.delay_t == 1 and self.indelay is True):  # unless just poked sample, in which case reward=5
                reward = 0

        return reward


class Tunl_nomem(object):
    def __init__(self, len_delay, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, rng_seed=1234):
        """
        :param len_delay: int
        :param len_edge: odd int -- length of long edge in triangle (minimum = 5)
        :param rwd: int -- size of reward when make a correct choice (eg. 100)
        :param step_rwd: float -- size of reward for each step (eg. -0.5)
        :param poke_rwd: float -- size of reward when poke correctly (eg. 1)
        :param rng_seed: int -- seed for rng that generates initial location (default=1234)
        """
        assert len_edge % 2 == 1
        assert len_edge >= 5
        self.h = (len_edge + 1) // 2 + 2
        self.w = len_edge + 2

        assert rwd >= 0
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
        self.rng = np.random.RandomState(rng_seed)

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
        self.step_rwd = step_rwd
        self.poke_rwd = poke_rwd
        self.inc_rwd = inc_rwd
        self.indelay = False
        self.delay_t = 0  # time since delay
        self.sample = "undefined"
        self.sample_loc = None
        self.current_loc = None
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]

    def reset(self):
        init_idx = self.rng.choice(np.arange(len(self.init_row)))
        self.current_loc = (self.init_row[init_idx], self.init_col[init_idx])  # random initial location
        self.dist_to_init = abs(self.current_loc[0] - self.initiation_loc[0]) + abs(
            self.current_loc[1] - self.initiation_loc[1])
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]
        self.observation[self.current_loc] += self.color["current_loc"]  # show current location in observation
        self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
        self.sample = "undefined"
        self.sample_loc = random.choices((self.left_loc, self.right_loc))[0]
        self.indelay = False
        self.delay_t = 0  # time since delay
        self.reward = 0
        self.nav_reward = 0  # keep a record of total navigation reward (movement & poking signals)
        self.done = False

    def step(self, action):
        """
        :param action
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if action in [0, 1, 2, 3, 4]:  # movement
            if not self.indelay:
                self.reward = self.step_rwd  # lightly punish all step actions except during delay
                self.nav_reward += self.reward
            next_loc = tuple(self.current_loc + self.directions[action])
            if not self.walls[next_loc]:  # if next location is not walls
                self.observation[self.current_loc] -= self.color["current_loc"]  # leave current location
                self.current_loc = next_loc  # update location
                self.observation[self.current_loc] += self.color["current_loc"]  # show on observation
        else:  # poke
            if np.sum(self.observation[self.current_loc]) > 255:  # currently at a signal location
                if self.current_loc == self.initiation_loc:  # currently at initiation signal location
                    self.observation[self.current_loc] -= self.color["initiation_loc"]  # turn signal off
                    self.reward = self.poke_rwd
                    self.nav_reward += self.reward
                    if self.sample == "undefined":  # initiate sample phase
                        self.observation[self.sample_loc] += self.color["touchscreen_loc"]  # turn on sample touchscreen
                    else:  # initiate choice phase
                        self.observation[self.left_loc] += self.color["touchscreen_loc"]
                        self.observation[self.right_loc] += self.color["touchscreen_loc"]  # turn on choice signals
                elif self.current_loc == self.left_loc:  # currently at left touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "L"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "R" or self.sample == "L":  # Choose L is correct regardless of sample
                        self.reward = self.rwd
                        self.done = True
                elif self.current_loc == self.right_loc:  # currently at the right touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "R"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "L" or self.sample == "R":  # Choose R is incorrect regardless of sample
                        self.reward = self.inc_rwd
                        self.done = True
            else:
                if not self.indelay:
                    self.reward = self.step_rwd  # lightly punish unnecessary poke actions unless during delay
                    self.nav_reward += self.reward
        if self.indelay:  # delay period
            if self.delay_t < self.len_delay:
                self.delay_t += 1
            else:
                self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
                self.indelay = False
            if not (
                    self.delay_t == 1 and self.indelay == True):  # unless just poked sample, in which case self.reward=5
                self.reward = 0

        return self.observation, self.reward, self.done, {}


class Tunl_vd(object):
    '''
    Varying-delay TUNL. In each trial, the length of delay will be randomly sampled from len_delays with probability
    len_delays_p.
    '''

    def __init__(self, len_delays, len_delays_p, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, rng_seed=1234):
        """
        :param len_delays: list (eg. [20, 40, 60])
        :param len_delays_p: list (eg. [1,1,1])
        :param len_edge: odd int -- length of long edge in triangle (minimum = 5)
        :param rwd: int -- size of reward when make a correct choice (eg. 100)
        :param inc_rwd: int  -- size of reward when make an incorrect choice (eg. -10)
        :param step_rwd: float -- size of reward for each step (eg. -0.5)
        :param poke_rwd: float -- size of reward when poke correctly (eg. 1)
        :param rng_seed: int -- seed for rng that generates initial location (default=1234)
        """
        assert len_edge % 2 == 1
        assert len_edge >= 5
        self.h = (len_edge + 1) // 2 + 2
        self.w = len_edge + 2

        assert rwd >= 0
        assert inc_rwd <= 0
        assert step_rwd <= 0
        assert poke_rwd >= 0

        assert len(len_delays) == len(len_delays_p)

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
        self.rng = np.random.RandomState(rng_seed)

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
        self.len_delay = None
        self.len_delays = len_delays
        self.len_delays_p = len_delays_p
        self.rwd = rwd
        self.inc_rwd = inc_rwd
        self.step_rwd = step_rwd
        self.poke_rwd = poke_rwd
        self.indelay = False
        self.delay_t = 0  # time since delay
        self.sample = "undefined"
        self.sample_loc = None
        self.current_loc = None
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]
        self.correction_trial = False

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
        self.delay_t = 0  # time since delay
        self.reward = 0
        self.nav_reward = 0  # keep a record of total navigation reward (movement & poking signals)
        self.done = False
        self.sample = "undefined"
        if not self.correction_trial:
            self.sample_loc = random.choices((self.left_loc, self.right_loc))[0]
            self.len_delay = random.choices(self.len_delays, weights=self.len_delays_p, k=1)[0]

    def step(self, action):
        """
        :param action
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if action in [0, 1, 2, 3, 4]:  # movement
            if not self.indelay:
                self.reward = self.step_rwd  # lightly punish all step actions except during delay
                self.nav_reward += self.reward
            next_loc = tuple(self.current_loc + self.directions[action])
            if not self.walls[next_loc]:  # if next location is not walls
                self.observation[self.current_loc] -= self.color["current_loc"]  # leave current location
                self.current_loc = next_loc  # update location
                self.observation[self.current_loc] += self.color["current_loc"]  # show on observation
        else:  # poke
            if np.sum(self.observation[self.current_loc]) > 255:  # currently at a signal location
                if self.current_loc == self.initiation_loc:  # currently at initiation signal location
                    self.observation[self.current_loc] -= self.color["initiation_loc"]  # turn signal off
                    self.reward = self.poke_rwd
                    self.nav_reward += self.reward
                    if self.sample == "undefined":  # initiate sample phase
                        self.observation[self.sample_loc] += self.color["touchscreen_loc"]  # turn on sample touchscreen
                    else:  # initiate choice phase
                        self.observation[self.left_loc] += self.color["touchscreen_loc"]
                        self.observation[self.right_loc] += self.color["touchscreen_loc"]  # turn on choice signals
                elif self.current_loc == self.left_loc:  # currently at left touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "L"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "L":  # poked incorrectly at match location
                        self.reward = self.inc_rwd
                        self.done = True
                        self.correction_trial = True  # flag so the next trial gets the same sample_loc and len_delay
                    elif self.sample == "R":  # poked correctly at nonmatch location
                        self.reward = self.rwd
                        self.done = True
                        self.correction_trial = False  # reset correction_trial Flag
                elif self.current_loc == self.right_loc:  # currently at the right touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "R"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "R":  # poked incorrectly at match location
                        self.reward = self.inc_rwd
                        self.done = True
                        self.correction_trial = True  # flag so the next trial gets the same sample_loc and len_delay
                    elif self.sample == "L":  # poked correctly at nonmatch location
                        self.reward = self.rwd
                        self.done = True
                        self.correction_trial = False  # reset correction_trial Flag
            else:
                if not self.indelay:
                    self.reward = self.step_rwd  # lightly punish unnecessary poke actions unless during delay
                    self.nav_reward += self.reward
        if self.indelay:  # delay period
            if self.delay_t < self.len_delay:
                self.delay_t += 1
            else:
                self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
                self.indelay = False
            if not (
                    self.delay_t == 1 and self.indelay is True):  # unless just poked sample, in which case self.reward=5
                self.reward = 0

        return self.observation, self.reward, self.done, {}


class Tunl_nomem_vd(object):
    '''
    No-memory-load TUNL with varying delay. For each trial, the length of delay is randomly sampled from len_delays
    with probability len_delays_p
    '''
    def __init__(self, len_delays, len_delays_p, len_edge, rwd, step_rwd, poke_rwd, rng_seed=1234):
        """
        :param len_delays: list (eg. [20, 40, 60])
        :param len_delays_p: list (eg. [1,1,1])
        :param len_edge: odd int -- length of long edge in triangle (minimum = 5)
        :param rwd: int -- size of reward when make a correct choice (eg. 100)
        :param step_rwd: float -- size of reward for each step (eg. -0.5)
        :param poke_rwd: float -- size of reward when poke correctly (eg. 1)
        :param rng_seed: int -- seed for rng that generates initial location (default=1234)
        """
        assert len_edge % 2 == 1
        assert len_edge >= 5
        self.h = (len_edge + 1) // 2 + 2
        self.w = len_edge + 2

        assert rwd >= 0
        assert step_rwd <= 0
        assert poke_rwd >= 0

        assert len(len_delays) == len(len_delays_p)

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
        self.rng = np.random.RandomState(rng_seed)

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
        self.len_delay = None
        self.len_delays = len_delays
        self.len_delays_p = len_delays_p
        self.rwd = rwd
        self.step_rwd = step_rwd
        self.poke_rwd = poke_rwd
        self.indelay = False
        self.delay_t = 0  # time since delay
        self.sample = "undefined"
        self.sample_loc = None
        self.current_loc = None
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]

    def reset(self):
        init_idx = self.rng.choice(np.arange(len(self.init_row)))
        self.current_loc = (self.init_row[init_idx], self.init_col[init_idx])  # random initial location
        self.dist_to_init = abs(self.current_loc[0] - self.initiation_loc[0]) + abs(
            self.current_loc[1] - self.initiation_loc[1])
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]
        self.observation[self.current_loc] += self.color["current_loc"]  # show current location in observation
        self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
        self.sample = "undefined"
        self.sample_loc = random.choices((self.left_loc, self.right_loc))[0]
        self.len_delay = random.choices(self.len_delays, weights=self.len_delays_p, k=1)[0]
        self.indelay = False
        self.delay_t = 0  # time since delay
        self.reward = 0
        self.nav_reward = 0  # keep a record of total navigation reward (movement & poking signals)
        self.done = False

    def step(self, action):
        """
        :param action
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if action in [0, 1, 2, 3, 4]:  # movement
            if not self.indelay:
                self.reward = self.step_rwd  # lightly punish all step actions except during delay
                self.nav_reward += self.reward
            next_loc = tuple(self.current_loc + self.directions[action])
            if not self.walls[next_loc]:  # if next location is not walls
                self.observation[self.current_loc] -= self.color["current_loc"]  # leave current location
                self.current_loc = next_loc  # update location
                self.observation[self.current_loc] += self.color["current_loc"]  # show on observation
        else:  # poke
            if np.sum(self.observation[self.current_loc]) > 255:  # currently at a signal location
                if self.current_loc == self.initiation_loc:  # currently at initiation signal location
                    self.observation[self.current_loc] -= self.color["initiation_loc"]  # turn signal off
                    self.reward = self.poke_rwd
                    self.nav_reward += self.reward
                    if self.sample == "undefined":  # initiate sample phase
                        self.observation[self.sample_loc] += self.color["touchscreen_loc"]  # turn on sample touchscreen
                    else:  # initiate choice phase
                        self.observation[self.left_loc] += self.color["touchscreen_loc"]
                        self.observation[self.right_loc] += self.color["touchscreen_loc"]  # turn on choice signals
                elif self.current_loc == self.left_loc:  # currently at left touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "L"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "R" or self.sample == "L":  # poked either match or nonmatch
                        self.reward = self.rwd
                        self.done = True
                elif self.current_loc == self.right_loc:  # currently at the right touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "R"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "L" or self.sample == "R":  # poked correctly at nonmatch location
                        self.reward = self.rwd
                        self.done = True
            else:
                if not self.indelay:
                    self.reward = self.step_rwd  # lightly punish unnecessary poke actions unless during delay
                    self.nav_reward += self.reward
        if self.indelay:  # delay period
            if self.delay_t < self.len_delay:
                self.delay_t += 1
            else:
                self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
                self.indelay = False
            if not (
                    self.delay_t == 1 and self.indelay == True):  # unless just poked sample, in which case self.reward=5
                self.reward = 0

        return self.observation, self.reward, self.done, {}


class Tunl_incentive(object):
    """
    Same as Tunl, but during the delay period there's a chance (with probability p) of getting a small reward (1/a of the final reward).
    Added 11/24:
    1. self.choice_loc for tracking behaviour;
    2. if choose incorrectly then punish and end trial right away.
    2.5. if choose correctly then turns off signal, wait until navigate to initiation signal, then give reward and end trial.
    3. slightly punish during delay until incentivized.
    """
    def __init__(self, len_delay, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, p, a, rng_seed=1234):
        """
        :param len_delay: int
        :param len_edge: odd int -- length of long edge in triangle (minimum = 5)
        :param rwd: int -- size of reward when make a correct choice (eg. 100)
        :param inc_rwd: int  -- size of reward when make an incorrect choice (eg. -10)
        :param step_rwd: float -- size of reward for each step (eg. -0.5)
        :param poke_rwd: float -- size of reward when poke correctly (eg. 1)
        :param p: float -- probability of getting a small reward during delay (eg. 0.5)
        :param a: int -- the small reward is 1/a of the final reward (eg. 3)
        :param rng_seed: int -- seed for rng that generates initial location (default=1234)
        """
        assert len_edge % 2 == 1
        assert len_edge >= 5
        self.h = (len_edge + 1) // 2 + 2
        self.w = len_edge + 2

        assert rwd >= 0
        assert inc_rwd <= 0
        assert step_rwd <= 0
        assert poke_rwd >= 0
        assert 0 <= p <= 1
        assert a > 0

        self.walls = np.ones((self.h, self.w))
        i = 1
        while i < len_edge + 2 - i:
            self.walls[i, i:self.w - i] = 0
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
            self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1)),
                               np.array((0, 0))]

            # Random number generator
            self.rng = np.random.RandomState(rng_seed)

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
            self.delay_t = 0  # time since delay
            self.sample = "undefined"
            self.sample_loc = None
            self.current_loc = None
            self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
            self.observation[self.walls == 1] = self.color["walls"]
            self.correction_trial = False
            self.p = p
            self.a = a
            self.incentivized = False
            self.choice_loc = None

    def reset(self):
        init_idx = self.rng.choice(np.arange(len(self.init_row)))
        self.current_loc = (self.init_row[init_idx], self.init_col[init_idx])
        self.dist_to_init = abs(self.current_loc[0] - self.initiation_loc[0]) + abs(
            self.current_loc[1] - self.initiation_loc[1])
        self.observation = np.zeros((self.h, self.w, 3))
        self.observation[self.walls == 1] = self.color["walls"]
        self.observation[self.current_loc] += self.color["current_loc"]
        self.observation[self.initiation_loc] += self.color["initiation_loc"]
        self.indelay = False
        self.delay_t = 0
        self.reward = 0
        self.nav_reward = 0
        self.done = False
        self.sample = "undefined"
        self.choice_loc = None
        self.incentivized = False  # whether the agent has received small reward during delay or not. To make sure agent only gets one small reward per trial
        self.to_incentivize = self.rng.rand() < self.p  # whether to incentivize the agent during this trial
        if not self.correction_trial:
            self.sample_loc = random.choices((self.left_loc, self.right_loc))[0]

    def step(self, action):
        """
        :param action
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if action in [0, 1, 2, 3, 4]:  # movement
            if not self.indelay:
                self.reward = self.step_rwd  # lightly punish all step actions except during delay
                self.nav_reward += self.reward
            next_loc = tuple(self.current_loc + self.directions[action])
            if not self.walls[next_loc]:  # if next location is not walls
                self.observation[self.current_loc] -= self.color["current_loc"]  # leave current location
                self.current_loc = next_loc  # update location
                self.observation[self.current_loc] += self.color["current_loc"]  # show on observation
        else:  # poke
            if np.sum(self.observation[self.current_loc]) > 255:  # currently at a signal location
                if self.current_loc == self.initiation_loc:  # currently at initiation signal location
                    self.observation[self.current_loc] -= self.color["initiation_loc"]  # turn signal off
                    self.reward = self.poke_rwd
                    self.nav_reward += self.reward
                    if self.sample == "undefined":  # initiate sample phase
                        self.observation[self.sample_loc] += self.color["touchscreen_loc"]  # turn on sample touchscreen
                    else:  # initiate choice phase
                        self.observation[self.left_loc] += self.color["touchscreen_loc"]
                        self.observation[self.right_loc] += self.color["touchscreen_loc"]  # turn on choice signals
                elif self.current_loc == self.left_loc:  # currently at left touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "L"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "L": # poked incorrectly at match location
                        self.reward = self.inc_rwd
                        self.correction_trial = True  # set the flag so the next trial gets the same sample_loc
                        self.choice_loc = self.current_loc
                        self.done = True
                    elif self.sample == "R":  # poked correctly at nonmatch location
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                        self.correction_trial = False  # reset correction_trial Flag
                        self.choice_loc = self.current_loc
                elif self.current_loc == self.right_loc:  # currently at the right touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "R"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "R":  # poked incorrectly at match location
                        self.reward = self.inc_rwd
                        self.correction_trial = True  # set the flag so the next trial gets the same sample_loc
                        self.choice_loc = self.current_loc
                        self.done = True
                    elif self.sample == "L":  # poked correctly at nonmatch location
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                        self.correction_trial = False  # reset correction_trial Flag
                        self.choice_loc = self.current_loc
                        # Note: at this point, the agent should run to the back. But some dumb agents run to the match location and get punished
            else:
                if not self.indelay:
                    if self.current_loc == self.initiation_loc and self.choice_loc is not None:
                        self.reward = self.rwd
                        self.done = True
                    else:
                        self.reward = self.step_rwd  # lightly punish unnecessary poke actions unless during delay
                        self.nav_reward += self.reward

        if self.indelay:  # delay period

            if self.delay_t < self.len_delay:
                self.delay_t += 1
            else:
                self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
                self.indelay = False

            # If the agent navigates to the incentive location, then it gets a small reward with probability p if it hasn't been incentivized yet in this trial
            if self.current_loc == self.initiation_loc and self.to_incentivize and not self.incentivized:
                self.reward = self.rwd / self.a
                self.incentivized = True
            else:
                if not (self.delay_t == 1):  # unless just poked sample, in which case self.reward=5
                    if not self.incentivized:
                        self.reward = self.step_rwd
                    else:
                        self.reward = 0

        return self.observation, self.reward, self.done, {}


class Tunl_nomem_incentive(object):
    """
       Same as Tunl_nomem, but during the delay period there's a chance (with probability p) of getting a small reward (1/a of the final reward).
       """
    def __init__(self, len_delay, len_edge, rwd, inc_rwd, step_rwd, poke_rwd, p, a, rng_seed=1234):
        """
        :param len_delay: int
        :param len_edge: odd int -- length of long edge in triangle (minimum = 5)
        :param rwd: int -- size of reward when make a correct choice (eg. 100)
        :param step_rwd: float -- size of reward for each step (eg. -0.5)
        :param poke_rwd: float -- size of reward when poke correctly (eg. 1)
        :param p: float -- probability of getting a small reward during delay (eg. 0.5)
        :param a: int -- the small reward is 1/a of the final reward (eg. 3)
        :param rng_seed: int -- seed for rng that generates initial location (default=1234)

        """
        assert len_edge % 2 == 1
        assert len_edge >= 5
        self.h = (len_edge + 1) // 2 + 2
        self.w = len_edge + 2

        assert rwd >= 0
        assert step_rwd <= 0
        assert poke_rwd >= 0
        assert 0 <= p <= 1
        assert a > 0

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
        self.rng = np.random.RandomState(rng_seed)

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
        self.step_rwd = step_rwd
        self.poke_rwd = poke_rwd
        self.inc_rwd = inc_rwd
        self.indelay = False
        self.delay_t = 0  # time since delay
        self.sample = "undefined"
        self.sample_loc = None
        self.current_loc = None
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]
        self.p = p
        self.a = a
        self.incentivized = False
        self.choice_loc = None

    def reset(self):
        init_idx = self.rng.choice(np.arange(len(self.init_row)))
        self.current_loc = (self.init_row[init_idx], self.init_col[init_idx])  # random initial location
        self.dist_to_init = abs(self.current_loc[0] - self.initiation_loc[0]) + abs(
            self.current_loc[1] - self.initiation_loc[1])
        self.observation = np.zeros((self.h, self.w, 3))  # black for empty spaces in arena
        self.observation[self.walls == 1] = self.color["walls"]
        self.observation[self.current_loc] += self.color["current_loc"]  # show current location in observation
        self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
        self.sample = "undefined"
        self.sample_loc = random.choices((self.left_loc, self.right_loc))[0]
        self.indelay = False
        self.delay_t = 0  # time since delay
        self.reward = 0
        self.nav_reward = 0  # keep a record of total navigation reward (movement & poking signals)
        self.done = False
        self.incentivized = False  # whether the agent has received small reward during delay or not. To make sure agent only gets one small reward per trial
        self.to_incentivize = self.rng.rand() < self.p  # whether to incentivize the agent during this trial
        self.choice_loc = None

    def step(self, action):
        """
        :param action
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if action in [0, 1, 2, 3, 4]:  # movement
            if not self.indelay:
                self.reward = self.step_rwd  # lightly punish all step actions except during delay
                self.nav_reward += self.reward
            next_loc = tuple(self.current_loc + self.directions[action])
            if not self.walls[next_loc]:  # if next location is not walls
                self.observation[self.current_loc] -= self.color["current_loc"]  # leave current location
                self.current_loc = next_loc  # update location
                self.observation[self.current_loc] += self.color["current_loc"]  # show on observation
        else:  # poke
            if np.sum(self.observation[self.current_loc]) > 255:  # currently at a signal location
                if self.current_loc == self.initiation_loc:  # currently at initiation signal location
                    self.observation[self.current_loc] -= self.color["initiation_loc"]  # turn signal off
                    self.reward = self.poke_rwd
                    self.nav_reward += self.reward
                    if self.sample == "undefined":  # initiate sample phase
                        self.observation[self.sample_loc] += self.color["touchscreen_loc"]  # turn on sample touchscreen
                    else:  # initiate choice phase
                        self.observation[self.left_loc] += self.color["touchscreen_loc"]
                        self.observation[self.right_loc] += self.color["touchscreen_loc"]  # turn on choice signals
                elif self.current_loc == self.left_loc:  # currently at left touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "L"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "R" or self.sample == "L":  # Choose L is correct regardless of sample
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                        self.choice_loc = self.current_loc
                elif self.current_loc == self.right_loc:  # currently at the right touchscreen
                    self.observation[self.current_loc] -= self.color[
                        "touchscreen_loc"]  # turn off touchscreen signal
                    if self.sample == "undefined":  # enter delay
                        self.sample = "R"
                        self.indelay = True
                        self.reward = self.poke_rwd
                        self.nav_reward += self.reward
                    elif self.sample == "L" or self.sample == "R":  # Choose R is incorrect regardless of sample
                        self.reward = self.inc_rwd
                        self.choice_loc = self.current_loc
                        self.done = True
            else:
                if not self.indelay:
                    if self.current_loc == self.initiation_loc and self.choice_loc is not None:
                        self.reward = self.rwd
                        self.done = True
                    else:
                        self.reward = self.step_rwd  # lightly punish unnecessary poke actions unless during delay
                        self.nav_reward += self.reward
        if self.indelay:  # delay period
            if self.delay_t < self.len_delay:
                self.delay_t += 1
            else:
                self.observation[self.initiation_loc] += self.color["initiation_loc"]  # turn on initiation signal
                self.indelay = False

            # If the agent navigates to the incentive location, then it gets a small reward with probability p if it hasn't been incentivized yet in this trial
            if self.current_loc == self.initiation_loc and self.to_incentivize and not self.incentivized:
                self.reward = self.rwd / self.a
                self.incentivized = True
            else:
                if not (self.delay_t == 1):  # unless just poked sample, in which case self.reward=5
                    if not self.incentivized:
                        self.reward = self.step_rwd
                    else:
                        self.reward = 0

        return self.observation, self.reward, self.done, {}