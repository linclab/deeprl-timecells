import numpy as np
import random
from gym import spaces

'''
Trial-Unique, Delayed Nonmatch-to-Location (TUNL) task in a triangular arena. 
Including the no-memory verion.
'''


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
    def __init__(self, len_delay, len_edge, rwd, step_rwd, poke_rwd, rng_seed=1234):
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