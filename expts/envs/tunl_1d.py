import numpy as np
from numpy import array
import random
from gym import spaces
from gym.utils import seeding


class TunlEnv(object):
    def __init__(self, delay, seed=1):
        """Trial-unique, delayed nonmatch-to-location (TUNL) task (Talpos et al., 2010)

        In each trial/episode, there are 5 stages:

        1. Initiation of sample phase: the agent will be given sound and light signal. In response, the agent must
        select the "initiate" action from the action space to proceed to sample phase. The trial will not proceed until
        the agent has selected the "initiate" action.

        2. Sample phase: the agent will be given one of two random samples: either left side (L) or the right side (R)
        of the touchscreen will be lit. The agent must poke the sample location on the touchscreen to proceed to the
        delay period. The trial will not proceed until the agent has poked the sample location.

        3. Delay: there will not be any stimulus during the delay period. The agent needs to memorize the sample during
        this period. Nothing that the agent does during delay period will have any consequences.

        4. Initiation of choice phase: after the delay period, the sound and light signal will be on again. In response,
        the agent must select the "initiate" action from the action space to proceed to choice phase. The trial will not
        proceed until the agent has selected the "initiate" action.

        5. Choice phase: the agent must choose the side of the touchscreen that was NOT lit in the sample phase
        (i.e. if the sample was L, the agent must select action R from action space). If agent correctly select the
        nonmatching action, it will receive a reward of 1. If the agent incorrectly select the matching action, it will
        receive a punishment (i.e. reward = -1), and be presented the same sample in the next trial (i.e. correction trial).
        If the agent does not do anything meaningful, it will receive neither reward nor punishment (i.e, reward = 0).


        The observation space consists of 5 possible arrays, each with 3 elements: initiation, L touchscreen, and
        R touchscreen. Each element is either 0 (off) or 1 (on).
        [1,0,0] = initiation on; this signals the animal to initiate sample / choice phase
        [0,1,0] = L touchscreen on
        [0,0,1] = R touchscreen on
        [0,0,0] = delay period, no signal
        [0,1,1] = both L and R touchscreen on; this signals the animal to make a choice

        The action space consists of 3 possible actions:
        0 = initiate
        1 = choose L on the touchscreen
        2 = choose R on the touchscreen

        """
        self.observation = [1, 0, 0]
        self.sample = "undefined"
        self.episode_sample = None
        self.delay_t = 0  # time since delay
        self.delay_length = delay
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(3)
        self.reward = 0
        self.done = False
        self.rng, self.np_seed = seeding.np_random(seed)
        self.correction_trial = False
        
    def reset(self):
        self.observation = [1, 0, 0]
        self.delay_t = 0  # time since delay
        self.reward = 0
        self.done = False
        self.sample = "undefined"  # {[0,1,0]=L, [0,0,1]=R}
        if self.correction_trial is False:
            self.episode_sample = random.choices(([0, 1, 0], [0, 0, 1]))[0]
            
    def step(self, action):
        """
        :param action:
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        if self.observation == [1, 0, 0]:  # initiation
            if action == 0:
                if self.sample == "undefined":  # initiate sample phase
                    self.observation = self.episode_sample  # either array([[0,1,0]]) or array([[0,0,1]])
                    self.sample = self.observation
                else:  # initiate choice phase
                    self.observation = [0, 1, 1]
        elif self.observation == [0, 1, 0]:  # L touchscreen on
            if action == 1:  # poke L to continue
                self.observation = [0, 0, 0]  # enters delay period
        elif self.observation == [0, 0, 1]:  # R touchscreen on
            if action == 2:  # poke R to continue
                self.observation = [0, 0, 0]  # enters delay period
        elif self.observation == [0, 0, 0]:  # delay period
            if self.delay_t < self.delay_length:
                self.delay_t += 1
            else:
                self.observation = [1, 0, 0]  # enters initiation
        elif self.observation == [0, 1, 1]:  # choice phase
            if (self.sample == [0, 1, 0] and action == 2) or (
                    self.sample == [0, 0, 1] and action == 1):  # choosing nonmatch
                self.reward = 1
                self.correction_trial = False  # reset correction_trial flag
                self.done = True
            elif (self.sample == [0, 1, 0] and action == 1) or (
                    self.sample == [0, 0, 1] and action == 2):  # choosing match
                self.reward = -1
                self.correction_trial = True  # set correction_trial flag to true to have the same sample in next trial
                self.done = True
            else:
                self.reward = 0
        return self.observation, self.reward, self.done, {}




class Tunl_simple(object):
    """
    States:
    [1,1]: init
    [0,0]: delay
    [1,0]: left sample
    [0,1]: right sample

    Actions:
    0: poke left & initiation
    1: poke right
    """
    def __init__(self, len_delay=40, rwd=10, inc_rwd=-10, seed=1):
        self.len_delay = len_delay
        self.rwd = rwd
        self.inc_rwd = inc_rwd
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiBinary(2)
        self.rng, self.np_seed = seeding.np_random(seed)
        self.reward = 0  # reward at this time step, not the cumulative reward of this episode
        self.task_stage = 'init'
        self.done = False
        self.delay_t = 0  # time since delay
        self.observation = [1,1]
        self.sample = None
        self.correction_trial = False

    def reset(self):
        self.reward = 0
        self.task_stage = 'init'
        self.done = False
        self.observation = [1,1]
        self.delay_t = 0  # time since delay
        if not self.correction_trial:  # receives new sample. Otherwise repeat old sample.
            self.sample = random.choices(([1,0],[0,1]))[0]  # [1,0] or [0,1]

    def step(self, action=None):
        if self.task_stage == 'init':
            if action == 0:
                self.task_stage = 'sample'
                self.observation = self.sample
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == "sample":
            if (self.observation == [1,0] and action == 0) or (self.observation == [0,1] and action == 1):  # touch sample to enter delay
                self.task_stage = 'delay'
                self.observation = [0,0]
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == 'delay':
            if self.delay_t < self.len_delay:
                self.delay_t += 1
            else:
                self.task_stage = 'choice'
                self.observation = [1,1]

        elif self.task_stage == 'choice':
            if (self.sample == [1,0] and action == 1) or (self.sample == [0,1] and action == 0):  # choose nonmatch
                self.reward = self.rwd
                self.correction_trial = False
            else:
                self.reward = self.inc_rwd
                self.correction_trial = True
            self.done = True

        return self.observation, self.reward, self.done


class Tunl_simple_nomem(object):
    """
    States:
    [1,1]: init
    [0,0]: delay
    [1,0]: left sample
    [0,1]: right sample

    Actions:
    0: poke left & initiation
    1: poke right
    """
    def __init__(self, len_delay=40, rwd=10, inc_rwd=-10, seed=1):
        self.len_delay = len_delay
        self.rwd = rwd
        self.inc_rwd = inc_rwd
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiBinary(2)
        self.rng, self.np_seed = seeding.np_random(seed)
        self.reward = 0  # reward at this time step, not the cumulative reward of this episode
        self.task_stage = 'init'
        self.done = False
        self.delay_t = 0  # time since delay
        self.observation = [1,1]
        self.sample = None

    def reset(self):
        self.reward = 0
        self.task_stage = 'init'
        self.done = False
        self.observation = [1,1]
        self.delay_t = 0  # time since delay
        self.sample = random.choices(([1,0],[0,1]))[0]  # [1,0] or [0,1]

    def step(self, action=None):
        if self.task_stage == 'init':
            if action == 0:
                self.task_stage = 'sample'
                self.observation = self.sample
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == "sample":
            if (self.observation == [1,0] and action == 0) or (self.observation == [0,1] and action == 1):  # touch sample to enter delay
                self.task_stage = 'delay'
                self.observation = [0,0]
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == 'delay':
            if self.delay_t < self.len_delay:
                self.delay_t += 1
            else:
                self.task_stage = 'choice'
                self.observation = [1,1]

        elif self.task_stage == 'choice':
            if action == 0 or action == 1:  # make a choice
                self.reward = self.rwd
                self.done = True

        return self.observation, self.reward, self.done


class TunlEnv_nomem(object):
    """
    For each episode, the agent receives a sample (L/R), experiences a delay,
    then is given two choices (L/R) - choosing either leads to a reward.

    action space (3):
    0 = initiation
    1 = touch left sample to enter delay
    2 = touch right sample to enter delay

    observation space: (5):
    array([[1,0,0]]) = light on, sound on
    array([[0,1,0]]) = left sample
    array([[0,0,1]]) = right sample
    array([[0,1,1]]) = waiting for choice
    array([[0,0,0]]) = delay
    """

    def __init__(self, delay, seed=1):
        self.observation = [1, 0, 0]
        self.delay_t = 0  # time since delay
        self.delay_length = delay
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(3)
        self.reward = 0
        self.done = False
        self.sample = "undefined"
        self.episode_sample = None
        self.rng, self.np_seed = seeding.np_random(seed)

    def reset(self):
        self.observation = [1, 0, 0]
        self.delay_t = 0  # time since delay
        self.reward = 0
        self.done = False
        self.sample = "undefined"  # {[0,1,0]=L, [0,0,1]=R}
        self.episode_sample = random.choices(([0, 1, 0], [0, 0, 1]))[0]

    def step(self, action):
        """
        :param action:
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        # print("action is", action)
        if self.observation == [1, 0, 0]:  # initiation
            if action == 0:
                if self.sample == "undefined":  # initiate sample phase
                    self.observation = self.episode_sample  # either array([[0,0,1,0]]) or array([[0,0,0,1]])
                    self.sample = self.observation
                else:  # initiate choice phase
                    self.observation = [0, 1, 1]
        elif self.observation == [0, 1, 0]:  # L touchscreen on
            if action == 1:  # poke L to continue
                self.observation = [0, 0, 0]  # enters delay period
        elif self.observation == [0, 0, 1]:  # R touchscreen on
            if action == 2:  # poke R to continue
                self.observation = [0, 0, 0]  # enters delay period
        elif self.observation == [0, 0, 0]:  # delay period
            if self.delay_t < self.delay_length:
                self.delay_t += 1
            else:
                self.observation = [1, 0, 0]  # enters initiation
        elif self.observation == [0, 1, 1]:  # waits for choice
            if action == 1 or action == 2:  # poke L or R
                self.observation = [1, 0, 0]
                self.reward = 1
                self.done = True
        return self.observation, self.reward, self.done, {}


