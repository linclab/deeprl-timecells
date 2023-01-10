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
        receive a punishment (i.e. reward = -1). If the agent does not do anything meaningful, it will receive neither
        reward nor punishment (i.e, reward = 0). The L/R choices will stay on the touchscreen until the agent chooses
        the correct action.


        The observation space consists of 5 possible arrays, each with four elements: light, sound, L touchscreen, and
        R touchscreen. Each element is either 0 (off) or 1 (on).
        [1,1,0,0] = light and sound on; this signals the animal to initiate sample / choice phase
        [0,0,1,0] = L touchscreen on
        [0,0,0,1] = R touchscreen on
        [0,0,0,0] = delay period, no signal
        [0,0,1,1] = both L and R touchscreen on; this signals the animal to make a choice

        The action space consists of 4 possible actions:
        0 = initiate
        1 = choose L on the touchscreen
        2 = choose R on the touchscreen
        3 = do nothing meaningful

        """
        self.observation = array([[1, 1, 0, 0]])
        self.sample = "undefined"  # {array([[0,0,1,0]]=L, array([[0,0,0,1]])=R}
        self.delay_t = 0  # time since delay;
        self.delay_length = delay
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary(4)
        self.reward = 0
        self.done = False
        self.rng, self.np_seed = seeding.np_random(seed)

    def step(self, action, episode_sample):

        assert self.action_space.contains(action)

        if np.all(self.observation == array([[1, 1, 0, 0]])):  # initiation
            if action == 0:
                if self.sample == "undefined":  # initiate sample phase
                    self.observation = episode_sample  # either array([[0,0,1,0]]) or array([[0,0,0,1]])
                    self.sample = self.observation
                    self.done = False
                else:  # initiate choice phase
                    self.observation = array([[0, 0, 1, 1]])
                    self.done = False
            else:
                self.done = False
        elif np.all(self.observation == array([[0, 0, 1, 0]])):  # L touchscreen on
            if action == 1:  # poke L to continue
                self.observation = array([[0, 0, 0, 0]])  # enters delay period
                self.done = False
            else:
                self.done = False
        elif np.all(self.observation == array([[0, 0, 0, 1]])):  # R touchscreen on
            if action == 2:  # poke R to continue
                self.observation = array([[0, 0, 0, 0]])  # enters delay period
                self.done = False
            else:
                self.done = False
        elif np.all(self.observation == array([[0, 0, 0, 0]])):  # delay period
            if self.delay_t < self.delay_length:
                self.delay_t += 1
                self.done = False
            else:
                self.observation = array([[1, 1, 0, 0]])  # enters initiation
                self.done = False
        elif np.all(self.observation == array([[0, 0, 1, 1]])):  # choice phase
            if (np.all(self.sample == array([[0, 0, 1, 0]])) and action == 2) or (
                    np.all(self.sample == array([[0, 0, 0, 1]])) and action == 1):
                self.reward = 1
                self.done = True
            elif (np.all(self.sample == array([[0, 0, 1, 0]])) and action == 1) or (
                    np.all(self.sample == array([[0, 0, 0, 1]])) and action == 2):
                self.reward = -1
                self.done = False
            else:
                self.reward = 0
                self.done = False
        return self.observation, self.reward, self.done

    def reset(self):
        self.observation = array([[1, 1, 0, 0]])
        self.sample = "undefined"  # {array([0,0,1,0])=L, array([0,0,0,1])=R}
        self.delay_t = 0  # time since delay
        self.reward = 0
        self.done = False


class TunlEnv_nomem(object):
    '''
    In choice phase, agent may poke L or R to receive a reward and end trial.

    action space (4):
    0 = initiation
    1 = touch left sample to enter delay
    2 = touch right sample to enter delay
    3 = do nothing

    observation space: (5):
    array([[1,1,0,0]]) = light on, sound on
    array([[0,0,1,0]]) = left sample
    array([[0,0,0,1]]) = right sample
    array([[0,0,1,1]]) = waiting for choice
    array([[0,0,0,0]]) = delay

    '''
    def __init__(self, delay, seed=1):
        self.observation = array([[1, 1, 0, 0]])
        self.sample = "undefined"  # {array([[0,0,1,0]]=L, array([[0,0,0,1]])=R}
        self.delay_t = 0  # time since delay;
        self.delay_length = delay
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary(4)
        self.reward = 0
        self.done = False
        self.rng, self.np_seed = seeding.np_random(seed)

    def step(self, action, episode_sample):

        assert self.action_space.contains(action)

        if np.all(self.observation == array([[1, 1, 0, 0]])):  # initiation
            if action == 0:
                if self.sample == "undefined":  # initiate sample phase
                    self.observation = episode_sample  # either array([[0,0,1,0]]) or array([[0,0,0,1]])
                    self.sample = self.observation
                    self.done = False
                else:  # initiate choice phase
                    self.observation = array([[0, 0, 1, 1]])
                    self.done = False
            else:
                self.done = False
        elif np.all(self.observation == array([[0, 0, 1, 0]])):  # L touchscreen on
            if action == 1:  # poke L to continue
                self.observation = array([[0, 0, 0, 0]])  # enters delay period
                self.done = False
            else:
                self.done = False
        elif np.all(self.observation == array([[0, 0, 0, 1]])):  # R touchscreen on
            if action == 2:  # poke R to continue
                self.observation = array([[0, 0, 0, 0]])  # enters delay period
                self.done = False
            else:
                self.done = False
        elif np.all(self.observation == array([[0, 0, 0, 0]])):  # delay period
            if self.delay_t < self.delay_length:
                self.delay_t += 1
                self.done = False
            else:
                self.observation = array([[1, 1, 0, 0]])  # enters initiation
                self.done = False
        elif np.all(self.observation == array([[0, 0, 1, 1]])):  # choice phase
            if action == 1 or action == 2:  # poke L or R
                self.reward = 1
                self.done = True
            else:
                self.reward = 0
                self.done = False
        return self.observation, self.reward, self.done

    def reset(self):
        self.observation = array([[1, 1, 0, 0]])
        self.sample = "undefined"  # {array([0,0,1,0])=L, array([0,0,0,1])=R}
        self.delay_t = 0  # time since delay
        self.reward = 0
        self.done = False
