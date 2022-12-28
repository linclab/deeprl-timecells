import numpy as np
from numpy import array
import random
import copy
from gym import spaces


class IntervalDiscrimination(object):
    '''
    Interval discrimination, head-fixed agent. After an initiation cue, the agent pokes to initiate the task.
    Then two stimulus are shown sequentially, seperated by a delay period. The duration of each stimulus is
    randomly drawn from [10,15,20,25,30,35,40]. The length of the delay period is 20 time steps. After the second
    stimulus presentation, a "Go" cue would show up, and the agent is going to produce an action of either "0"
    or "1" to indicate whether the first stimulus was longer in duration or the second one.
    '''

    def __init__(self, rwd=10, inc_rwd=-10):
        self.stimulus_set = [10,15,20,25,30,35,40]
        self.delay_duration = 20
        self.action_space = spaces.Discrete(2)      # Boolean variable that stim_1 > stim_2
        self.observation_space = spaces.MultiBinary(2)
        self.rng = np.random.RandomState(1234)
        self.reward = 0
        self.task_stage = 'init'
        self.done = False
        self.first_stim = np.random.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.rwd = rwd
        self.inc_rwd = inc_rwd
        self.elapsed_t = 0
        self.correct_trial = False
        self.observation = [1,1]
        self.groundtruth = self.first_stim > self.second_stim

    def reset(self):
        self.reward = 0
        self.task_stage = 'init'
        self.done = False
        self.first_stim = np.random.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.groundtruth = self.first_stim > self.second_stim  # 1 if L1>L2, 0 if L1<L2
        self.elapsed_t = 0
        self.correct_trial = False
        self.observation = [1,1]

    def step(self, action=None):
        """
        :param action
        :return: observation, reward, done, info
        """

        if self.task_stage == "init":               # the agent needs to take an action
            if action == 1:
                self.task_stage = "first_stim"
                self.observation = [1,0]
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == "first_stim":
            if self.elapsed_t >= self.first_stim:
                self.task_stage = "delay_init"
                self.observation = [1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "delay_init":
            self.task_stage = "delay"
            self.observation = [0,0]
        elif self.task_stage == "delay":
            if self.elapsed_t >= self.delay_duration:
                self.task_stage = "delay_end"
                self.observation = [1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "delay_end":
            self.task_stage = "second_stim"
            self.observation = [0,1]
        elif self.task_stage == "second_stim":
            if self.elapsed_t >= self.second_stim:
                self.task_stage = "choice_init"
                self.observation = [1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "choice_init":                      # the agent needs to take an action
            if action == self.groundtruth:
                self.reward = self.rwd
                self.correct_trial = True
            else:
                self.reward = self.inc_rwd
            self.done = True

        return self.observation, self.reward, self.done


    def select_second_stim(self):
        stimulus_set_copy = copy.deepcopy(self.stimulus_set)
        stimulus_set_copy.remove(self.first_stim)
        return np.random.choice(stimulus_set_copy)