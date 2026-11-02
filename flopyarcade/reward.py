#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - reward
# author: Philipp Hoehn

#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade
# provides: simulated groundwater flow environments to test reinforcement learning algorithms
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


# imports for environments
from builtins import min as py_min, max as py_max
from copy import copy, deepcopy
from ctypes import sizeof, c_voidp
import gc
import hashlib
from matplotlib import use as matplotlibBackend
from matplotlib.ticker import NullLocator
matplotlibBackend('Agg')
from flopy.modflow import Modflow, ModflowBas, ModflowDis, ModflowLpf
from flopy.modflow import ModflowOc, ModflowPcg, ModflowWel
from flopy.mf6 import MFSimulation, ModflowTdis, ModflowIms, ModflowGwf
from flopy.mf6 import ModflowGwfdis, ModflowGwfic, ModflowGwfchd, ModflowGwfnpf
from flopy.mf6 import ModflowGwfoc, ModflowGwfsto, ModflowGwfrcha
from flopy.mf6.modflow.mfgwfwel import ModflowGwfwel
from flopy.modpath import Modpath6 as Modpath
from flopy.modpath import Modpath6Bas as ModpathBas
from flopy.plot import PlotMapView
from flopy.utils import HeadFile, PathlineFile
from flopy.mf6.utils.binarygrid_util import MfGrdFile
from glob import glob
import warnings
warnings.filterwarnings('ignore', 'SelectableGroups dict interface')
warnings.filterwarnings("ignore", category=ResourceWarning)
# warnings.filterwarnings("ignore", category=ResourceWarning, message="Implicitly cleaning up TemporaryDirectory")

import gymnasium as gym
from gymnasium import Env as gymEnv
from gymnasium import spaces
# from gymnasium.wrappers import EnvCompatibility
from imageio import get_writer, imread
from itertools import chain, product
from joblib import dump as joblibDump
from joblib import load as joblibLoad
from math import ceil, floor
from math import cos as mathCos
from math import sin as mathSin
from matplotlib import colormaps
# from matplotlib.colormaps import get_cmap
# from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import Circle, close, figure, pause, show
from matplotlib.pyplot import imsave
from matplotlib.pyplot import get_current_fig_manager
from matplotlib.pyplot import margins, NullLocator
from matplotlib.pyplot import waitforbuttonpress
from matplotlib import text as mtext
from matplotlib.transforms import Bbox
import numpy as np
from numpy import cos as numpyCos
from numpy import sin as numpySin
from numpy import pi as numpyPi
# from numpy import chararray
from numpy import add, arange, argmax, argsort, array, ceil, copy, concatenate, divide, empty
from numpy import expand_dims, extract, float32, frombuffer, int32, linspace, max, maximum, min, minimum
from numpy import mean, multiply, ones, prod, reshape, shape, sqrt, subtract, uint8, unique, zeros
from numpy import sum as numpySum
from numpy import abs as numpyAbs
from numpy.random import randint, random, randn, uniform
from numpy.random import seed as numpySeed
from os import chdir, chmod, environ, listdir, makedirs, pathsep, remove, rmdir, unlink
from os.path import abspath, dirname, exists, join, sep
from platform import system as platformSystem
from shutil import rmtree
from shutil import copy2
from sys import modules
if 'ipykernel' in modules:
    from IPython import display
from time import sleep, time
from xmipy import XmiWrapper

import flopy.utils.binaryfile as bf


# suppressing TensorFlow output on import, except fatal errors
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
from logging import getLogger, FATAL
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
getLogger('tensorflow').setLevel(FATAL)

# currently ignoring nested array error
# from numpy import VisibleDeprecationWarning
# warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
warnings.filterwarnings("ignore")

# additional imports for agents
from collections import deque, defaultdict
from datetime import datetime
from gc import collect as garbageCollect
from itertools import count, repeat
from pathos.multiprocessing import cpu_count
from pathos import helpers as pathosHelpers
# from multiprocessing import Pool as Pool
# from pathos.pools import _ProcessPool as Pool
# from pathos.pools import ProcessPool as Pool
# from pathos.pools import ParallelPool as Pool
from pathos.multiprocessing import Pool as Pool
from pathos.pools import _ThreadPool as ThreadPool
from pickle import dump, load
from tensorflow.keras.initializers import glorot_uniform, glorot_normal, random_uniform, random_normal
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import clone_model, load_model, model_from_json
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.random import SeedGenerator
from random import sample as randomSample, seed as randomSeed
from random import shuffle
from tensorflow.compat.v1 import ConfigProto, set_random_seed
from tensorflow.compat.v1 import Session as TFSession
from tensorflow.keras import backend as K
# from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import load_model as TFload_model
from tqdm import tqdm
from uuid import uuid4

from tempfile import TemporaryDirectory
from tempfile import NamedTemporaryFile

# avoiding freeze issues on Linux when loading Tensorflow model
# https://github.com/keras-team/keras/issues/9964
# https://stackoverflow.com/questions/40615795/pathos-enforce-spawning-on-linux
pathosHelpers.mp.context._force_start_method('spawn')

# to avoid thread erros
# https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
# matplotlibBackend('Agg')




class RewardMixin:

    def update_reward_on_fail(self):
        """Update the reward when the particle fails or hits a boundary.

        For environment type '5s-c-cost', the reward is set to the sum of
        costQFail and the current reward. Otherwise, if the current reward
        is non-negative, the reward is set to the negative of the current reward.

        Args:
            self (object): The instance of the class.

        """
        if self.ENVTYPE == '5s-c-cost':
            self.reward = self.costQFail + self.rewardCurrent
        else:
            if self.rewardCurrent >= 0.:
                self.reward = -self.rewardCurrent
            else:
                self.reward = 0.0
        self.rewardCurrent = 0.0

    def calculate_reward(self):
        """
        Calculate and update the reward based on the environment type.

        Args:
            self (object): Instance with environment type and reward calculation methods.

        """
        other_reward_calculators = {
            '5s-c-cost': self.calculateGameRewardOperationCost,
        }
        # Get the appropriate reward function or default to trajectory reward
        reward_func = other_reward_calculators.get(
            self.ENVTYPE, self.calculateGameRewardTrajectory
        )
        self.reward = reward_func()

    def calculateGameRewardTrajectory(self):
        """Calculate game reward based on deviation from the straightest path.

        Returns:
            float: Calculated reward value.

        """
        x, y = self.trajectories['x'][-1], self.trajectories['y'][-1]
        length_actual = self.calculatePathLength(x, y)
        length_shortest = x[-1] - x[0]

        if length_shortest > 0:
            return self._reward_forward(length_shortest, length_actual)
        if length_shortest == 0:
            return self._reward_no_movement()
        return self._reward_backward(length_shortest)

    def _reward_forward(self, length_shortest, length_actual):
        """Calculate reward for forward movement along path.

        Args:
            length_shortest (float): Straight-line distance from start to end.
            length_actual (float): Actual path length taken.

        Returns:
            float: Reward value.

        """
        dist_frac = length_shortest / self.distanceMax
        self.rewardMaxSegment = dist_frac * self.rewardMax
        path_ratio = length_shortest / length_actual
        reward = self.rewardMaxSegment * (path_ratio ** self.deviationPenaltyFactor)
        self.gameReward = reward
        return reward

    def _reward_no_movement(self):
        """Return zero reward for no movement.

        Returns:
            float: Zero reward.

        """
        self.gameReward = 0.0
        return 0.0

    def _reward_backward(self, length_shortest):
        """Calculate penalty reward for backward movement.

        Args:
            length_shortest (float): Negative straight-line distance.

        Returns:
            float: Negative reward penalty.

        """
        dist_frac = length_shortest / self.distanceMax
        self.rewardMaxSegment = dist_frac * self.rewardMax
        reward = -5 * abs(self.rewardMaxSegment)
        self.gameReward = reward
        return reward


    def calculateGameRewardHeadChange(self, targetField, realizedField):
        """Calculate reward based on deviation between target and realized fields.

        Args:
            targetField (numpy.ndarray): Target head field values.
            realizedField (numpy.ndarray): Realized head field values.

        Returns:
            float: Negative sum of absolute differences as reward.

        """
        return -np.sum(np.abs(targetField - realizedField))

    def calculateGameRewardOperationCost(self):
        """Calculate reward based on operation costs from pump and recharge flows.

        Returns:
            float: Computed reward based on summed flows.

        """
        pump, recharge = self._sum_flows()
        return self._compute_reward(pump, recharge)

    def _compute_reward(self, pump, recharge):
        """Compute reward based on pump and recharge costs.

        Args:
            pump (float): Total pump flow.
            recharge (float): Total recharge flow.

        Returns:
            float: Negative cost as reward.

        """
        return -pump * self.costQm3dPump - recharge * self.costQm3dRecharge

    def format_score(self) -> str:
        """Return formatted score string depending on environment type.

        Args:
            self (object): The instance of the class.

        Returns:
            str: Score string with units if applicable.
        
        """
        if self.ENVTYPE == '5s-c-cost':
            return f"score: {self.rewardCurrent:.2f} €"
        return f"score: {int(self.rewardCurrent)}"

