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




from .flopyarcade import FloPyEnv
from .flopyarcade import FloPyEnv as FloPyEnvSurrogate


class FloPyArcade:
    """Instance of a FloPyArcade game.

    Initializes a game environment and allows playing the game.
    """

    def __init__(self, agent=None, modelNameLoad=None, modelName='FloPyArcade',
                 animationFolder=None, NAGENTSTEPS=200, PATHMF2005=None,
                 PATHMP6=None, surrogateSimulator=None, flagSavePlot=False,
                 flagManualControl=False, actions=None, flagRender=False,
                 keepTimeSeries=False, ENVTYPE='1s-d', nLay=1, nRow=100,
                 nCol=100, OBSPREP='perceptron'):
        """Constructor.

        Args:
            self (object): The instance of the class.
            agent (object, optional): Game agent. Defaults to None.
            modelNameLoad (str, optional): Model name to load. Defaults to None.
            modelName (str, optional): Model name. Defaults to 'FloPyArcade'.
            animationFolder (str, optional): Animation folder. Defaults to None.
            NAGENTSTEPS (int, optional): Number of agent steps. Defaults to 200.
            PATHMF2005 (str, optional): Path to MODFLOW-2005. Defaults to None.
            PATHMP6 (str, optional): Path to MODPATH6. Defaults to None.
            surrogateSimulator (object, optional): Surrogate sim. Defaults to None.
            flagSavePlot (bool, optional): Save plots. Defaults to False.
            flagManualControl (bool, optional): Manual control. Defaults to False.
            actions (list, optional): Predefined actions. Defaults to None.
            flagRender (bool, optional): Render flag. Defaults to False.
            keepTimeSeries (bool, optional): Keep time series. Defaults to False.
            ENVTYPE (str, optional): Environment type. Defaults to '1s-d'.
            nLay (int, optional): Number of layers. Defaults to 1.
            nRow (int, optional): Number of rows. Defaults to 100.
            nCol (int, optional): Number of columns. Defaults to 100.
            OBSPREP (str, optional): Observation prep. Defaults to 'perceptron'.
        
        """
        self._init_config(
            agent, modelNameLoad, modelName, animationFolder, NAGENTSTEPS,
            PATHMF2005, PATHMP6, surrogateSimulator, flagSavePlot,
            flagManualControl, actions, flagRender, keepTimeSeries,
            ENVTYPE, nLay, nRow, nCol, OBSPREP
        )

    def _init_config(self, agent, modelNameLoad, modelName, animationFolder,
                     NAGENTSTEPS, PATHMF2005, PATHMP6, surrogateSimulator,
                     flagSavePlot, flagManualControl, actions, flagRender,
                     keepTimeSeries, ENVTYPE, nLay, nRow, nCol, OBSPREP):
        """Initialize configuration for FloPyArcade.

        Args:
            self (object): The instance of the class.
            ... (see __init__ for argument details)
        
        """
        self.ENVTYPE = str(ENVTYPE)
        self.PATHMF2005 = PATHMF2005
        self.PATHMP6 = PATHMP6
        self.SURROGATESIMULATOR = surrogateSimulator
        self.NAGENTSTEPS = NAGENTSTEPS
        self.SAVEPLOT = flagSavePlot
        self.MANUALCONTROL = flagManualControl
        self.RENDER = flagRender
        self.MODELNAME = modelName or modelNameLoad or "untitled"
        self.ANIMATIONFOLDER = (
            animationFolder or modelName or modelNameLoad or "untitled"
        )
        self.agent = agent
        self.MODELNAMELOAD = modelNameLoad
        self.done = False
        self.keepTimeSeries = keepTimeSeries
        self.actions = actions
        self.nLay = nLay
        self.nRow = nRow
        self.nCol = nCol
        self.OBSPREP = OBSPREP
        self.ENVTYPES = [
            '0s-c', '1s-d', '1s-c', '1r-d', '1r-c', '2s-d', '2s-c', '2r-d',
            '2r-c', '3s-d', '3s-c', '3r-d', '3r-c', '4s-c', '4r-c', '5s-c',
            '5s-c-cost', '5r-c', '6s-c', '6r-c'
        ]
        self.wrkspc = FloPyAgent().wrkspc

    def play(self, env=None, ENVTYPE='1s-d', seed=None,
             returnReward=False, verbose=False):
        """Play an instance of the FloPy arcade game.

        Args:
            self (object): The instance of the class.
            env (object, optional): Pre-initialized environment. Defaults to None.
            ENVTYPE (str, optional): Environment type. Defaults to '1s-d'.
            seed (int, optional): Random seed. Defaults to None.
            returnReward (bool, optional): Return total reward. Defaults to False.
            verbose (bool, optional): Print summary. Defaults to False.

        Returns:
            float or None: Total reward if returnReward is True, else None.
        
        """
        t0 = time()
        self._setup_env(env, ENVTYPE, seed)
        self._setup_game()
        self._run_game_loop()
        self._finalize_game(verbose, t0)
        if returnReward:
            return self.rewardTotal

    def _setup_env(self, env, ENVTYPE, seed):
        """Set up or assign the FloPy environment.

        Args:
            self (object): The instance of the class.
            env (object): Pre-initialized environment or None.
            ENVTYPE (str): Environment type.
            seed (int or None): Random seed.
        
        """
        if env is not None:
            self.env = env
        elif self.SURROGATESIMULATOR is None:
            self.env = FloPyEnv(
                ENVTYPE=ENVTYPE, PATHMF2005=self.PATHMF2005,
                PATHMP6=self.PATHMP6, _seed=seed, MODELNAME=self.MODELNAME,
                ANIMATIONFOLDER=self.ANIMATIONFOLDER, flagSavePlot=self.SAVEPLOT,
                flagManualControl=self.MANUALCONTROL, flagRender=self.RENDER,
                NAGENTSTEPS=self.NAGENTSTEPS, nLay=self.nLay, nRow=self.nRow,
                nCol=self.nCol, OBSPREP=self.OBSPREP, initWithSolution=True
            )
        else:
            self.env = FloPyEnvSurrogate(
                self.SURROGATESIMULATOR, ENVTYPE, MODELNAME=self.MODELNAME,
                _seed=seed, NAGENTSTEPS=self.NAGENTSTEPS
            )

    def _setup_game(self):
        """Prepare game variables and agent for a new episode.

        Args:
            self (object): The instance of the class.
        
        """
        self.observations = self.env.observationsVectorNormalized
        self.done = self.env.done
        if self.keepTimeSeries:
            self._init_time_series(self.observations)
        self.actionSpace = self.env.actionSpace
        self.agentObj = FloPyAgent(actionSpace=self.actionSpace)
        self.success = False
        self.rewardTotal = 0.

    def _run_game_loop(self):
        """Main game loop for agent-environment interaction.

        Args:
            self (object): The instance of the class.
        
        """
        for self.timeSteps in range(self.NAGENTSTEPS):
            if self.done:
                break
            action = self._get_action()
            self.observations, reward, self.done, _, _ = self.env.step(action)
            self.rewardTotal += reward
            if self.keepTimeSeries:
                self._update_time_series(reward, action)

    def _get_action(self):
        """Select the next action for the agent.

        Args:
            self (object): The instance of the class.

        Returns:
            object: Selected action.
        
        """
        if self.actions:
            return self.actions[self.timeSteps]
        if self.MANUALCONTROL:
            return self.agentObj.getAction(
                'manual', self.env.keyPressed, actionType=self.env.actionType
            )
        if self.MODELNAMELOAD:
            return self.agentObj.getAction(
                'modelNameLoad', modelNameLoad=self.MODELNAMELOAD,
                state=self.env.observationsVectorNormalized,
                actionType=self.env.actionType
            )
        if self.agent:
            return self.agentObj.getAction(
                'model', agent=self.agent,
                state=self.env.observationsVectorNormalized,
                actionType=self.env.actionType
            )
        return self.agentObj.getAction('random', actionType=self.env.actionType)

    def _init_time_series(self, initial_observation):
        """Initialize time series storage for the episode.

        Args:
            self (object): The instance of the class.
            initial_observation (object): Initial observation from environment.
        
        """
        self.timeSeries = {
            'statesNormalized': [initial_observation],
            'stressesNormalized': [],
            'rewards': [0.],
            'doneFlags': [self.done],
            'successFlags': [-1],
            'heads': [self.env.heads],
            'wellCoords': [self.env.wellCoords],
            'actions': [],
            'trajectories': []
        }

    def _update_time_series(self, reward, action):
        """Update time series data after each step.

        Args:
            self (object): The instance of the class.
            reward (float): Reward received at this step.
            action (object): Action taken at this step.
        
        """
        ts = self.timeSeries
        ts['statesNormalized'].append(self.env.observationsVectorNormalized)
        ts['stressesNormalized'].append(self.env.stressesVectorNormalized)
        ts['rewards'].append(reward)
        ts['heads'].append(self.env.heads)
        ts['doneFlags'].append(self.done)
        ts['wellCoords'].append(self.env.wellCoords)
        ts['actions'].append(action)
        ts['successFlags'].append(-1 if not self.done else int(self.env.success))

    def _finalize_game(self, verbose, t0):
        """Finalize the game, print results, and clean up.

        Args:
            self (object): The instance of the class.
            verbose (bool): Print game summary.
            t0 (float): Start time for runtime calculation.
        
        """
        if self.MANUALCONTROL:
            sleep(5)
        if self.keepTimeSeries:
            self.timeSeries['trajectories'] = self.env.trajectories
        self.success = self.env.success
        if verbose:
            outcome = 'won' if self.env.success else 'lost'
            surrogate = 'surrogate ' if self.SURROGATESIMULATOR else ''
            print(
                f"The {surrogate}game was {outcome} after "
                f"{self.timeSteps + 1} timesteps "
                f"with a reward of {int(self.rewardTotal)} points."
            )
        close('all')
        if self.env.ENVTYPE == '0s-c':
            self.env.teardown()
        self.runtime = (time() - t0) / 60.


