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
from os import chdir, chmod, environ, listdir, makedirs, pathsep, remove, rmdir
from os.path import abspath, dirname, exists, join, sep
from platform import system as platformSystem
from shutil import rmtree
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

# avoiding freeze issues on Linux when loading Tensorflow model
# https://github.com/keras-team/keras/issues/9964
# https://stackoverflow.com/questions/40615795/pathos-enforce-spawning-on-linux
pathosHelpers.mp.context._force_start_method('spawn')

# to avoid thread erros
# https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
# matplotlibBackend('Agg')


class FloPyEnv(gym.Env):
    """Environment for forward simulation using MODFLOW and MODPATH.

    On the first call, this environment:
        - Initializes a model with a randomly placed operating well.
        - Initializes the corresponding steady-state flow solution as the starting state.
        - Initializes a random starting action.
        - Initializes a random particle on the western side.

    On each step call, this environment:
        - Loads the current state.
        - Tracks the particle's trajectory through the model domain.
        - Returns the environment's new state.
        - Returns the new particle location as an observation.
        - Returns a flag indicating whether the particle has reached the operating well.
    """
    
    # necessary to render with rllib
    # https://discourse.aicrowd.com/t/how-to-save-rollout-video-render/3246/9
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 ENVTYPE='1s-d',
                 PATHMF2005=None,
                 PATHMP6=None,
                 MODELNAME='FloPyArcade',
                 ANIMATIONFOLDER='FloPyArcade',
                 _seed=None,
                 flagSavePlot=False,
                 flagManualControl=False,
                 manualControlTime=0.1,
                 flagRender=False,
                 NAGENTSTEPS=200,
                 nLay=1,
                 nRow=100,
                 nCol=100,
                 OBSPREP='perceptron',
                 initWithSolution=True,
                 PATHMF6DLL=None,
                 env_config=None):
        """Initialize the FloPyEnv environment and configuration.

        Args:
            self (object): The instance of the class.
            ENVTYPE (str): Environment type. Defaults to '1s-d'.
            PATHMF2005 (str, optional): Path to MODFLOW-2005 executable. Defaults to None.
            PATHMP6 (str, optional): Path to MODPATH6 executable. Defaults to None.
            MODELNAME (str): Name of the model. Defaults to 'FloPyArcade'.
            ANIMATIONFOLDER (str): Folder for animation output. Defaults to 'FloPyArcade'.
            _seed (int, optional): Random seed for reproducibility. Defaults to None.
            flagSavePlot (bool): Flag to save plots. Defaults to False.
            flagManualControl (bool): Flag to enable manual control. Defaults to False.
            manualControlTime (float): Time interval for manual control steps. Defaults to 0.1.
            flagRender (bool): Flag to enable rendering. Defaults to False.
            NAGENTSTEPS (int): Number of agent steps. Defaults to 200.
            nLay (int): Number of model layers. Defaults to 1.
            nRow (int): Number of model rows. Defaults to 100.
            nCol (int): Number of model columns. Defaults to 100.
            OBSPREP (str): Observation preparation method. Defaults to 'perceptron'.
            initWithSolution (bool): Whether to initialize with solution. Defaults to True.
            PATHMF6DLL (str, optional): Path to MODFLOW 6 DLL. Defaults to None.
            env_config (dict, optional): Additional environment configuration overrides.

        """

        self._init_env_config(locals(), env_config)
        self._setup_workspace()
        self._init_env_vars(_seed)
        self._init_env_specific()

        if self.initWithSolution:
            self.step(action='keep')
            self._setup_gym_spaces()

    def _init_env_config(self, local_vars, env_config):
        """Set environment configuration defaults and override with provided config.

        Sets instance attributes based on default parameters and any overrides
        provided in `env_config`.

        Args:
            self (object): The instance of the class.
            local_vars (dict): Local variables from __init__ method.
            env_config (dict or None): Environment configuration overrides.

        """
        env_config = env_config or {}
        defaults = {
            'ENVTYPE': local_vars['ENVTYPE'],
            'PATHMF2005': local_vars['PATHMF2005'],
            'PATHMP6': local_vars['PATHMP6'],
            'MODELNAME': local_vars['MODELNAME'],
            'ANIMATIONFOLDER': local_vars['ANIMATIONFOLDER'],
            'flagSavePlot': local_vars['flagSavePlot'],
            'flagManualControl': local_vars['flagManualControl'],
            'manualControlTime': local_vars['manualControlTime'],
            'flagRender': local_vars['flagRender'],
            'NAGENTSTEPS': local_vars['NAGENTSTEPS'],
            'nLay': local_vars['nLay'],
            'nRow': local_vars['nRow'],
            'nCol': local_vars['nCol'],
            'OBSPREP': local_vars['OBSPREP'],
            'initWithSolution': local_vars['initWithSolution'],
            'PATHMF6DLL': local_vars['PATHMF6DLL'],
        }
        # Merge overrides
        self.env_config = {**defaults, **env_config}

        # Set instance attributes
        for key, value in self.env_config.items():
            setattr(self, key, value)

        # Aliases for flags
        self.SAVEPLOT = self.flagSavePlot
        self.MANUALCONTROL = self.flagManualControl
        self.MANUALCONTROLTIME = self.manualControlTime
        self.RENDER = self.flagRender

        self.versionMODFLOW = 'mf2005'
        self.versionMODPATH = 'mp6'

    def _setup_workspace(self):
        """Set workspace and temporary directory paths.

        Sets the workspace directory based on the current file location,
        handles special cases (e.g., running from a zipped library),
        and creates a temporary directory for model files.

        Also generates or truncates the model name as needed.

        Args:
            self (object): The instance of the class.

        """
        self.wrkspc = dirname(abspath(__file__))
        if 'library.zip' in self.wrkspc:
            self.wrkspc = dirname(dirname(self.wrkspc))

        self.tempDir = TemporaryDirectory(suffix=str(uuid4())[:5])
        self.modelpth = str(self.tempDir.name)

        # Generate or truncate MODELNAME if needed
        if not self.MODELNAME:
            self.MODELNAME = f'FloPyArcade{str(uuid4())[:5]}'
        if self.ENVTYPE == '0s-c':
            self.MODELNAMEGENCOUNT = self.MODELNAME
            self.MODELNAME = self.MODELNAME[:15]

    def _init_env_vars(self, seed):
        """Initialize runtime variables and flags.

        Args:
            self (object): The instance of the class.
            seed (int or None): Random seed for reproducibility.
        
        """
        self.info = {}
        self.comments = ''
        self.done = False
        self.closed = False
        self.max_episode_steps = 200
        self.actionType = self.getActionType(self.ENVTYPE)
        self._SEED = seed
        self.timeStep = 0
        self.keyPressed = None
        self.reward = 0.0
        self.rewardCurrent = 0.0
        self.delFiles = True
        self.canvasInitialized = False
        self.renderFlag = self.RENDER
        self.success = False
        self.current_time = 0

    def _init_env_specific(self):
        """Perform environment-specific initialization based on ENVTYPE.

        Calls the appropriate initialization method depending on the environment type.

        Args:
            self (object): The instance of the class.
        
        """
        if self.ENVTYPE == '0s-c':
            self._init_env_0s_c()
        else:
            self._init_env_other()

    def _init_env_0s_c(self):
        """Initialize environment specific to '0s-c' ENVTYPE.

        Sets up simulators, workspace, model workspace, and constructs the model.
        Adds Windows-specific library dependencies if needed.

        Args:
            self (object): The instance of the class.
        
        """
        self.initializeSimulators(PATHMF6DLL=self.PATHMF6DLL)
        self.defineEnvironment(self._SEED)
        self.wrkspc = dirname(abspath(__file__))
        self.exdir = self.MODELNAME[:15]
        self.model_ws = join(self.wrkspc, 'models', self.exdir)
        self.mf6dll = self.PATHMF6DLL
        self.name = self.exdir  # Already sliced MODELNAME
        self.simpath = None
        makedirs(self.model_ws, exist_ok=True)
        self.actionSpaceSize = self.getActionSpaceSize(self.actionsDict)
        if platformSystem() == 'Windows':
            win_lib_path = join(self.wrkspc, 'simulators', 'win64', 'win-builds', 'bin')
            self.add_lib_dependencies([win_lib_path])
        self.constructModel()

    def _init_env_other(self):
        """Initialize environment for ENVTYPEs other than '0s-c'.

        Sets up simulators, environment, actions, particles, wells, and trajectories.

        Args:
            self (object): The instance of the class.
        
        """
        self.initializeSimulators(self.PATHMF2005, self.PATHMP6)
        self.defineEnvironment(self._SEED)
        if self.ENVTYPE in {
            '1s-d', '1s-c', '1r-d', '1r-c', '2s-d', '2s-c', '2r-d', '2r-c'
        }:
            self.initializeAction()
        self.initializeParticle()
        self.particleCoords[0] = self.extentX - self.particleCoords[0]
        self.update_head_specs_if_needed()
        self.initializeModel()
        self.wellX, self.wellY, self.wellZ, self.wellCoords, self.wellQ = \
            self.initializeWellRate(self.minQ, self.maxQ)
        self.initialize_helper_wells()
        self.initializeWell()
        if self.ENVTYPE.startswith(('3s', '3r', '4s', '4r', '5s', '5r', '6s', '6r')):
            self.initializeAction()
        self.trajectories = {axis: [] for axis in ('x', 'y', 'z')}

    def _setup_gym_spaces(self):
        """Set up Gym observation and action spaces based on environment configuration.

        Defines observation space as a Box with fixed bounds.
        Defines action space as Discrete or continuous Box depending on actionType.

        Args:
            self (object): The instance of the class.
        
        """
        import gymnasium

        obs_array = np.array(self.observations)

        self.observation_space = gymnasium.spaces.Box(
        # self.observation_space = gym.spaces.Box(
            low=-25.0,
            high=25.0,
            shape=obs_array.shape,
            dtype=np.float32
        )

        if self.actionType == 'discrete':
            self.action_space = gymnasium.spaces.Discrete(len(self.actionSpace))
            # self.action_space = gym.spaces.Discrete(len(self.actionSpace))
        elif self.actionType == 'continuous':
            if self.ENVTYPE == '3s-c':
                self.actionSpaceSize = 4
                self.distanceMax = 98.0
                self.rewardMax = 1000.0
            self.action_space = gymnasium.spaces.Box(
            # self.action_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.actionSpaceSize,),
                dtype=np.float32
            )

    def seed(self, seed=None):
        """Set the random seed for reproducibility.

        Sets the seed for NumPy's random number generator. Extend this method
        to seed other random generators as needed.

        Args:
            self (object): The instance of the class.
            seed (int, optional): The seed value. Defaults to None.
        
        """
        np.random.seed(seed)
        # Example: to seed other random generators, add here
        # import random
        # random.seed(seed)

    def close(self):
        """Mark the resource as closed.

        Sets the `closed` attribute to True to indicate that the resource has been closed.

        Args:
            self (object): The instance of the class.
        
        """
        self.closed = True

    def teardown(self):
        """Clean up the workspace by removing the example folder if deletion is enabled.

        Changes the current working directory to the workspace, finalizes the model,
        and removes the model workspace directory if `delFiles` is True. If `delFiles`
        is False, files are retained.

        Args:
            self (object): The instance of the class.

        Raises:
            AssertionError: If the model workspace directory removal fails.
        
        """
        self._change_to_workspace()
        if not self.delFiles:
            print("Retaining files")
            return

        self._finalize_model()
        self._remove_model_workspace()
        assert self.teardownSuccess, f"Failed to remove directory {self.model_ws}"

    def _change_to_workspace(self):
        """Change the current working directory to the workspace directory.

        Args:
            self (object): The instance of the class.
        
        """
        chdir(self.wrkspc)

    def _finalize_model(self):
        """Finalize the model instance, ignoring any exceptions.

        Calls the `finalize` method on the model (`mf6`) to clean up resources.
        Any exceptions during finalization are caught and ignored.

        Args:
            self (object): The instance of the class.
        
        """
        try:
            self.mf6.finalize()
        except Exception:
            pass  # Ignore finalize errors

    def _remove_model_workspace(self):
        """Remove the model workspace directory if it exists.

        Attempts to delete the directory specified by `model_ws`.
        Sets `teardownSuccess` to True if successful, otherwise False.

        Args:
            self (object): The instance of the class.
        
        """
        if not exists(self.model_ws):
            self.teardownSuccess = True
            return

        try:
            rmtree(self.model_ws)
            self.teardownSuccess = True
        except Exception as e:
            print(f"Could not remove test {self.model_ws}\nError: {e}")
            self.teardownSuccess = False

    def bmi_return(self, success, model_ws, modelname):
        """Read the standard output file of a MODFLOW 6 simulation.

        Attempts to open and read the contents of the simulation's stdout file
        located in the specified workspace directory. The filename is constructed
        as 'mfsim.stdout' within the model workspace directory.

        Args:
            self (object): The instance of the class.
            success (bool): Indicator of whether the model run was successful.
            model_ws (str): Path to the model workspace directory.
            modelname (str): Name of the model (not used in current filename).

        Returns:
            tuple: A tuple containing:
                - success (bool): The input success flag.
                - lines (list of str): Lines read from the stdout file.

        Raises:
            FileNotFoundError: If the stdout file does not exist.
            IOError: If there is an error reading the file.
        
        """
        fpth = join(model_ws, 'mfsim.stdout')
        try:
            with open(fpth, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            # Optionally, handle or log the error here
            raise e

        return success, lines

    def example_feature_extractor(self, obs):
        """Extract features by flattening and normalizing an observation.

        Flattens the input observation array and normalizes it by its Euclidean norm.
        If the norm is zero, returns the flattened observation unchanged.

        Args:
            self (object): The instance of the class.
            obs (numpy.ndarray): The observation array to be processed.

        Returns:
            numpy.ndarray: The normalized flattened observation.
        
        """
        flat = obs.flatten()
        norm = np.linalg.norm(flat)
        if norm > 0:
            return flat / norm
        return flat

    def create_single_hash(self, observations):
        """Create a single SHA-256 hash from a list of observations.

        Converts each observation to a feature vector, flattens all features into
        a single array, and computes a SHA-256 hash of the resulting bytes.

        Args:
            self (object): The instance of the class.
            observations (list): List of raw observations (e.g., numpy arrays).

        Returns:
            str: SHA-256 hash string representing the combined observations.
        
        """
        features = np.array([self.example_feature_extractor(obs) for obs in observations])
        feature_bytes = features.ravel().tobytes()
        return hashlib.sha256(feature_bytes).hexdigest()

    def defineEnvironment(self, seed=None):
        """Set up environmental variables and model configuration.

        Initializes random seed, environment type, action space, and model parameters.

        Args:
            self (object): The instance of the class.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            None
        
        """
        np_seed = getattr(self, '_SEED', seed)
        numpySeed(np_seed)
        if self.ENVTYPE == '0s-c':
            self._setup_0sc()
        else:
            self._setup_general_env()
        self._setup_action_space()
        self.rewardMax = 1000
        self.distanceMax = 98
        self.nouter = self.ninner = 100
        self.hclose = self.rclose = 1e-8
        self.relax = 1.0

    def _setup_0sc(self):
        """Set up 0s-c environment configuration.

        Initializes grid, well, storm, and recharge parameters for 0s-c type.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        self.perlen, self.nper, self.nstp = 50, 1, 50
        self.maxSteps = self.nstp
        self.tdis_rc = [(self.perlen, self.nstp, 1)]
        self.nlay, self.nrow, self.ncol = 1, 25, 25
        self.delr = self.delc = 100.0
        self.extentX = self.ncol * self.delr
        self.extentY = self.nrow * self.delc
        self.domainArea = self.delr * self.delc
        self.top, self.botm, self.hk = 50.0, 0.0, 0.05
        self.strt, self.chd_west, self.chd_east = 58.0, 58.5, 57.5
        self.chd_diff = numpyAbs(self.chd_west - self.chd_east)
        self.nHelperWells, self.nStorms = 5, 2
        self.actionSpaceIndividual = ['up', 'down', 'left', 'right', 'Qup', 'Qdown']
        self.actionSpace = [
            f"{a}{i+1}" for i in range(self.nHelperWells)
            for a in self.actionSpaceIndividual
        ]
        self.actionsDict = {}
        self.well_dxMax = 2 * self.delr
        self.well_dyMax = 2 * self.delc
        self.minQ, self.maxQ = -15.0, 15.0
        self.diffQ = numpyAbs(self.minQ - self.maxQ)
        self.well_dQMax = 5.0
        self._init_helper_wells()
        self.avg_rch_min = self.avg_rch_max = 0.0
        self.storm_rch_min, self.storm_rch_max = 1e-1, 20.0
        self.rch_diff = numpyAbs(self.avg_rch_min - self.storm_rch_max)
        self.minStormDuration, self.maxStormDuration = 3, 20
        self.minStormRadius, self.maxStormRadius = 300, 1000
        self._init_storms()
        self.shapeLayer = (1, self.nrow, self.ncol)
        self.shapeGrid = (self.nlay, self.nrow, self.ncol)
        self.rech0 = uniform(self.avg_rch_min, self.avg_rch_max, size=self.shapeLayer)
        self._init_recharge_time_series()

    def _init_helper_wells(self):
        """Initialize helper wells' positions and attributes.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        for i in range(self.nHelperWells):
            x = uniform(1.1 * self.delr, self.extentX - 1.1 * self.delr)
            y = uniform(1.1 * self.delc, self.extentY - 1.1 * self.delc)
            cellID = self.get_cellID(x, y)
            iRow, iCol = self.get_cellColRow(x, y)
            prefix = f"well{i}"
            self.actionsDict.update({
                f"{prefix}x": x,
                f"{prefix}y": y,
                f"{prefix}cellID": cellID,
                f"{prefix}iRow": iRow,
                f"{prefix}iCol": iCol,
                f"{prefix}Q": 0.0,
            })

    def _init_storms(self):
        """Randomly initialize storm event parameters.

        Sets start time, duration, intensity, center coordinates, and radius
        for each storm event.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        self.stormCentersX, self.stormCentersY = [], []
        self.stormStarts, self.stormDurations = [], []
        self.stormIntensities, self.stormRadii = [], []

        for _ in range(self.nStorms):
            start = int(uniform(2, self.nstp - 1))
            duration = int(uniform(self.minStormDuration, self.maxStormDuration))
            intensity = uniform(self.storm_rch_min, self.storm_rch_max)
            centerX = uniform(0, self.extentX)
            centerY = uniform(0, self.extentY)
            radius = uniform(self.minStormRadius, self.maxStormRadius)

            self.stormStarts.append(start)
            self.stormDurations.append(duration)
            self.stormIntensities.append(intensity)
            self.stormCentersX.append(centerX)
            self.stormCentersY.append(centerY)
            self.stormRadii.append(radius)

    def _init_recharge_time_series(self):
        """Create recharge time series including storm effects.

        Updates recharge values over time steps accounting for active storms.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        self.rechTimeSeries = [self.rech0]
        activeStormsIdx = []

        for iStp in range(self.nstp):
            if iStp + 1 in self.stormStarts:
                activeStormsIdx.append(self.stormStarts.index(iStp + 1))

            rech = copy(self.rech0)
            stormDurationsTemp = copy(self.stormDurations)

            for iStorm in activeStormsIdx:
                if stormDurationsTemp[iStorm] > 0:
                    ID = self.get_cellID(
                        self.stormCentersX[iStorm], self.stormCentersY[iStorm]
                    ) - 1
                    rech_flat = rech.flatten()
                    rech_flat[ID] = self.stormIntensities[iStorm]
                    rech = rech_flat.reshape(self.shapeLayer)
                stormDurationsTemp[iStorm] -= 1

            self.rechTimeSeries.append(rech)

    def _setup_general_env(self):
        """Set up general (non-0s-c) environment configuration.

        Initializes spatial extents, layering, head specifications, well parameters,
        and observation settings.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        self.minX = self.minY = 0.0
        self.extentX = self.extentY = 100.0
        self.zBot, self.zTop = 0.0, 50.0
        self.headSpecWest, self.headSpecEast = 60.0, 56.0
        self.minQ, self.maxQ = -2000.0, -500.0
        self.wellSpawnBufferXWest, self.wellSpawnBufferXEast = 50.0, 20.0
        self.wellSpawnBufferY = 20.0
        self.periods, self.periodLength, self.periodSteps = 1, 1.0, 11
        self.periodSteadiness = True
        self.maxSteps = self.NAGENTSTEPS
        self.sampleHeadsEvery = 3
        self.dRow = self.extentX / self.nCol
        self.dCol = self.extentY / self.nRow
        self.dVer = (self.zTop - self.zBot) / self.nLay
        self.botM = linspace(self.zTop, self.zBot, self.nLay + 1)
        self.wellRadius = sqrt(8)
        self.actionSpaceIndividual = ['up', 'down', 'left', 'right', 'Qup', 'Qdown']
        self._setup_envtype_specifics()

    def _setup_envtype_specifics(self):
        """Set environment-specific parameters based on ENVTYPE.

        Configures head limits, helper wells, penalty factors, and costs
        depending on the environment type.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        t = self.ENVTYPE
        if t in {'1s-d', '1s-c', '1r-d', '1r-c'}:
            self.minH, self.maxH, self.nHelperWells = 56.0, 60.0, 0
            self.deviationPenaltyFactor, self.actionRange = 10.0, 0.5
        elif t in {'2s-d', '2s-c', '2r-d', '2r-c'}:
            self.minH, self.maxH, self.nHelperWells = 56.0, 62.0, 0
            self.deviationPenaltyFactor, self.actionRange = 10.0, 0.5
        elif t in {'3s-d', '3s-c', '3r-d', '3r-c'}:
            self.minH, self.maxH, self.nHelperWells = 56.0, 60.0, 0
            self.deviationPenaltyFactor, self.actionRange = 10.0, 10.0
        elif t in {
            '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost',
            '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c'
        }:
            self.helperWellRadius = self.wellRadius / 4
            self.minH, self.maxH, self.nHelperWells = 56.0, 60.0, 20
            self.minQhelper, self.maxQhelper = -600.0, 600.0
            self.deviationPenaltyFactor = 10.0
            self.costQm3dPump, self.costQm3dRecharge = 0.48, 0.70
            self.costQFail = -2 * self.maxSteps * self.nHelperWells * \
                self.maxQhelper * self.costQm3dRecharge
        if t in {
            '1r-d', '1r-c', '2r-d', '2r-c', '3r-d', '3r-c', '4r-d', '4r-c',
            '5r-d', '5r-c', '6r-d', '6r-c'
        }:
            self.maxHChange, self.maxQChange, self.maxCoordChange = 0.2, 50.0, 2.5

    def _setup_action_space(self):
        """Configure the action space based on environment and action type.

        Sets `actionSpace` and `actionSpaceSize` according to `ENVTYPE` and `actionType`.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        t, a = self.ENVTYPE, self.actionType
        if a == 'discrete':
            if t in {'1s-d', '1r-d', '2s-d', '2r-d'}:
                self.actionSpace = ['down', 'up', 'keep']
            elif t in {'3s-d', '3r-d'}:
                self.actionSpace = ['right', 'up', 'keep', 'down', 'left']
            elif t in {'4s-d', '4r-d', '5s-d', '5r-d', '6s-d', '6r-d'}:
                self._setup_discrete_multiwell_actions()
        elif a == 'continuous':
            if t in {'1s-c', '1r-c', '2s-c', '2r-c'}:
                self.actionSpace = ['up', 'down']
            elif t in {'3s-c', '3r-c'}:
                self.actionSpace = ['up', 'down', 'left', 'right']
            elif t in {'4s-c', '4r-c', '5s-c', '5s-c-cost', '5r-c', '6s-c', '6r-c'}:
                self._setup_continuous_multiwell_actions()

        self.actionSpaceSize = len(
            getattr(self, 'actionSpace', self.actionSpaceIndividual)
        )

    def _setup_discrete_multiwell_actions(self):
        """Set up discrete action space for multiwell environments.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        t = self.ENVTYPE
        if t in {'4s-d', '4r-d'}:
            self.actionSpaceIndividual, self.actionRange = (
                ['up', 'keep', 'down', 'left', 'right'], 2.5)
        elif t in {'5s-d', '5r-d'}:
            self.actionSpaceIndividual, self.actionRangeQ = (
                ['moreQ', 'keepQ', 'lessQ'], 100.0)
        elif t in {'6s-d', '6r-d'}:
            self.actionSpaceIndividual, self.actionRange, self.actionRangeQ = (
                ['up', 'keep', 'down', 'left', 'right', 'moreQ', 'keepQ', 'lessQ'],
                2.5, 100.0)

        self.actionSpace = [
            ''.join(comb) for comb in product(self.actionSpaceIndividual, repeat=self.nHelperWells)
        ]

    def _setup_continuous_multiwell_actions(self):
        """Set up continuous action space for multiwell environments.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        t = self.ENVTYPE
        if t in {'4s-c', '4r-c'}:
            self.actionSpaceIndividual, self.actionRange = (
                ['up', 'down', 'left', 'right'], 2.5)
        elif t in {'5s-c', '5s-c-cost', '5r-c'}:
            self.actionSpaceIndividual, self.actionRangeQ = (
                ['moreQ', 'lessQ'], 100.0)
        elif t in {'6s-c', '6r-c'}:
            self.actionSpaceIndividual, self.actionRange, self.actionRangeQ = (
                ['up', 'down', 'left', 'right', 'moreQ', 'lessQ'], 2.5, 50.0)

        self.actionSpace = [
            f"{a}{i+1}"
            for i in range(self.nHelperWells)
            for a in self.actionSpaceIndividual
        ]

    def update_head_specs_if_needed(self):
        """Update head specifications if ENVTYPE requires it.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        prefixes = ('3s', '3r', '4s', '4r', '5s', '5r', '6s', '6r')
        if self.ENVTYPE.startswith(prefixes):
            self.headSpecNorth = uniform(self.minH, self.maxH)
            self.headSpecSouth = uniform(self.minH, self.maxH)

    def initialize_helper_wells(self):
        """Initialize helper wells if ENVTYPE requires it.

        Initializes dictionary entries for helper wells with values from `initializeWellRate`.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        prefixes = ('4s', '4r', '5s', '5r', '6s', '6r')
        if not self.ENVTYPE.startswith(prefixes):
            return

        self.helperWells = {}
        for i in range(self.nHelperWells):
            w = str(i + 1)
            self.helperWells.update({
                f'wellX{w}': None,
                f'wellY{w}': None,
                f'wellZ{w}': None,
                f'wellCoords{w}': None,
                f'wellQ{w}': None,
            })
            vals = self.initializeWellRate(self.minQhelper, self.maxQhelper)
            (self.helperWells[f'wellX{w}'], self.helperWells[f'wellY{w}'],
             self.helperWells[f'wellZ{w}'], self.helperWells[f'wellCoords{w}'],
             self.helperWells[f'wellQ{w}']) = vals

    def initializeSimulators(self, PATHMF2005=None, PATHMP6=None, PATHMF6DLL=None):
        """Initialize simulators depending on the operating system.

        Executables must be specified or located in the simulators subfolder.

        Args:
            self (object): The instance of the class.
            PATHMF2005 (str, optional): Path to MODFLOW-2005 executable.
            PATHMP6 (str, optional): Path to MODPATH6 executable.
            PATHMF6DLL (str, optional): Path to MODFLOW 6 DLL.

        Returns:
            None
        
        """
        subfolder, ext_exe, ext_dll, chmod_required = self._get_platform_info()
        if subfolder is None:
            print('Operating system is unknown.')
            return

        self.exe_name = self._get_executable_path(PATHMF2005, subfolder, 'mf2005', ext_exe)
        self.exe_mp = self._get_executable_path(PATHMP6, subfolder, 'mp6', ext_exe)
        self.PATHMF6DLL = self._get_executable_path(PATHMF6DLL, subfolder, 'libmf6', ext_dll)

        if chmod_required:
            self._set_executable_permissions(self.exe_name, self.exe_mp, self.PATHMF6DLL)

    def _get_platform_info(self):
        """Detect platform details: subfolder, file extensions, and chmod requirement.

        Args:
            self (object): The instance of the class.

        Returns:
            tuple: (subfolder (str), ext_exe (str), ext_dll (str), chmod_required (bool))
                or (None, None, None, None) if unknown platform.
        
        """
        bits = 64 if sizeof(c_voidp) == 8 else 32
        platform_name = platformSystem()

        if platform_name == 'Windows':
            subfolder = 'win64' if bits == 64 else 'win32'
            ext_exe, ext_dll, chmod_required = '.exe', '.dll', False
        elif platform_name == 'Linux':
            subfolder, ext_exe, ext_dll, chmod_required = 'linux', '', '.so', True
        elif platform_name == 'Darwin':
            subfolder, ext_exe, ext_dll, chmod_required = 'mac', '', '.so', True
        else:
            return None, None, None, None

        return subfolder, ext_exe, ext_dll, chmod_required

    def _get_executable_path(self, override_path, subfolder, base_name, extension):
        """Return executable or DLL path, using override if provided.

        Args:
            self (object): The instance of the class.
            override_path (str or None): User-specified path.
            subfolder (str): Platform-specific subfolder.
            base_name (str): Base executable or DLL name.
            extension (str): File extension.

        Returns:
            str: Full path to the executable or DLL.
        
        """
        if override_path:
            return override_path
        return join(self.wrkspc, 'simulators', subfolder, base_name) + extension

    def _set_executable_permissions(self, *paths):
        """Set executable permissions (chmod 775) on given paths.

        Args:
            self (object): The instance of the class.
            *paths (str): Paths to set permissions on.

        Returns:
            None
        
        """
        for path in paths:
            chmod(path, 0o775)

    def initializeAction(self):
        """Initialize actions randomly based on ENVTYPE.

        Calls the appropriate initialization method for the ENVTYPE group.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        t = self.ENVTYPE
        if t in {'1s-d', '1s-c', '1r-d', '1r-c'}:
            self._init_type_1()
        elif t in {'2s-d', '2s-c', '2r-d', '2r-c'}:
            self._init_type_2()
        elif t in {'3s-d', '3s-c', '3r-d', '3r-c'}:
            self._init_type_3()
        elif t in {'4s-d', '4s-c', '4r-d', '4r-c'}:
            self._init_type_4()
        elif t in {'5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c'}:
            self._init_type_5()
        elif t in {'6s-d', '6s-c', '6r-d', '6r-c'}:
            self._init_type_6()

    def _init_type_1(self):
        """Initialize for ENVTYPE group 1.

        Sets action values for North and South within the range [minH, maxH].

        Args:
            self (object): The instance of the class.
        
        """
        self.actionValueNorth = uniform(self.minH, self.maxH)
        self.actionValueSouth = uniform(self.minH, self.maxH)

    def _init_type_2(self):
        """Initialize for ENVTYPE group 2.

        Sets a single action value within the range [minH, maxH].

        Args:
            self (object): The instance of the class.
        
        """
        self.actionValue = uniform(self.minH, self.maxH)

    def _init_type_3(self):
        """Initialize for ENVTYPE group 3.

        Sets action values X and Y to well coordinates.

        Args:
            self (object): The instance of the class.
        
        """
        self.actionValueX = self.wellX
        self.actionValueY = self.wellY

    def _init_type_4(self):
        """Initialize for ENVTYPE group 4.

        Copies X and Y well coordinates to corresponding action values for helper wells.

        Args:
            self (object): The instance of the class.
        
        """
        for i in range(1, self.nHelperWells + 1):
            self.helperWells[f'actionValueX{i}'] = self.helperWells[f'wellX{i}']
            self.helperWells[f'actionValueY{i}'] = self.helperWells[f'wellY{i}']

    def _init_type_5(self):
        """Initialize for ENVTYPE group 5.

        Copies well flow rates to corresponding action values for helper wells.

        Args:
            self (object): The instance of the class.
        
        """
        for i in range(1, self.nHelperWells + 1):
            self.helperWells[f'actionValueQ{i}'] = self.helperWells[f'wellQ{i}']

    def _init_type_6(self):
        """Initialize for ENVTYPE group 6.

        Copies X, Y coordinates and flow rates to action values for helper wells.

        Args:
            self (object): The instance of the class.
        
        """
        for i in range(1, self.nHelperWells + 1):
            self.helperWells[f'actionValueX{i}'] = self.helperWells[f'wellX{i}']
            self.helperWells[f'actionValueY{i}'] = self.helperWells[f'wellY{i}']
            self.helperWells[f'actionValueQ{i}'] = self.helperWells[f'wellQ{i}']

    def initializeParticle(self):
        """Initialize particle spawn position randomly along western border.

        Places the particle just east of the western stream with vertical buffer.

        Args:
            self (object): The instance of the class.

        Updates:
            particleSpawnBufferY (float): Vertical buffer from boundaries.
            particleX (float): X-coordinate of spawn.
            particleY (float): Y-coordinate of spawn.
            particleZ (float): Z-coordinate of spawn (top boundary).
            particleCoords (list of float): 3D coordinates of the particle.

        Assumes:
            self.extentX, self.extentY define domain size.
            self.dCol relates to grid spacing.
            self.zTop is top boundary in Z direction.

        Raises:
            AssertionError: If domain extents or parameters are invalid.
        
        """
        assert self.extentX > 0, "extentX must be positive"
        assert self.extentY > 40.0, "extentY must exceed twice the spawn buffer"
        assert self.dCol > 0, "dCol must be positive"
        assert self.zTop is not None, "zTop must be set"

        self.particleSpawnBufferY = 20.0
        self.particleX = self.extentX - 1.1 * self.dCol
        ymin, ymax = self.particleSpawnBufferY, self.extentY - self.particleSpawnBufferY
        self.particleY = uniform(ymin, ymax)
        self.particleZ = self.zTop
        self.particleCoords = [self.particleX, self.particleY, self.particleZ]

    def initializeModel(self):
        """Initialize groundwater flow model by constructing its components.

        Args:
            self (object): The instance of the class.

        Calls `constructModel` to build and prepare the model for simulation.

        Assumes:
            `constructModel` initializes all necessary model attributes.
        
        """
        self.constructModel()

    def initializeWellRate(self, minQ, maxQ):
        """Initialize a well randomly within aquifer margins with random flow rate.

        The well is placed within buffered domain boundaries to avoid edges.

        Args:
            self (object): The instance of the class.
            minQ (float): Minimum well flow rate.
            maxQ (float): Maximum well flow rate.

        Returns:
            tuple: (wellX, wellY, wellZ, wellCoords, wellQ)
                wellX (float): X-coordinate.
                wellY (float): Y-coordinate.
                wellZ (float): Z-coordinate (top boundary).
                wellCoords (list): 3D coordinates.
                wellQ (float): Flow rate.

        Raises:
            AssertionError: If domain extents or buffers are invalid, or minQ > maxQ.
        
        """
        assert minQ <= maxQ, "minQ must not exceed maxQ"
        xmin = self.wellSpawnBufferXWest
        xmax = self.extentX - self.wellSpawnBufferXEast
        ymin = self.wellSpawnBufferY
        ymax = self.extentY - self.wellSpawnBufferY

        wellX = uniform(xmin, xmax)
        wellY = uniform(ymin, ymax)
        wellZ = self.zTop
        wellCoords = [wellX, wellY, wellZ]
        wellQ = uniform(minQ, maxQ)

        return wellX, wellY, wellZ, wellCoords, wellQ

    def initializeWell(self):
        """Initialize the well feature in the groundwater flow model.

        Validates main well attributes, sets cell indices, initializes helper wells
        if present, and creates the well package.

        Args:
            self (object): The instance of the class.
        
        """
        self._validateMainWellAttributes()
        self._setMainWellCellIndices()
        if self._envHasHelperWells():
            self._initializeHelperWells()
        self._createWellPackage()

    def _validateMainWellAttributes(self):
        """Validate that main well coordinates and flow rate are set.

        Args:
            self (object): The instance of the class.

        Raises:
            AssertionError: If any required attribute is missing.
        
        """
        required = ['wellX', 'wellY', 'wellZ', 'wellQ']
        missing = [attr for attr in required if not hasattr(self, attr)]
        assert not missing, f"Missing well attributes: {', '.join(missing)}"

    def _setMainWellCellIndices(self):
        """Set the main well's cell indices based on its coordinates.

        Converts the main well's (X, Y, Z) coordinates into corresponding
        cell indices: layer, column, and row, and assigns them to the
        object's attributes.

        This method updates:
            - self.wellCellLayer
            - self.wellCellColumn
            - self.wellCellRow

        Args:
            self (object): The instance of the class.
        """
        l, c, r = self.cellInfoFromCoordinates([self.wellX, self.wellY, self.wellZ])
        self.wellCellLayer, self.wellCellColumn, self.wellCellRow = l, c, r

    def _envHasHelperWells(self):
        """Check if the environment type includes helper wells.

        Args:
            self (object): The instance of the class.

        Returns:
            bool: True if helper wells are present, False otherwise.
        
        """
        envtypes = {
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        }
        return self.ENVTYPE in envtypes

    def _initializeHelperWells(self):
        """Initialize helper wells' cell indices from their coordinates.

        Args:
            self (object): The instance of the class.

        Raises:
            AssertionError: If any helper well coordinate is missing.
        
        """
        for i in range(1, self.nHelperWells + 1):
            hx = self.helperWells.get(f'wellX{i}')
            hy = self.helperWells.get(f'wellY{i}')
            hz = self.helperWells.get(f'wellZ{i}')
            assert None not in (hx, hy, hz), f"Helper well {i} coordinates missing"
            hl, hc, hr = self.cellInfoFromCoordinates([hx, hy, hz])
            self.helperWells.update({f'l{i}': hl, f'c{i}': hc, f'r{i}': hr})

    def _createWellPackage(self):
        """Create the MODFLOW well package with main and helper wells.
        Converts cell indices to zero-based and arranges stress period data.

        Args:
            self (object): The instance of the class.

        """
        l = self.wellCellLayer - 1
        c = self.wellCellColumn - 1
        r = self.wellCellRow - 1
        main_well = [l, r, c, self.wellQ]

        if self._envHasHelperWells():
            wells = [main_well]
            for i in range(self.nHelperWells):
                idx = str(i + 1)
                hl = self.helperWells[f'l{idx}'] - 1
                hc = self.helperWells[f'c{idx}'] - 1
                hr = self.helperWells[f'r{idx}'] - 1
                hq = self.helperWells[f'wellQ{idx}']
                wells.append([hl, hr, hc, hq])
            stress_data = {0: wells}
        else:
            stress_data = {0: [main_well]}

        self.wel = ModflowWel(self.mf, stress_period_data=stress_data)

    def initializeState(self, state):
        """
        Initialize the aquifer hydraulic head using the state from a previous time step.

        This method copies the 'heads' data from the provided state dictionary and
        stores it in the instance attribute `headsPrev` to represent the previous
        hydraulic head distribution.

        Args:
            self (object): The instance of the class.
            state (dict): A dictionary containing the simulation state, expected to
                have a 'heads' key.

        Raises:
            AssertionError: If 'heads' key is missing in the state dictionary.
        
        """
        assert 'heads' in state, "Input state must contain 'heads' key"
        self.headsPrev = copy(state['heads'])

    def step(self, action, teardownOnFinish=True):
        """
        Perform a single simulation step, handling initialization inline on the 
        first step.

        Args:
            self (object): Instance of the class.
            action: Action to take.
            teardownOnFinish (bool): Whether to clean up after episode finishes.

        Returns:
            tuple: observation, reward, done, truncated, info
        
        """
        if self.actionType == 'discrete':
            if isinstance(action, int):
                action = self.actionSpace[action]
            elif action not in self.actionSpace:
                # Handle invalid action or convert accordingly
                pass

        reward = 0.0
        info = None

        if self.timeStep == 0 and self.initWithSolution:
            return self._first_step(reward, info)

        if self.ENVTYPE == '0s-c':
            return self._handle_0sc_environment(
                action, reward, info, teardownOnFinish
            )
        return self._handle_other_environments(
            action, reward, info, teardownOnFinish
        )

    def _first_step(self, reward, info):
        """
        Initialize environment at first time step.

        Args:
            self (object): Instance of the class.
            reward (float): Initial reward.
            info: Additional info.

        Returns:
            tuple: observations, reward, done, truncated, info
        
        """
        self.truncated = False
        self.done = False

        if self.ENVTYPE == '0s-c':
            self.setup_workspace_and_model()
            self.load_mf6_model(self.mf6dll, self.name)
            self.run_steady_state_solution(self.mf6dll, self.name)
            self.prepare_transient_simulation(self.mf6dll, self.name)
            new_recharge = self.retrieve_boundary_conditions()
            self.prepare_observations(new_recharge)
            self.observationsVectorNormalized = copy(self.observations)
            self.observations_history = [self.observationsVectorNormalized]
            if self.renderFlag or self.SAVEPLOT:
                self.render()
        else:
            self.runMODFLOW()
            self.update_state()
            self.get_normalized_observations()
            self.get_normalized_stresses()
            self.observations = np.array(self.observationsVectorNormalized)
            self.observations_history = [self.observations]
            if self.renderFlag or self.SAVEPLOT:
                self.render()

        self.timeStep += 1
        return self.observationsVectorNormalized, reward, self.done, self.truncated, info

    def _handle_0sc_environment(self, action, reward, info, teardownOnFinish):
        """Handle the '0s-c' environment type.

        Args:
            self (object): The instance of the class.
            action (Any): The action to apply in the environment.
            reward (float): The reward from the previous step.
            info (dict): Additional info dictionary.
            teardownOnFinish (bool): Flag to teardown environment after finish.

        Returns:
            tuple: observations vector normalized, reward, done flag, truncated flag,
                and info dictionary.
        
        """
        if self.timeStep == self.nstp:
            self.done = True

        self.prepare_time_step()
        self.get_simulated_heads()
        new_recharge = self.get_recharge()
        self.update_well_cells()

        self.actionsDict = self.changeActionDict(self.actionsDict, action)
        self.actionsDict = self.getActionValues(self.actionsDict)
        self.update_well_Q()
        self.mf6.set_value(self.well_tag, self.well)

        new_recharge = self.rechTimeSeries[self.timeStep]
        self.mf6.set_value(self.rch_tag, new_recharge)

        if not self.run_convergence_loop():
            return self.handle_convergence_failure(self.name)

        self.reward = self.calculateGameRewardHeadChange(
            self.head_steadyState_flat, self.head)
        self.rewardCurrent += self.reward

        self.prepare_observations(new_recharge)
        self.observationsVectorNormalized = copy(self.observations)
        self.observations_history += self.observationsVectorNormalized

        if self.renderFlag or self.SAVEPLOT:
            self.render()

        self.timeStep += 1
        self.info = {
            'timestep': self.timeStep,
            'max_steps': self.maxSteps,
            'action': action
        }
        self.update_envhash()
        self.handle_done(teardownOnFinish)

        return (
            self.observationsVectorNormalized,
            self.reward,
            self.done,
            self.truncated,
            self.info,
        )

    def _handle_other_environments(self, action, reward, info, teardownOnFinish):
        """Handle environment types other than '0s-c'.

        Args:
            self (object): The instance of the class.
            action (Any): The action to apply in the environment.
            reward (float): The reward from the previous step.
            info (dict): Additional info dictionary.
            teardownOnFinish (bool): Flag to teardown environment after finish.

        Returns:
            tuple: observations array, reward, done flag, truncated flag, and info dict.
        
        """
        if self.timeStep == 0:
            # Render initial timestep if required (after initialization)
            if self.RENDER or self.MANUALCONTROL or self.SAVEPLOT:
                self.render()

        self.keyPressed = None
        self.periodSteadiness = False

        self.setActionValue(action)
        self.initializeState(self.state)
        self.updateModel()
        self.updateWellRate()
        self.updateWell()
        self.runMODFLOW()
        self.runMODPATH()
        self.evaluateParticleTracking()
        self.calculate_reward()

        self.update_state()
        self.get_normalized_observations()
        self.get_normalized_stresses()
        self.check_particle_state()

        self.rewardCurrent += self.reward

        if self.RENDER or self.MANUALCONTROL or self.SAVEPLOT:
            self.render()

        self.observations = np.array(self.observationsVectorNormalized)
        self.observations_history += self.observations

        self.timeStep += 1

        self.info = {
            'timestep': self.timeStep,
            'max_steps': self.maxSteps,
            'action': action,
        }

        self.update_envhash()

        if self.done and teardownOnFinish:
            self.tempDir.cleanup()

        return (
            self.observations,
            self.reward,
            self.done,
            self.truncated,
            self.info,
        )

    def get_well_cells(self):
        """Retrieve well cell IDs as a numpy array.

        Args:
            self (object): The instance of the class.

        Returns:
            numpy.ndarray: Array of well cell IDs as int32.
        
        """
        return np.array(
            [self.actionsDict[f'well{i}cellID'] for i in range(self.nHelperWells)],
            dtype=np.int32,
        )

    def get_wel_x(self):
        """Get the current WEL_0 NODELIST variable value.

        Args:
            self (object): The instance of the class.

        Returns:
            tuple: The tag and its corresponding value.
        
        """
        tag = self.mf6.get_var_address('NODELIST', self.nameUpper, 'WEL_0')
        return tag, self.mf6.get_value(tag)

    def update_wel_x(self, tag, new_values):
        """Set new values to the WEL_0 NODELIST variable.

        Args:
            self (object): The instance of the class.
            tag (Any): The variable tag/address to update.
            new_values (Any): The new values to set.
        
        """
        self.mf6.set_value(tag, new_values)

    def update_well_cells(self):
        """Update well cells in the model.

        Args:
            self (object): The instance of the class.
        
        """
        well_cells = self.get_well_cells()
        tag, wel_x = self.get_wel_x()
        # Replace wel_x values with well_cells, preserving shape
        new_values = np.zeros_like(wel_x) + well_cells
        self.update_wel_x(tag, new_values)

    def update_state(self):
        """Update the environment state based on the environment type.

        Args:
            self (object): The instance of the class.
        
        """
        self.state = {'heads': self.heads}

        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self._update_1()
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            self._update_2()
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c']:
            self._update_3()
        elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c']:
            self._update_helper_wells(['X', 'Y'])
        elif self.ENVTYPE in ['5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c']:
            self._update_helper_wells(['Q'])
        elif self.ENVTYPE in ['6s-d', '6s-c', '6r-d', '6r-c']:
            self._update_helper_wells(['X', 'Y', 'Q'])

    def _update_1(self):
        """Update state for environment types '1s-d', '1s-c', '1r-d', '1r-c'.

        Args:
            self (object): The instance of the class.
        
        """
        self.state['actionValueNorth'] = self.actionValueNorth
        self.state['actionValueSouth'] = self.actionValueSouth

    def _update_2(self):
        """Update state for environment types '2s-d', '2s-c', '2r-d', '2r-c'.

        Args:
            self (object): The instance of the class.
        
        """
        self.state['actionValue'] = self.actionValue

    def _update_3(self):
        """Update state for environment types '3s-d', '3s-c', '3r-d', '3r-c'.

        Args:
            self (object): The instance of the class.
        
        """
        self.state['actionValueX'] = self.actionValueX
        self.state['actionValueY'] = self.actionValueY

    def _update_helper_wells(self, keys):
        """Update state for helper wells with specified keys.

        Args:
            self (object): The instance of the class.
            keys (list of str): List of keys to update the state with.
        
        """
        for i in range(self.nHelperWells):
            w = str(i + 1)
            for key in keys:
                state_key = f'actionValue{key}{w}'
                self.state[state_key] = self.helperWells[state_key]

    def _init_observations(self):
        """Initialize observation dictionaries.

        Args:
            self (object): The instance of the class.
        
        """
        self.observations = {}
        self.observationsNormalized = {}
        self.observationsNormalizedHeads = {}

    def _set_basic_observations(self):
        """Set basic observation values in the observations dictionary.

        Args:
            self (object): The instance of the class.
        
        """
        self.observations['particleCoords'] = self.particleCoords
        self.observations['headsSampledField'] = self.heads[
            0::self.sampleHeadsEvery,
            0::self.sampleHeadsEvery,
            0::self.sampleHeadsEvery,
        ]
        self.observations['heads'] = self.heads

    def _set_heads_observations(self):
        """Set head observations based on OBSPREP and ENVTYPE.

        Args:
            self (object): The instance of the class.
        
        """
        if self.OBSPREP == 'perceptron':
            if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
                self.observations['heads'] = [
                    self.actionValueNorth,
                    self.actionValueSouth,
                ]
            elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
                self.observations['heads'] = [self.actionValue]
            elif self.ENVTYPE in [
                '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c',
                '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
                '6s-d', '6s-c', '6r-d', '6r-c'
            ]:
                self.observations['heads'] = [
                    self.headSpecNorth,
                    self.headSpecSouth,
                ]
            self.observations['heads'] += list(
                np.array(self.observations['headsSampledField']).flatten()
            )
        elif self.OBSPREP == 'convolution':
            self.observations['heads'] = self.heads

    def _set_well_observations(self):
        """Set well-related observations.

        Args:
            self (object): The instance of the class.
        
        """
        self.observations['wellQ'] = self.wellQ
        self.observations['wellCoords'] = self.wellCoords

        if self.ENVTYPE in [
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        ]:
            for i in range(self.nHelperWells):
                w = str(i + 1)
                self.observations[f'wellQ{w}'] = self.helperWells[f'wellQ{w}']
                self.observations[f'wellCoords{w}'] = self.helperWells[f'wellCoords{w}']

    def _normalize_heads(self, heads, minH, maxH):
        """Normalize heads values to a 0-1 range.

        Args:
            self (object): The instance of the class.
            heads (array-like): Array of head values to normalize.
            minH (float): Minimum head value for normalization.
            maxH (float): Maximum head value for normalization.

        Returns:
            numpy.ndarray: Normalized head values.
        
        """
        return np.divide(np.array(heads) - minH, maxH - minH)
    
    def _normalize_coords(self, coords, minX, extentX):
        """Normalize coordinate values based on minimum and extent.

        Args:
            self (object): The instance of the class.
            coords (array-like): Coordinates to normalize.
            minX (float): Minimum coordinate value.
            extentX (float): Extent (range) of the coordinates.

        Returns:
            numpy.ndarray: Normalized coordinates.
        
        """
        return np.divide(np.array(coords), abs(minX + extentX))

    def _normalize_wellQ(self, wellQ, minQ):
        """Normalize wellQ values by dividing by minQ.

        Args:
            self (object): The instance of the class.
            wellQ (array-like): Well flow rates to normalize.
            minQ (float): Minimum well flow rate for normalization.

        Returns:
            numpy.ndarray: Normalized well flow rates.
        
        """
        return np.divide(np.array(wellQ), minQ)

    def _normalize_observations(self):
        """Normalize observations and store them in normalized dictionaries.

        Args:
            self (object): The instance of the class.
        
        """
        self.observationsNormalized['particleCoords'] = self._normalize_coords(
            np.copy(self.particleCoords), self.minX, self.extentX)
        self.observationsNormalized['heads'] = self._normalize_heads(
            self.observations['heads'], self.minH, self.maxH)
        self.observationsNormalized['wellQ'] = self._normalize_wellQ(
            self.wellQ, self.minQ)
        self.observationsNormalized['wellCoords'] = self._normalize_coords(
            self.wellCoords, self.minX, self.extentX)

        if self.ENVTYPE in [
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        ]:
            for i in range(self.nHelperWells):
                w = str(i + 1)
                self.observationsNormalized[f'wellQ{w}'] = self._normalize_wellQ(
                    self.helperWells[f'wellQ{w}'], self.minQhelper)
                self.observationsNormalized[f'wellCoords{w}'] = self._normalize_coords(
                    self.helperWells[f'wellCoords{w}'], self.minX, self.extentX)

        self.observationsNormalizedHeads['heads'] = self._normalize_heads(
            self.heads, self.minH, self.maxH)

    def _vectorize_observations(self):
        """Convert observation dictionaries to vector form based on OBSPREP.

        Args:
            self (object): The instance of the class.
        
        """
        if self.OBSPREP == 'perceptron':
            self.observationsVector = self.observationsDictToVector(self.observations)
            self.observationsVectorNormalized = self.observationsDictToVector(
                self.observationsNormalized)
            self.observationsVectorNormalizedHeads = self.observationsDictToVector(
                self.observationsNormalizedHeads)
        elif self.OBSPREP == 'convolution':
            self.observationsVector = self.observationsDictToVector(self.observations)
            self.observationsVectorNormalized = self.observationsDictToVector(
                self.observationsNormalized)

    def get_normalized_observations(self):
        """Initialize, set, normalize, and vectorize observations.

        Args:
            self (object): The instance of the class.
        
        """
        self._init_observations()
        self._set_basic_observations()
        self._set_heads_observations()
        self._set_well_observations()
        self._normalize_observations()
        self._vectorize_observations()

    def _normalize(self, val, vmin, vmax):
        """Normalize a value or array to the range [0, 1].

        Args:
            self (object): The instance of the class.
            val (float or array-like): Value(s) to normalize.
            vmin (float): Minimum value of the range.
            vmax (float): Maximum value of the range.

        Returns:
            float or numpy.ndarray: Normalized value(s) in [0, 1].
        
        """
        return (val - vmin) / (vmax - vmin)

    def _normalize_well(self, Q, X, Y, Z):
        """Normalize well parameters Q, X, Y, and Z.

        Args:
            self (object): The instance of the class.
            Q (float or array-like): Well flow rate(s).
            X (float or array-like): X coordinate(s).
            Y (float or array-like): Y coordinate(s).
            Z (float or array-like): Z coordinate(s).

        Returns:
            list: Normalized [Q, X, Y, Z] values.
        
        """
        return [
            Q / self.minQ,
            X / (self.minX + self.extentX),
            Y / (self.minY + self.extentY),
            Z / (self.zBot + self.zTop),
        ]

    def _normalize_basic_stresses(self, vals):
        """Normalize a list of stress values using minH and maxH.

        Args:
            self (object): The instance of the class.
            vals (list or array-like): Stress values to normalize.

        Returns:
            list: Normalized stress values.
        
        """
        return [self._normalize(v, self.minH, self.maxH) for v in vals]

    def _normalize_helper_wells(self):
        """Normalize parameters for all helper wells and return as a flat list.

        Args:
            self (object): The instance of the class.

        Returns:
            list: Normalized values of wellQ, wellX, wellY, wellZ for all helper wells.
        
        """
        stresses = []
        for i in range(self.nHelperWells):
            idx = str(i + 1)
            stresses.extend(self._normalize_well(
                self.helperWells[f'wellQ{idx}'],
                self.helperWells[f'wellX{idx}'],
                self.helperWells[f'wellY{idx}'],
                self.helperWells[f'wellZ{idx}'],
            ))
        return stresses

    def get_normalized_stresses(self):
        """Get normalized stress and well parameters based on environment type.

        Args:
            self (object): The instance of the class.

        Returns:
            list: Combined list of normalized basic stresses and well parameters.
                  Returns empty list if ENVTYPE is unrecognized.

        """
        env = self.ENVTYPE

        if env in ['1s-d', '1s-c', '1r-d', '1r-c']:
            base = self._normalize_basic_stresses([self.actionValueSouth, self.actionValueNorth])
            wells = self._normalize_well(self.wellQ, self.wellX, self.wellY, self.wellZ)
        elif env in ['2s-d', '2s-c', '2r-d', '2r-c']:
            base = self._normalize_basic_stresses([self.actionValue])
            wells = self._normalize_well(self.wellQ, self.wellX, self.wellY, self.wellZ)
        elif env in ['3s-d', '3s-c', '3r-d', '3r-c']:
            base = self._normalize_basic_stresses([self.headSpecSouth, self.headSpecNorth])
            wells = self._normalize_well(self.wellQ, self.wellX, self.wellY, self.wellZ)
        elif env in [
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        ]:
            base = self._normalize_basic_stresses([self.headSpecSouth, self.headSpecNorth])
            wells = self._normalize_helper_wells()
        else:
            return []

        return base + wells

    def distance_to_well(self, particle, well):
        """Calculate Euclidean distance between a particle and a well with Y correction.

        The Y coordinate is corrected by subtracting particle and well Y values
        from self.extentY to account for coordinate system specifics.

        Args:
            self (object): The instance of the class.
            particle (tuple or list): Coordinates of the particle (x, y).
            well (tuple or list): Coordinates of the well (x, y).

        Returns:
            float: Euclidean distance between particle and well with Y correction.

        """
        dx = particle[0] - well[0]
        dy = self.extentY - particle[1] - well[1]  # Y correction as per coordinate system
        return sqrt(dx**2 + dy**2)

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

    def check_particle_within_well(self):
        """
        Check if the particle is within the main well radius.

        Args:
            self (object): Instance containing particle and well attributes.

        """
        dist = self.distance_to_well(self.particleCoords, self.wellCoords)
        if dist <= self.wellRadius:
            self.done = True
            self.update_reward_on_fail()

    def check_particle_within_helper_wells(self):
        """
        Check if the particle is within any of the helper wells.

        Args:
            self (object): Instance containing particle, helper wells, and environment info.

        """
        valid_envs = {
            '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost',
            '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c'
        }
        if self.ENVTYPE not in valid_envs:
            return

        for i in range(self.nHelperWells):
            well_key = f'wellCoords{i + 1}'
            well_coords = self.helperWells[well_key]
            dist = self.distance_to_well(self.particleCoords, well_coords)
            if dist <= self.helperWellRadius:
                self.done = True
                self.update_reward_on_fail()
                break

    def check_boundaries(self):
        """
        Check if the particle has reached any boundary and update done/reward.

        Args:
            self (object): Instance containing particle coordinates and boundary info.

        """
        x, y, _ = self.particleCoords
        min_x, min_y = self.minX, self.minY
        max_x = min_x + self.extentX
        max_y = min_y + self.extentY

        if self._check_eastern_boundary(x, max_x):
            return
        if self._check_western_boundary(x, min_x):
            return
        if self._check_northern_boundary(y, max_y):
            return
        self._check_southern_boundary(y, min_y)

    def _check_eastern_boundary(self, x, max_x):
        """
        Check if the particle has reached the eastern boundary.

        Args:
            self (object): Instance containing boundary and state info.
            x (float): Particle x-coordinate.
            max_x (float): Maximum x boundary coordinate.

        Returns:
            bool: True if eastern boundary is reached, else False.
        
        """
        if x >= max_x - self.dCol:
            self.done = True
            return True
        return False

    def _check_western_boundary(self, x, min_x):
        """
        Check if the particle has reached the western boundary.

        Args:
            self (object): Instance containing boundary and state info.
            x (float): Particle x-coordinate.
            min_x (float): Minimum x boundary coordinate.

        Returns:
            bool: True if western boundary is reached, else False.
        
        """
        if x <= min_x + self.dCol:
            self.done = True
            self.update_reward_on_fail()
            return True
        return False

    def _check_northern_boundary(self, y, max_y):
        """
        Check if the particle has reached the northern boundary.

        Args:
            self (object): Instance containing environment, boundary, and state info.
            y (float): Particle y-coordinate.
            max_y (float): Maximum y boundary coordinate.

        Returns:
            bool: True if northern boundary is reached, else False.
        
        """
        valid_envs = {
            '1s-d', '1s-c', '1r-d', '1r-c', '3s-d', '3s-c', '3r-d', '3r-c',
            '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d',
            '5r-c', '6s-d', '6s-c', '6r-d', '6r-c'
        }
        if self.ENVTYPE in valid_envs and y >= max_y - self.dRow:
            self.done = True
            self.update_reward_on_fail()
            return True
        return False

    def _check_southern_boundary(self, y, min_y):
        """
        Check if the particle has reached the southern boundary.

        Args:
            self (object): Instance containing boundary and state info.
            y (float): Particle y-coordinate.
            min_y (float): Minimum y boundary coordinate.

        """
        if y <= min_y + self.dRow:
            self.done = True
            self.update_reward_on_fail()

    def check_max_steps(self):
        """
        Abort the game if the maximum number of steps is reached.

        Args:
            self (object): Instance containing timestep and state info.

        """
        if self.timeStep == self.maxSteps and not self.done:
            self.done = True
            self.update_reward_on_fail()

    def check_particle_state(self):
        """
        Main method to check particle state and update done/reward flags.

        Args:
            self (object): Instance with particle and environment state methods.

        """
        self.check_particle_within_well()
        self.check_particle_within_helper_wells()
        self.check_boundaries()
        self.check_max_steps()

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

    def load_mf6_model(self, exe, modelname):
        """
        Attempt to load the MF6 model executable using XmiWrapper.

        This method constructs the MF6 configuration filename based on the model
        workspace and model name, then tries to initialize the MF6 model wrapper
        with the given executable. If loading fails, it prints an error message
        and returns a fallback value using `bmi_return`.

        Args:
            exe (str): Path or name of the MF6 executable to be loaded.
            modelname (str): Name of the model used to construct the config file.

        Returns:
            object: Initialized MF6 wrapper instance on success,
                    or result of `self.bmi_return` on failure.

        """
        mf6_config_file = join(self.model_ws, modelname + '.nam')
        try:
            self.mf6 = XmiWrapper(exe)
            return self.mf6
        except Exception as e:
            print(f"Failed to load {exe}")
            print(f"with message: {e}")
            return self.bmi_return(self.success, self.model_ws, modelname)

    def initialize_simulation(self, modelname):
        """
        Initialize the MF6 simulation using the specified model name.

        This method attempts to initialize the MF6 model with the given model
        name's configuration file (assumed to have a '.nam' extension). If
        initialization fails, it returns a fallback value using `self.bmi_return`.

        Args:
            modelname (str): Base name of the model configuration file (no extension).

        Returns:
            object or None: None if initialization succeeds,
                            or result of `self.bmi_return` if it fails.

        """
        try:
            self.mf6.initialize(modelname + '.nam')
        except Exception as e:
            # Optionally, log the exception e here
            return self.bmi_return(self.success, self.model_ws)

    def setup_workspace_and_model(self):
        """
        Set up the model workspace directory and initialize related attributes.

        This method constructs the model workspace path by joining the base
        workspace, 'models' folder, and experiment directory. It also sets the
        success flag to False and converts the model name to uppercase. If the
        workspace path exists, it changes the current working directory to it.

        Args:
            self (object): Instance with workspace, experiment directory, and name.

        """
        self.model_ws = join(self.wrkspc, 'models', self.exdir)
        self.success = False
        self.nameUpper = self.name.upper()
        if self.model_ws is not None:
            chdir(self.model_ws)

    def load_and_initialize_model(self, exe, modelname):
        """
        Load and initialize the MF6 model.

        Args:
            self (object): Instance with model loading and initialization methods.
            exe (str): Path or name of the MF6 executable.
            modelname (str): Name of the model configuration file base.

        """
        self.load_mf6_model(exe, modelname)
        self.initialize_simulation(modelname)


    def run_steady_state_solution(self, exe, modelname):
        """
        Run the steady-state solution for the MF6 model.

        This method writes the simulation files silently, initializes the model,
        prepares the time step, runs the convergence loop, and handles failure if
        convergence is not achieved. It then retrieves and reshapes the hydraulic
        head data and finalizes the model.

        Args:
            self (object): Instance with simulation and model methods.
            exe (str): Path or name of the MF6 executable.
            modelname (str): Name of the model configuration file base.

        Returns:
            object or None: Result of convergence failure handler if convergence fails,
                            otherwise None.

        """
        self.sim_steadyState.write_simulation(silent=True)
        self.initialize_simulation(modelname)

        dt = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

        self.get_max_iterations()
        if not self.run_convergence_loop():
            return self.handle_convergence_failure(modelname)

        head_tag = self.mf6.get_var_address("X", self.nameUpper)
        head_flat = deepcopy(self.mf6.get_value_ptr(head_tag))
        self.head_steadyState = deepcopy(head_flat).reshape(
            (self.nlay, self.nrow, self.ncol)
        )
        self.head_steadyState_flat = deepcopy(head_flat)

        self.mf6.finalize()

    def prepare_transient_simulation(self, exe, modelname):
        """
        Prepare the transient simulation for the MF6 model.

        This method writes the simulation files silently, resets the timestep,
        loads and initializes the MF6 model, retrieves time step parameters,
        obtains simulated heads, and sets the steady-state heads as the initial
        condition.

        Args:
            self (object): Instance with simulation and model methods.
            exe (str): Path or name of the MF6 executable.
            modelname (str): Name of the model configuration file base.

        """
        self.sim.write_simulation(silent=True)
        self.timeStep = 0

        self.load_mf6_model(exe, modelname)
        self.initialize_simulation(modelname)

        self.dt = self.mf6.get_time_step()
        self.current_time = self.mf6.get_current_time()
        self.end_time = self.mf6.get_end_time()

        self.get_simulated_heads()

        # Set steady-state heads as initial condition
        head_tag = self.mf6.get_var_address("X", self.nameUpper)
        self.mf6.set_value(head_tag, self.head_steadyState)

    def retrieve_boundary_conditions(self):
        """
        Retrieve boundary condition values from the MF6 model.

        This method fetches the recharge boundary condition and well data
        from the model and stores the well data for later use.

        Args:
            self (object): Instance with MF6 model access methods.

        Returns:
            numpy.ndarray: Copy of the new recharge boundary condition array.

        """
        rch_tag = self.mf6.get_var_address("BOUND", self.nameUpper, "RCHA_0")
        new_recharge = self.mf6.get_value(rch_tag).copy()

        self.well_tag = self.mf6.get_var_address("BOUND", self.nameUpper, "WEL_0")
        self.well = self.mf6.get_value(self.well_tag)

        return new_recharge


    def prepare_observations(self, new_recharge):
        """
        Prepare observations using recharge and well/storm data.

        This method builds well and storm observations and combines them
        with the new recharge data to update the observations attribute.

        Args:
            self (object): Instance with observation-building methods.
            new_recharge (numpy.ndarray): Recharge boundary condition array.

        """
        obs_wells = self.build_observations_wells()
        obs_storms = self.build_observations_storms()
        self.observations = self.build_observations(new_recharge, obs_wells, obs_storms)

    def build_observations_wells(self):
        """
        Construct and return a list of normalized well observation parameters.

        For each helper well (up to self.nHelperWells), this method normalizes:
        - well x-coordinate between 0 and self.extentX
        - well y-coordinate between 0 and self.extentY
        - well flow rate Q between self.minQ and self.minQ + self.diffQ

        The normalized parameters for each well are appended sequentially to a list.

        Returns:
            list of float: Flat list of normalized well observation values for all wells.
        
        """
        actions = self.actionsDict  # Local reference for speed
        observations_wells = []
        for iHelper in range(self.nHelperWells):
            x = self.normalize(actions[f'well{iHelper}x'], 0, self.extentX)
            y = self.normalize(actions[f'well{iHelper}y'], 0, self.extentY)
            Q = self.normalize(actions[f'well{iHelper}Q'], self.minQ, self.diffQ)
            observations_wells.extend([x, y, Q])

        return observations_wells


    def build_observations_storms(self):
        """
        Construct and return a list of normalized storm observations.

        For each storm (up to self.nStorms), this method normalizes and appends:
        - storm start time (self.stormStarts[i]) normalized by self.nstp
        - normalized current time step (self.timeStep / self.nstp)
        - storm duration normalized between 0 and self.maxStormDuration
        - storm intensity normalized between self.storm_rch_min and intensity range
        - storm center X coordinate normalized between 0 and self.extentX
        - storm center Y coordinate normalized between 0 and self.extentY

        Returns:
            list of float: Flat list of normalized storm observation values for all storms.
        
        """
        observations_storms = []
        time_step_norm = self.timeStep / self.nstp
        intensity_range = abs(self.storm_rch_min - self.storm_rch_max)

        for i in range(self.nStorms):
            observations_storms.extend([
                self.stormStarts[i] / self.nstp,
                time_step_norm,
                self.normalize(self.stormDurations[i], 0, self.maxStormDuration),
                self.normalize(self.stormIntensities[i], self.storm_rch_min, intensity_range),
                self.normalize(self.stormCentersX[i], 0., self.extentX),
                self.normalize(self.stormCentersY[i], 0., self.extentY),
            ])

        return observations_storms

    def build_observations(self, new_recharge, observations_wells, observations_storms):
        """
        Normalize and flatten input arrays, then extend and return a combined list.

        This method normalizes these arrays:
        - self.head using self.chd_east and self.chd_diff
        - new_recharge using self.avg_rch_min and self.rch_diff

        After normalization and flattening, it extends the observation list with:
        - normalized and flattened self.head
        - normalized and flattened new_recharge
        - observations_wells (iterable)
        - observations_storms (iterable)

        Args:
            self (object): Instance of the class.
            new_recharge (array-like): New recharge data to normalize and add.
            observations_wells (iterable): Well observation data to add.
            observations_storms (iterable): Storm observation data to add.

        Returns:
            list: Combined list of normalized, flattened, and extended observations.
        
        """
        obs = []
        obs.extend(self.normalize(self.head, self.chd_east, self.chd_diff).flatten())
        obs.extend(self.normalize(new_recharge, self.avg_rch_min, self.rch_diff).flatten())
        obs.extend(observations_wells)
        obs.extend(observations_storms)

        return obs

    def prepare_time_step(self):
        """Prepare the model for the current time step.

        Args:
            self (object): Instance of the class containing the model and time step.
        
        """
        self.mf6.prepare_time_step(self.dt)
        self.current_time = self.mf6.get_current_time()

    def get_simulated_heads(self):
        """Retrieve pointer to simulated heads.

        Args:
            self (object): Instance containing the model and variable info.
        
        """
        head_tag = self.mf6.get_var_address("X", self.nameUpper)
        self.head = self.mf6.get_value_ptr(head_tag)

    def get_recharge(self):
        """Retrieve the current recharge value.

        Args:
            self (object): Instance containing the model and variable info.

        Returns:
            float: Current recharge value.
        
        """
        self.rch_tag = self.mf6.get_var_address("BOUND", self.nameUpper, "RCHA_0")
        return self.mf6.get_value(self.rch_tag)

    def get_action_value(self, iHelper, suffix):
        """Get scaled action value from actionsDict for a given well and suffix.

        Args:
            self (object): Instance containing actions and scaling factors.
            iHelper (int): Well index or identifier.
            suffix (str): Action suffix indicating the type of action.

        Returns:
            float: Scaled action value.
        
        """
        key = f"well{iHelper}{suffix}"
        if "dx" in suffix or "dy" in suffix or "dQ" in suffix:
            scale = self.well_dxMax if "dx" in suffix else self.well_dyMax
            return scale * self.actionsDict[key]
        return self.actionsDict[key]

    def collect_Qs(self):
        """Collect Q values for all helper wells.

        Args:
            self (object): Instance containing helper wells and action values.

        Returns:
            list[float]: List of Q values for each helper well.
        
        """
        Qs = [self.get_action_value(i, 'Q') for i in range(self.nHelperWells)]
        return Qs

    def update_well_Q(self):
        """Update the well Q values.

        Args:
            self (object): Instance containing well data and Q values.
        
        """
        self.well[:, 0] = self.collect_Qs()

    def get_max_iterations(self):
        """Retrieve maximum iterations from MF6 solver settings. Sets self.max_iter to the maximum number of solver iterations.
        
        Args:
            self (object): Instance containing MF6 model interface.

        """
        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        self.max_iter = int(self.mf6.get_value(mxit_tag))

    def run_convergence_loop(self):
        """Run solver iterations until convergence or max iterations reached.

        Args:
            self (object): Instance containing MF6 model interface and max_iter.

        Returns:
            bool: True if solver converged within max iterations, else False.
        
        """
        self.mf6.prepare_solve(1)
        for _ in range(self.max_iter):
            if self.mf6.solve(1):
                return True
        return False

    def handle_convergence_failure(self, modelname):
        """Handle failed convergence scenario.

        Args:
            self (object): Instance containing model state and methods.
            modelname (str): Name of the model.

        Returns:
            Any: Result of bmi_return method.
        
        """
        print("failed converging")
        return self.bmi_return(self.success, self.model_ws, modelname)

    def update_envhash(self):
        """Create and store a hash of the observations history.

        Args:
            self (object): Instance containing observations history.
        
        """
        self.envhash = self.create_single_hash(self.observations_history)

    def cleanup_temp_dir(self):
        """Clean up the temporary directory.

        Args:
            self (object): Instance containing tempDir attribute.
        
        """
        self.tempDir.cleanup()

    def finalize_simulation(self):
        """Finalize the MF6 simulation and mark success.

        Handles exceptions gracefully without raising.

        Args:
            self (object): Instance containing MF6 model and success flag.
        
        """
        try:
            self.mf6.finalize()
            self.success = True
        except Exception:
            _ = self.bmi_return(self.success, self.model_ws)

    def post_simulation_teardown(self, teardownOnFinish):
        """Perform post-simulation cleanup and teardown if requested.

        Args:
            self (object): Instance containing model workspace and teardown method.
            teardownOnFinish (bool): Whether to run teardown after finishing.
        
        """

        chdir(self.model_ws)
        if teardownOnFinish:
            self.teardown()

    def handle_done(self, teardownOnFinish):
        """Handle tasks when simulation is done.

        Tasks performed:
        - Cleanup temp directory
        - Finalize simulation
        - Change working directory and optionally teardown

        Args:
            self (object): Instance with done flag and cleanup/finalize methods.
            teardownOnFinish (bool): Whether to run teardown after finishing.
        
        """
        if not self.done:
            return

        self.cleanup_temp_dir()
        self.finalize_simulation()
        self.post_simulation_teardown(teardownOnFinish)

    def getActionType(self, ENVTYPE):
        """
        Retrieve the action type based on the ENVTYPE string.

        Args:
            self (object): Instance of the class.
            ENVTYPE (str): Environment type string containing '-d' or '-c'.

        Returns:
            str: 'discrete' if '-d' in ENVTYPE, 'continuous' if '-c' in ENVTYPE.

        Raises:
            ValueError: If ENVTYPE does not contain '-d' or '-c'.
        
        """
        if '-d' in ENVTYPE:
            return 'discrete'
        if '-c' in ENVTYPE:
            return 'continuous'
        raise ValueError(
            f"Unknown environment type: '{ENVTYPE}'. Expected '-d' or '-c'."
        )

    def updateModel(self):
        """
        Update the model domain for a transient simulation.

        This method reconstructs or refreshes the model setup to reflect changes
        in the simulation domain or parameters for the current time step.

        Args:
            self (object): Instance of the class.
        
        """
        self.constructModel()

    def updateWellRate(self):
        """
        Update the well pumping rate with random fluctuations within constraints.

        Applies a random change to `wellQ` scaled by `maxQChange`. The updated rate
        is clamped between `minQ` and `maxQ`. Update occurs only if `ENVTYPE` is in
        the valid environment types.

        Args:
            self (object): Instance of the class with attributes:
        
        """
        valid_envtypes = {
            '1r-d', '1r-c', '2r-d', '2r-c', '3r-d', '3r-c',
            '4r-d', '4r-c', '5r-d', '5r-c', '6r-d', '6r-c'
        }

        if self.ENVTYPE in valid_envtypes:
            dQ = uniform(-1.0, 1.0) * self.maxQChange
            updatedQ = self.wellQ + dQ
            # Clamp updatedQ within minQ and maxQ
            self.wellQ = py_max(self.minQ, py_min(updatedQ, self.maxQ))

    def updateWell(self):
        """
        Update well locations, flow rates, cell info, and MODFLOW well data.

        Calls internal methods to update the main well location, helper wells'
        locations and rates, cell information, and MODFLOW well configuration.

        Args:
            self (object): Instance of the class.
        
        """
        self._update_main_well_location()
        self._update_helper_wells_location()
        self._update_helper_wells_Q()
        self._update_cell_info()
        self._update_modflow_wel()

    def _update_main_well_location(self):
        """
        Update main well location based on environment type.

        For specific environment types, apply random coordinate changes within
        allowed bounds. For others, set coordinates from action values.

        Args:
            self (object): Instance with attributes:
        
        """
        env_rand = {
            '1r-d', '1r-c', '2r-d', '2r-c', '4r-d', '4r-c', '5r-d', '5r-c',
            '6r-d', '6r-c'
        }
        env_action = {'3s-d', '3s-c', '3r-d', '3r-c'}

        if self.ENVTYPE in env_rand:
            dx = uniform(-1, 1) * self.maxCoordChange
            dy = uniform(-1, 1) * self.maxCoordChange
            newX = self.wellX + dx
            newY = self.wellY + dy
            if self.dCol < newX < self.extentX - self.dCol:
                self.wellX = newX
            if self.dRow < newY < self.extentY - self.dRow:
                self.wellY = newY

        elif self.ENVTYPE in env_action:
            self.wellX = self.actionValueX
            self.wellY = self.actionValueY

        self.wellCoords = [self.wellX, self.wellY, self.wellZ]


    def _update_helper_wells_location(self):
        """
        Update helper wells' locations if environment type allows.

        Sets helper wells' X and Y coordinates from action values and updates
        their coordinate lists. Z coordinate remains unchanged.

        Args:
            self (object): Instance with attributes:
                - ENVTYPE (str): Environment type.
                - nHelperWells (int): Number of helper wells.
                - helperWells (dict): Helper wells data with keys like
                  'wellX1', 'actionValueX1', etc.
        
        """
        env_helpers_loc = {
            '4s-d', '4s-c', '4r-d', '4r-c', '6s-d', '6s-c', '6r-d', '6r-c'
        }
        if self.ENVTYPE not in env_helpers_loc:
            return

        for i in range(self.nHelperWells):
            w = str(i + 1)
            self.helperWells[f'wellX{w}'] = self.helperWells[f'actionValueX{w}']
            self.helperWells[f'wellY{w}'] = self.helperWells[f'actionValueY{w}']
            self.helperWells[f'wellCoords{w}'] = [
                self.helperWells[f'wellX{w}'],
                self.helperWells[f'wellY{w}'],
                self.helperWells[f'wellZ{w}']
            ]

    def _update_helper_wells_Q(self):
        """Update helper wells' Q values from action value Qs.

        Args:
            self (object): Instance containing helper wells and environment info.
        
        """
        env_helpers_q = {
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        }
        if self.ENVTYPE not in env_helpers_q:
            return

        for i in range(self.nHelperWells):
            w = str(i + 1)
            self.helperWells[f'wellQ{w}'] = self.helperWells[f'actionValueQ{w}']


    def _update_cell_info(self):
        """Update cell info for main well and helper wells.

        Args:
            self (object): Instance containing well coordinates and helper wells.
        
        """
        self.wellCellLayer, self.wellCellColumn, self.wellCellRow = (
            self.cellInfoFromCoordinates([self.wellX, self.wellY, self.wellZ])
        )

        env_helpers = {
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        }
        if self.ENVTYPE not in env_helpers:
            return

        for i in range(self.nHelperWells):
            w = str(i + 1)
            l, c, r = self.cellInfoFromCoordinates([
                self.helperWells[f'wellX{w}'],
                self.helperWells[f'wellY{w}'],
                self.helperWells[f'wellZ{w}']
            ])
            self.helperWells[f'l{w}'] = l
            self.helperWells[f'c{w}'] = c
            self.helperWells[f'r{w}'] = r

    def _update_modflow_wel(self):
        """Update the MODFLOW well package with current well and helper wells data.

        Constructs the stress period data for the well package (`wel`) using the
        main well's cell coordinates and flow rate, and includes helper wells if
        applicable based on the environment type.

        Args:
            self (object): Instance containing well info, helper wells, and MODFLOW model.

        """
        l, c, r = self.wellCellLayer, self.wellCellColumn, self.wellCellRow
        lrcq_list = [[l - 1, r - 1, c - 1, self.wellQ]]

        env_helpers = {
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        }

        if self.ENVTYPE in env_helpers:
            for i in range(self.nHelperWells):
                w = str(i + 1)
                lrcq_list.append([
                    self.helperWells[f'l{w}'] - 1,
                    self.helperWells[f'r{w}'] - 1,
                    self.helperWells[f'c{w}'] - 1,
                    self.helperWells[f'wellQ{w}']
                ])

        lrcq = {0: lrcq_list}
        self.wel = ModflowWel(self.mf, stress_period_data=lrcq, options=['NOPRINT'])

    def constructModel(self):
        """Construct the groundwater flow model for the arcade game.

        Uses Flopy as a MODFLOW wrapper to build input files. Applies specified head
        boundary conditions on the western, eastern, and southern boundaries. The
        southern boundary condition is modifiable during gameplay. Western and eastern
        boundaries promote groundwater flow westward. The model assumes homogeneous
        parameters and aquifer thickness.

        Args:
            self (object): Instance containing environment type and model construction methods.

        """
        if self.ENVTYPE == '0s-c':
            self.build_model_0sc()
        else:
            self.build_model_other()

    def create_simulation_0sc(self):
        """Create a MODFLOW simulation.

        Creates a MODFLOW simulation using MF6.

        Args:
            self (object): Instance to attach the simulation.

        """
        self.sim = MFSimulation(
            sim_name=self.name,
            version='mf6',
            exe_name=self.mf6dll,
            sim_ws=self.model_ws,
            memory_print_option='all'
        )


    def create_tdis_0sc(self, steady_state=True):
        """Create a MODFLOW time discretization (TDIS) package.

        Creates a TDIS package for steady-state or transient simulations.

        Args:
            self (object): The instance to attach the TDIS package.
            steady_state (bool, optional): Whether to create a steady-state
                simulation. Defaults to True.

        """
        if steady_state:
            # Very long period for steady state
            period_data = [(1e13, 100, 1)]
            self.tdis = ModflowTdis(
                self.sim,
                time_units="DAYS",
                nper=1,
                perioddata=period_data
            )
        else:
            self.tdis = ModflowTdis(
                self.sim,
                time_units='DAYS',
                nper=self.nper,
                perioddata=self.tdis_rc
            )

    def create_ims_0sc(self):
        """Create the MODFLOW IMS (iterative model solution) package.

        Configures the IMS package with solver parameters for the simulation.

        Args:
            self (object): Instance containing simulation and solver settings.

        """
        self.ims = ModflowIms(
            self.sim,
            print_option='SUMMARY',
            outer_dvclose=self.hclose,
            outer_maximum=self.nouter,
            under_relaxation='SIMPLE',
            under_relaxation_gamma=0.98,
            inner_maximum=self.ninner,
            inner_dvclose=self.hclose,
            rcloserecord=self.rclose,
            linear_acceleration='BICGSTAB',
            relaxation_factor=self.relax
        )

    def create_gwf_model_0sc(self):
        """Create the groundwater flow (GWF) model for the simulation.

        Initializes the GWF model with Newton solver options and model parameters.

        Args:
            self (object): Instance containing simulation and model settings.

        """
        newton_opts = ['NEWTON', 'UNDER_RELAXATION']
        self.gwf = ModflowGwf(
            self.sim,
            newtonoptions=newton_opts,
            modelname=self.name,
            print_input=False,
            save_flows=True
        )

    def setup_dis_ic_0sc(self):
        """Set up the discretization (DIS) and initial conditions (IC) packages.

        Configures the grid discretization and initial head values for the GWF model.

        Args:
            self (object): Instance containing model grid and initial condition data.

        """
        ModflowGwfdis(
            self.gwf,
            nlay=self.nlay,
            nrow=self.nrow,
            ncol=self.ncol,
            delr=self.delr,
            delc=self.delc,
            top=self.top,
            botm=self.botm
        )
        ModflowGwfic(self.gwf, strt=self.strt)

    def setup_chd_0sc(self):
        """Set up the constant head (CHD) boundary conditions.

        Applies constant head boundaries on the western and eastern edges for all layers and rows.

        Args:
            self (object): Instance containing model dimensions and boundary head values.

        """
        chd_rec = []
        for ilay in range(self.nlay):
            for irow in range(self.nrow):
                chd_rec.append([(ilay, irow, 0), self.chd_west])
                chd_rec.append([(ilay, irow, self.ncol - 1), self.chd_east])

        ModflowGwfchd(
            self.gwf,
            maxbound=len(chd_rec),
            stress_period_data=chd_rec,
            save_flows=True
        )

    def setup_npf_0sc(self):
        """Set up the node property flow (NPF) package.

        Configures hydraulic conductivity and cell type for the groundwater flow model.

        Args:
            self (object): Instance containing model and hydraulic conductivity data.

        """
        ModflowGwfnpf(
            self.gwf,
            save_flows=True,
            icelltype=1,
            k=self.hk
        )

    def setup_rch_0sc(self, recharge):
        """Set up the recharge (RCH) package.

        Applies recharge to the groundwater flow model.

        Args:
            self (object): Instance containing the groundwater flow model.
            recharge (float or array-like): Recharge rate(s) to apply.

        """
        ModflowGwfrcha(self.gwf, recharge=recharge)

    def setup_oc_0sc(self):
        """Set up the output control (OC) package.

        Configures output files and print options for heads.

        Args:
            self (object): Instance containing model name and groundwater flow model.

        """
        ModflowGwfoc(
            self.gwf,
            head_filerecord=f'{self.name}.hds',
            headprintrecord=[('COLUMNS', 10, 'WIDTH', 15, 'DIGITS', 6, 'GENERAL')],
            saverecord=[('HEAD', 'ALL')]
        )

    def setup_sto_0sc(self):
        """Set up the storage (STO) package.

        Configures specific storage, specific yield, and transient settings.

        Args:
            self (object): Instance containing the groundwater flow model.

        """
        ModflowGwfsto(
            self.gwf,
            save_flows=True,
            iconvert=1,
            ss=1e-5,
            sy=0.2,
            transient={0: True}
        )

    def setup_wells_0sc(self):
        """Set up the well (WEL) package.

        Defines well locations and flow rates for the simulation stress period 0.

        Args:
            self (object): Instance containing well actions and groundwater flow model.

        """
        wd = [
            (
                (0,
                 self.actionsDict[f'well{i}iCol'],
                 self.actionsDict[f'well{i}iRow']),
                self.actionsDict[f'well{i}Q']
            )
            for i in range(self.nHelperWells)
        ]
        ModflowGwfwel(self.gwf, stress_period_data={0: wd}, save_flows=True)

    def build_steady_state_sim_0sc(self):
        """Build a steady-state MODFLOW simulation.

        Calls setup methods in sequence to create a steady-state simulation.
        Saves a deepcopy of the simulation for later transient use.

        Args:
            self (object): Instance containing all simulation setup methods and data.

        """
        self.create_simulation_0sc()
        self.create_tdis_0sc(steady_state=True)
        self.create_ims_0sc()
        self.create_gwf_model_0sc()
        self.setup_dis_ic_0sc()
        self.setup_chd_0sc()
        self.setup_npf_0sc()
        self.setup_rch_0sc(self.rechTimeSeries[0])
        self.setup_oc_0sc()
        # Save a deepcopy for transient simulation later
        self.sim_steadyState = deepcopy(self.sim)

    def build_transient_sim_0sc(self):
        """Build a transient MODFLOW simulation.

        Replaces the steady-state TDIS package with a transient one, sets up storage,
        wells, and updates the recharge package for transient simulation.

        Args:
            self (object): Instance containing simulation and model data.

        """
        # Remove steady-state TDIS and add transient TDIS
        self.sim.remove_package(self.tdis)
        self.create_tdis_0sc(steady_state=False)
        self.setup_sto_0sc()
        self.setup_wells_0sc()
        # Remove and re-add recharge package to update it
        self.gwf.remove_package('rcha')
        # self.gwf.remove_package(self.gwf.package_key_dict['rcha'])
        self.setup_rch_0sc(self.rechTimeSeries[0])

    def build_model_0sc(self):
        """Build the full MODFLOW model for the '0s-c' environment.

        Creates steady-state simulation first, then configures transient simulation.

        Args:
            self (object): Instance containing all setup methods.

        """
        self.build_steady_state_sim_0sc()
        self.build_transient_sim_0sc()

    def create_model_other(self):
        """Create a MODFLOW model for other environment types.

        Initializes a Modflow model with specified executable and workspace.

        Args:
            self (object): Instance containing model parameters.

        """
        self.mf = Modflow(
            self.MODELNAME,
            exe_name=self.exe_name,
            verbose=False
        )
        self.mf.change_model_ws(new_pth=self.modelpth)

    def create_discretization_other(self):
        """Create discretization (DIS) package for non-'0s-c' environments.

        Sets up grid and temporal discretization parameters for the model.

        Args:
            self (object): Instance containing model grid and temporal parameters.

        """
        common_kwargs = dict(
            nlay=self.nLay,
            nrow=self.nRow,
            ncol=self.nCol,
            delr=self.dRow,
            delc=self.dCol,
            top=self.zTop,
            botm=self.botM[1:],
            steady=self.periodSteadiness,
            itmuni=4,  # days
            lenuni=2   # meters
        )
        if self.periodSteadiness:
            self.dis = ModflowDis(self.mf, **common_kwargs)
        else:
            self.dis = ModflowDis(
                self.mf,
                nper=self.periods,
                nstp=self.periodSteps,
                perlen=[2 * self.periodLength],
                **common_kwargs
            )

    def initialize_ibound_other(self):
        """Initialize the ibound array defining active and inactive cells.

        Sets boundary cells inactive (-1) based on environment type.

        Args:
            self (object): Instance containing model dimensions and environment type.

        """
        self.ibound = ones((self.nLay, self.nRow, self.nCol), dtype=int32)
        env = self.ENVTYPE
        if env in {
            '1s-d', '1s-c', '1r-d', '1r-c',
            '3s-d', '3s-c', '3r-d', '3r-c',
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        }:
            self.ibound[:, 1:-1, 0] = -1
            self.ibound[:, 1:-1, -1] = -1
            self.ibound[:, 0, :] = -1
            self.ibound[:, -1, :] = -1
        elif env in {'2s-d', '2s-c', '2r-d', '2r-c'}:
            self.ibound[:, :-1, 0] = -1
            self.ibound[:, :-1, -1] = -1
            self.ibound[:, -1, :] = -1

    def initialize_start_heads_other(self):
        """Initialize starting heads for the model.

        Uses uniform heads for steady state or previous heads for transient.
        Updates heads for certain environment types and assigns boundary heads.

        Args:
            self (object): Instance containing model state and environment info.

        """
        if self.periodSteadiness:
            self.strt = ones((self.nLay, self.nRow, self.nCol), dtype=float32)
        else:
            self.strt = self.headsPrev

        # Update heads if timeStep > 0 and environment requires it
        if self.timeStep > 0 and self.ENVTYPE in {
            '3r-d', '3r-c', '4r-d', '4r-c',
            '5r-d', '5r-c', '6r-d', '6r-c'
        }:
            self._update_random_head_fluctuations()

        # Assign boundary heads based on environment type
        self._assign_boundary_heads()

    def _update_random_head_fluctuations(self):
        """Update boundary heads with random fluctuations within specified limits.

        Applies random perturbations to southern and northern specified heads,
        clamping the results between minimum and maximum head values.

        Args:
            self (object): Instance containing head specifications and limits.

        """
        dHSouth = uniform(-1, 1) * self.maxHChange
        dHNorth = uniform(-1, 1) * self.maxHChange

        self.headSpecSouth = self._clamp(self.headSpecSouth + dHSouth, self.minH, self.maxH)
        self.headSpecNorth = self._clamp(self.headSpecNorth + dHNorth, self.minH, self.maxH)

    def _clamp(self, val, vmin, vmax):
        """Clamp a value between minimum and maximum limits.

        Args:
            self (object): Instance (required for method).
            val (float): Value to clamp.
            vmin (float): Minimum allowed value.
            vmax (float): Maximum allowed value.

        Returns:
            float: Clamped value.
        """
        return py_max(vmin, py_min(val, vmax))

    def _assign_boundary_heads(self):
        """Assign boundary heads to the starting head array based on environment type.

        Sets boundary heads on west, east, south, and north edges according to ENVTYPE.

        Args:
            self (object): Instance containing environment type and head arrays.

        """
        env = self.ENVTYPE

        if env in {'1s-d', '1s-c', '1r-d', '1r-c'}:
            self.strt[:, 1:-1, 0] = self.headSpecWest
            self.strt[:, 1:-1, -1] = self.headSpecEast
            self.strt[:, 0, :] = self.actionValueSouth
            self.strt[:, -1, :] = self.actionValueNorth

        elif env in {'2s-d', '2s-c', '2r-d', '2r-c'}:
            self.strt[:, :-1, 0] = self.headSpecWest
            self.strt[:, :-1, -1] = self.headSpecEast
            self.strt[:, -1, :] = self.actionValue

        elif env in {
            '3s-d', '3s-c', '3r-d', '3r-c',
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        }:
            self.strt[:, 1:-1, 0] = self.headSpecWest
            self.strt[:, 1:-1, -1] = self.headSpecEast
            self.strt[:, 0, :] = self.headSpecSouth
            self.strt[:, -1, :] = self.headSpecNorth

    def add_bas_package_other(self):
        """Add the Basic (BAS) package to the MODFLOW model.

        Sets the ibound and starting head arrays.

        Args:
            self (object): Instance containing model and boundary data.

        """
        self.mf_bas = ModflowBas(self.mf, ibound=self.ibound, strt=self.strt)

    def add_lpf_package_other(self):
        """Add the Layer Property Flow (LPF) package with uniform hydraulic conductivity.

        Args:
            self (object): Instance containing the MODFLOW model.

        """
        hk = 10.0  # uniform hydraulic conductivity
        self.mf_lpf = ModflowLpf(
            self.mf,
            hk=hk,
            vka=10.0,
            ss=1e-5,
            sy=0.15,
            ipakcb=53
        )


    def add_oc_package_other(self):
        """Add the Output Control (OC) package.

        Configures saving heads and budgets at beginning and end of stress period.

        Args:
            self (object): Instance containing the MODFLOW model and period steps.

        """
        spd = {
            (0, 0): ['save head', 'save budget'],
            (0, self.periodSteps - 1): ['save head', 'save budget']
        }
        self.mf_oc = ModflowOc(self.mf, stress_period_data=spd, compact=True)


    def add_solver_package_other(self):
        """Add the solver package to the MODFLOW model.

        Uses ModflowPcg by default. Comments include alternatives for speed-ups
        with empirical termination criteria.

        For speed-ups try different solvers.
        Termination criteria currently empirically set. Too large termination
        criterion might cause unrealistic near-well particle trajectories.
        ModflowGmg seems slightly faster.

        from flopy.modflow import ModflowPcgn, ModflowGmg, ModflowSms, ModflowDe4, ModflowNwt
        self.mf_pcg = ModflowPcg(self.mf, hclose=1e-1, rclose=1e-1)
        self.mf_pcg = ModflowPcgn(self.mf, close_h=1e-6, close_r=1e-6)
        self.mf_pcg = ModflowGmg(self.mf, hclose=1e-1, rclose=1e-1)
        self.mf_pcg = ModflowSms(self.mf)
        self.mf_pcg = ModflowDe4(self.mf)
        self.mf_pcg = ModflowNwt(self.mf)

        Args:
            self (object): Instance containing the MODFLOW model.

        """
        self.mf_pcg = ModflowPcg(self.mf)

    def build_model_other(self):
        """Build the full MODFLOW model for non-'0s-c' environments.

        Calls setup methods in sequence to create the model.

        Args:
            self (object): Instance containing all setup methods.

        """
        self.create_model_other()
        self.create_discretization_other()
        self.initialize_ibound_other()
        self.initialize_start_heads_other()
        self.add_bas_package_other()
        self.add_lpf_package_other()
        self.add_oc_package_other()
        self.add_solver_package_other()

    def get_cellID(self, x, y):
        """Calculate the 1-based cell ID for given (x, y) coordinates.

        The ID counts cells row-wise from top-left corner.

        Args:
            self (object): Instance containing grid parameters.
            x (float): X coordinate (horizontal).
            y (float): Y coordinate (vertical).

        Returns:
            int: 1-based cell ID.

        Raises:
            ValueError: If coordinates are outside the model domain.

        """
        if not (0 <= x < self.delc * self.ncol):
            raise ValueError(
                f"x coordinate {x} is outside the model domain (0 to {self.delc * self.ncol})"
            )
        if not (0 <= y < self.delr * self.nrow):
            raise ValueError(
                f"y coordinate {y} is outside the model domain (0 to {self.delr * self.nrow})"
            )

        iCol = int(x / self.delc)
        iRow = self.nrow - 1 - int(y / self.delr)  # rows counted top to bottom
        cellID = (iRow * self.ncol) + iCol + 1  # 1-based cell ID

        return cellID

    def get_cellColRow(self, x, y):
        """Get the (column, row) indices for given (x, y) coordinates.

        Args:
            self (object): Instance containing grid parameters.
            x (float): X coordinate (horizontal).
            y (float): Y coordinate (vertical).

        Returns:
            tuple: (iCol, iRow) zero-based indices.

        Raises:
            ValueError: If coordinates are outside the model domain.

        """
        if not (0 <= x < self.delc * self.ncol):
            raise ValueError(
                f"x coordinate {x} is outside the model domain (0 to {self.delc * self.ncol})"
            )
        if not (0 <= y < self.delr * self.nrow):
            raise ValueError(
                f"y coordinate {y} is outside the model domain (0 to {self.delr * self.nrow})"
            )

        iCol = int(x / self.delc)
        iRow = self.nrow - 1 - int(y / self.delr)  # rows counted top to bottom

        return iCol, iRow

    def changeActionDict(self, actionsDict, action):
        """Update the actions dictionary with values for each helper well.

        Each helper well has six associated action values:
        'actiondxRight', 'actiondxLeft', 'actiondyUp', 'actiondyDown',
        'actiondQUp', and 'actiondQDown'.

        Args:
            self (object): Instance containing number of helper wells.
            actionsDict (dict): Dictionary to update.
            action (list or iterable): List of action values.

        Returns:
            dict: Updated actionsDict.

        Raises:
            ValueError: If length of action does not match expected size.

        """
        offset = 6
        expected_length = self.nHelperWells * offset
        if len(action) != expected_length:
            raise ValueError(
                f"Length of action ({len(action)}) does not match expected size ({expected_length})."
            )

        action_keys = [
            'actiondxRight', 'actiondxLeft',
            'actiondyUp', 'actiondyDown',
            'actiondQUp', 'actiondQDown'
        ]

        for iHelper in range(self.nHelperWells):
            base_idx = iHelper * offset
            for j, key in enumerate(action_keys):
                actionsDict[f'well{iHelper}{key}'] = action[base_idx + j]

        return actionsDict

    def getActionSpaceSize(self, actionsDict):
        """
        Returns the number of actions in the actionsDict.

        Parameters:
            actionsDict (dict): Dictionary containing actions.

        Returns:
            int: Number of actions.

        """
        return len(actionsDict)

    def getActionValues(self, actionsDict):
        """Update actionsDict with new well positions and flow rates.

        For each helper well, updates x, y, and Q values based on action increments,
        then recalculates the cellID based on updated coordinates.

        Args:
            self (object): Instance containing well parameters and methods.
            actionsDict (dict): Dictionary with current well actions and states.

        Returns:
            dict: Updated actionsDict with new positions, flow rates, and cellIDs.

        """
        for iHelper in range(self.nHelperWells):
            actionsDict = self._update_well_x(actionsDict, iHelper)
            actionsDict = self._update_well_y(actionsDict, iHelper)
            actionsDict = self._update_well_Q(actionsDict, iHelper)
            # Update cellID after x and y are updated
            x = actionsDict[f'well{iHelper}x']
            y = actionsDict[f'well{iHelper}y']
            actionsDict[f'well{iHelper}cellID'] = self.get_cellID(x, y)
        return actionsDict

    def _update_well_x(self, actionsDict, iHelper):
        """Update the x-coordinate of a helper well within domain bounds.

        Args:
            self (object): Instance containing domain and well parameters.
            actionsDict (dict): Dictionary with current well states.
            iHelper (int): Index of the helper well.

        Returns:
            dict: Updated actionsDict with new x-coordinate.

        """
        key_x = f'well{iHelper}x'
        dx_right = self.well_dxMax * actionsDict.get(f'well{iHelper}actiondxRight', 0)
        dx_left = self.well_dxMax * actionsDict.get(f'well{iHelper}actiondxLeft', 0)
        x = actionsDict[key_x] + dx_right - dx_left

        min_x = 0.0 + 1.1 * self.delr
        max_x = self.extentX - 1.1 * self.delr
        x = py_max(min_x, py_min(x, max_x))

        actionsDict[key_x] = x
        return actionsDict

    def _update_well_y(self, actionsDict, iHelper):
        """Update the y-coordinate of a helper well within domain bounds.

        Args:
            self (object): Instance containing domain and well parameters.
            actionsDict (dict): Dictionary with current well states.
            iHelper (int): Index of the helper well.

        Returns:
            dict: Updated actionsDict with new y-coordinate.

        """
        key_y = f'well{iHelper}y'
        dy_up = self.well_dyMax * actionsDict.get(f'well{iHelper}actiondyUp', 0)
        dy_down = self.well_dyMax * actionsDict.get(f'well{iHelper}actiondyDown', 0)
        y = actionsDict[key_y] + dy_up - dy_down

        min_y = 0.0 + 0.1 * self.delc
        max_y = self.extentY - 0.1 * self.delc
        y = py_max(min_y, py_min(y, max_y))

        actionsDict[key_y] = y
        return actionsDict

    def _update_well_Q(self, actionsDict, iHelper):
        """Update the flow rate Q of a helper well within min and max bounds.

        Args:
            self (object): Instance containing flow rate limits and parameters.
            actionsDict (dict): Dictionary with current well states.
            iHelper (int): Index of the helper well.

        Returns:
            dict: Updated actionsDict with new flow rate Q.

        """
        key_Q = f'well{iHelper}Q'
        dQ_up = self.well_dQMax * actionsDict.get(f'well{iHelper}actiondQUp', 0)
        dQ_down = self.well_dQMax * actionsDict.get(f'well{iHelper}actiondQDown', 0)
        Q = actionsDict[key_Q] + dQ_up - dQ_down

        Q = py_min(self.maxQ, py_max(Q, self.minQ))
        actionsDict[key_Q] = Q
        return actionsDict

    def add_lib_dependencies(self, lib_dependencies):
        """Add library dependencies for MODFLOW 6 and BMI integration.

        This method assists in configuring the environment for running MODFLOW 6
        with the Basic Model Interface (BMI), particularly for coupling with
        frameworks such as Wflow.jl via Python.

        The process may require additional system libraries and tools depending
        on the operating system:

        - **Windows**: 
            - Install a Fortran compiler, such as MinGW-w64:
              http://mingw-w64.org/doku.php
            - Optionally, use Win-builds for additional libraries:
              http://win-builds.org/doku.php/download_and_installation_from_windows

        - **Linux**:
            - Download MODFLOW 6 executables:
              https://github.com/MODFLOW-USGS/executables/releases/download/5.0/linux.zip
            - Install required Fortran libraries (e.g., libifport) as needed.
            - Reference: 
              https://software.intel.com/content/www/us/en/develop/articles/
              redistributable-libraries-for-intel-c-and-fortran-2020-compilers-for-linux.html

        - **General**:
            - Use MODFLOW 6 nightly builds if needed:
              https://github.com/MODFLOW-USGS/modflow6-nightly-build/releases
            - For Python integration, you may need the latest version of xmipy:
              https://github.com/Deltares/xmipy/blob/develop/xmipy/xmiwrapper.py

        For practical examples and troubleshooting, see:
        https://github.com/JoerivanEngelen/wflow_modflow6_bmi

        Args:
            self (object): The instance of the class.
            lib_dependencies (list[str]): List of library paths to add to the
                system PATH for dynamic linking.

        """
        if not lib_dependencies:
            return

        path_env = environ['PATH']
        sep = pathsep

        if platformSystem() == 'Windows':
            for dep in lib_dependencies:
                if dep not in path_env:
                    environ['PATH'] = dep + sep + path_env
        else:
            for dep in lib_dependencies:
                if dep not in path_env:
                    environ['PATH'] = dep + sep + path_env

    def runMODFLOW(self, check=False):
        """Execute forward groundwater flow simulation using MODFLOW.

        Args:
            self (object): Instance of ModflowRunner.
            check (bool): If True, perform model checks before running.

        """
        self._write_input()
        if check:
            self._check_model()
        self._time_run_model()
        self._load_simulation_results()

    def _write_input(self):
        """Write MODFLOW input files based on the current time step.

        Args:
            self (object): Instance of ModflowRunner.

        """
        if self.timeStep >= 1:
            self.debug_written = 'partial'
            self.mf_bas.write_file(check=False)
            self.wel.write_file()
        elif self.timeStep == 0:
            self.debug_written = 'full'
            self.mf.write_input()

    def _check_model(self):
        """Run MODFLOW model setup check with verbose output.

        Args:
            self (object): Instance of ModflowRunner.

        """
        self.check = self.mf.check(verbose=True)

    def _time_run_model(self):
        """Run the MODFLOW model and time the simulation.

        Args:
            self (object): Instance of ModflowRunner.

        Raises:
            RuntimeError: If MODFLOW does not terminate normally.

        """
        start = time()
        success, buff = self.mf.run_model(silent=True)
        if not success:
            raise RuntimeError(
                'MODFLOW did not terminate normally. '
                'Check model output for errors.'
            )
        self.success_modflow = success
        self.buff = buff
        self.t_simulate = time() - start

    def _load_simulation_results(self):
        """Load simulation heads and times from MODFLOW output files.

        Args:
            self (object): Instance of ModflowRunner.

        """
        self.fname_heads = join(self.modelpth, self.MODELNAME + '.hds')
        with HeadFile(self.fname_heads) as hf:
            self.times = hf.get_times()
            self.heads = hf.get_data(totim=self.times[-1])

    def runMODPATH(self):
        """Run forward particle tracking simulation using MODPATH.

        This method transforms particle coordinates, initializes MODPATH
        objects and input files on the first time step, modifies simulation
        files, writes particle locations, and executes the MODPATH model.

        Args:
            self (object): Instance of ModflowRunner.

        """
        self._transform_particle_coords()
        if self.timeStep == 1:
            self._initialize_modpath_objects()
            self._write_modpath_input()
        self._modify_mpsim_file()
        self._write_particle_location()
        self.mp.run_model(silent=True)

    def _transform_particle_coords(self):
        """Transform particle coordinates by mirroring along the X-axis.

        Args:
            self (object): Instance of ModflowRunner.

        """
        self.particleCoords[0] = self.extentX - self.particleCoords[0]

    def _initialize_modpath_objects(self):
        """Create MODPATH simulation objects on the first time step.

        Initializes MODPATH objects including the MODPATH model, basic
        package, and simulation with specified settings.

        Args:
            self (object): Instance of ModflowRunner.

        """
        self.mp = Modpath(
            self.MODELNAME,
            exe_name=self.exe_mp,
            modflowmodel=self.mf,
            model_ws=self.modelpth
        )
        self.mpbas = ModpathBas(
            self.mp,
            hnoflo=self.mf.bas6.hnoflo,
            hdry=self.mf.lpf.hdry,
            ibound=self.mf.bas6.ibound.array,
            prsity=0.2,
            prsityCB=0.2
        )
        self.sim = self.mp.create_mpsim(
            trackdir='forward',
            simtype='pathline',
            packages='RCH'
        )

    def _write_modpath_input(self):
        """Write MODPATH input files to the model workspace.

        Args:
            self (object): Instance of ModflowRunner.

        """
        self.mp.write_input()

    def _modify_mpsim_file(self):
        """Modify the .mpsim file to include custom particle location settings.

        Reads the existing .mpsim file lines, inserts custom particle settings,
        updates particle list lines, and writes the modified lines back.

        Args:
            self (object): Instance of ModflowRunner.

        """
        lines = self._read_mpsim_lines()
        lines = self._insert_custom_particle_settings(lines)
        lines = self._update_mplst_line(lines)
        self._write_mpsim_lines(lines)

    def _write_particle_location(self):
        """Write the current particle location to the .mploc file.

        The particle location is converted to cell indices and fractional
        positions within the cell, then written to the .mploc file in the
        required MODPATH format.

        Args:
            self (object): Instance of ModflowRunner.

        """
        l, c, r = self.cellInfoFromCoordinates(self.particleCoords)

        frac_col = 1.0 - ((self.particleCoords[0] / self.dCol) -
                          int(self.particleCoords[0] / self.dCol))
        frac_row = ((self.particleCoords[1] / self.dRow) -
                    int(self.particleCoords[1] / self.dRow))
        frac_ver = self.particleCoords[2] / self.dVer

        mploc_path = join(self.modelpth, self.MODELNAME + '.mploc')
        with open(mploc_path, 'w', encoding='utf-8') as f:
            f.write('1\n1\nparticle\n1\n')
            f.write(
                f"1 1 1 {self.nLay - l + 1} {self.nRow - r + 1} "
                f"{self.nCol - c + 1} {frac_col:.6f} {frac_row:.6f} "
                f"{frac_ver:.6f} 0.000000 particle\n"
            )
            f.write('particle\n')
            f.write('1 0.000000 1\n')

    def _read_mpsim_lines(self):
        """Read and return all lines from the .mpsim file.

        Args:
            self (object): Instance of ModflowRunner.

        Returns:
            list[str]: Lines read from the .mpsim file.

        """
        path = join(self.modelpth, f'{self.MODELNAME}.mpsim')
        with open(path, 'r', encoding='utf-8') as f:
            return f.readlines()

    def _write_mpsim_lines(self, lines):
        """Write modified lines back to the .mpsim file.

        Args:
            self (object): Instance of ModflowRunner.
            lines (list[str]): Lines to write to the .mpsim file.

        """
        path = join(self.modelpth, f'{self.MODELNAME}.mpsim')
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    def _insert_custom_particle_settings(self, lines):
        """Modify lines to insert custom particle location settings.

        Args:
            lines (list of str): Lines from the input file to modify.

        Returns:
            list of str: Modified lines with custom particle settings inserted.

        """
        out = []
        keep = True
        for line in lines:
            if 'rch' in line.lower():
                keep = False
                out.append(f"{self.MODELNAME}.mploc\n")
                out.append("2\n")  # Particle generation option 2, budget output 3
                # Safely remove line 7 if it exists
                if len(out) > 7:
                    del out[7]
                out.append("0.000000   1.000000\n")  # TimePoints
            if keep:
                out.append(line)
        return out

    def _update_mplst_line(self, lines):
        """Update the line after 'mplst' with specific settings.

        Args:
            lines (list of str): Lines from the input file.

        Returns:
            list of str: Lines with updated mplst settings.

        """
        for i, line in enumerate(lines):
            if 'mplst' in line.lower() and i + 1 < len(lines):
                lines[i + 1] = ('2   1   2   1   1   2   2   3   1   1   1   1\n')
                break
        return lines

    def evaluateParticleTracking(self):
        """Evaluate particle tracking results from MODPATH.

        Loads particle data, filters by time, appends trajectories,
        and updates particle coordinates.
        
        Args:
            self (object): Instance of the class.

        """
        data = self._loadParticleData()
        valid = self._filterByTime(data)
        self._appendTrajectories(data, valid)
        self._updateParticleCoords(data, valid)

    def _loadParticleData(self):
        """Load particle data from the MODPATH pathline file.

        Returns:
            pandas.DataFrame: Particle data with partid=0.

        """
        path = join(self.modelpth, self.mp.sim.pathline_file)
        return PathlineFile(path).get_data(partid=0)


    def _filterByTime(self, data):
        """Filter particle data by time within the period length.

        Args:
            data (pandas.DataFrame): Particle data including 'time' column.

        Returns:
            pandas.Series: Boolean mask of valid data within periodLength.

        Raises:
            ValueError: If no data points are within periodLength.

        """
        valid = data['time'] <= self.periodLength
        if not valid.any():
            raise ValueError("No particle data within periodLength.")
        return valid

    def _appendTrajectories(self, data, valid):
        """Append valid particle coordinates to trajectories.

        Args:
            data (pandas.DataFrame): Particle data.
            valid (pandas.Series): Boolean mask for valid data points.

        """
        for coord in 'xyz':
            self.trajectories[coord].append(data[coord][valid])


    def _updateParticleCoords(self, data, valid):
        """Update particle coordinates to last valid position.

        Args:
            data (pandas.DataFrame): Particle data.
            valid (pandas.Series): Boolean mask for valid data points.

        """
        self.particleCoords = copy([data[c][valid][-1] for c in 'xyz'])

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
        diff = targetField - realizedField
        reward = -1.0 * numpySum(numpyAbs(diff))
        return reward

    def calculateGameRewardOperationCost(self):
        """Calculate reward based on operation costs from pump and recharge flows.

        Returns:
            float: Computed reward based on summed flows.

        """
        pump, recharge = self._sum_flows()
        return self._compute_reward(pump, recharge)

    def _sum_flows(self):
        """Sum pump and recharge flows from helper wells.

        Returns:
            tuple: (pump, recharge) total flow values.

        """
        pump = recharge = 0.0
        for i in range(self.nHelperWells):
            Q = self.helperWells[f'wellQ{i + 1}']
            if Q <= 0:
                pump += -Q
            else:
                recharge += Q
        return pump, recharge

    def _compute_reward(self, pump, recharge):
        """Compute reward based on pump and recharge costs.

        Args:
            pump (float): Total pump flow.
            recharge (float): Total recharge flow.

        Returns:
            float: Negative cost as reward.

        """
        return -pump * self.costQm3dPump - recharge * self.costQm3dRecharge

    def reset(self, _seed=None, MODELNAME=None, initWithSolution=None):
        """Reset environment with same settings and optional new seed.

        Args:
            _seed (int, optional): Random seed for environment reset.
            MODELNAME (str, optional): Model name for reset.
            initWithSolution (optional): Initial solution to use.

        Returns:
            tuple: (observationsVectorNormalized as numpy array, empty dict)

        """
        initWithSolution = self._get_init_solution(initWithSolution)

        if self.ENVTYPE == '0s-c':
            self._reset_0s_c(_seed)
        else:
            self._reset_other(_seed, MODELNAME, initWithSolution)

        return array(self.observationsVectorNormalized), {}

    def _get_init_solution(self, initWithSolution):
        """Determine initial solution to use for reset.

        Args:
            initWithSolution (optional): Provided initial solution.

        Returns:
            Initial solution to use.

        """
        return self.initWithSolution if initWithSolution is None else initWithSolution

    def _reset_0s_c(self, _seed):
        """Reset environment for '0s-c' ENVTYPE.

        Args:
            _seed (int, optional): Seed for environment setup.

        """
        self.model_ws = join(self.wrkspc, 'models', self.exdir)
        self.defineEnvironment(_seed)
        self.constructModel()

    def _reset_other(self, _seed, MODELNAME, initWithSolution):
        """Reset environment for ENVTYPE other than '0s-c'.

        Args:
            _seed (int or None): Random seed for environment.
            MODELNAME (str or None): Model name override.
            initWithSolution (optional): Initial solution to use.
        
        """
        env_config = {
            'ENVTYPE': self.ENVTYPE,
            'PATHMF2005': self.PATHMF2005,
            'PATHMP6': self.PATHMP6,
            'MODELNAME': self.MODELNAME if MODELNAME is None else MODELNAME,
            '_seed': _seed,
            'flagSavePlot': self.SAVEPLOT,
            'flagManualControl': self.MANUALCONTROL,
            'flagRender': self.RENDER,
            'nLay': self.nLay,
            'nRow': self.nRow,
            'nCol': self.nCol,
            'OBSPREP': self.OBSPREP,
            'initWithSolution': initWithSolution,
        }
        self.__init__(env_config=env_config)
        close()

    def render(self, mode='human', dpi=120):
        """Plot the simulation state at the current timestep.

        Displays and/or saves the visualization. The active display can take
        user input from the keyboard to control the environment.

        Args:
            mode (str): Render mode, 'human' or 'rgb_array'.
            dpi (int): Resolution of the plot.

        Returns:
            Optional: RGB array of the figure if mode is 'rgb_array'.
        
        """
        return_figure = (mode == 'rgb_array') or self.SAVEPLOT

        if self.ENVTYPE == '0s-c':
            self.render_envtype_0s_c(return_figure, mode, dpi)
        else:
            self.render_envtype_other(return_figure, dpi)

    def fig_to_rgb_array(self, fig):
        """Convert a Matplotlib figure canvas to an RGB numpy array.

        Args:
            fig (matplotlib.figure.Figure): Figure to convert.

        Returns:
            numpy.ndarray: RGB array of the figure image.

        """
        fig.canvas.draw()
        data = fig.canvas.tostring_argb()
        # data = fig.canvas.tostring_rgb()
        width, height = fig.canvas.get_width_height()
        return copy(frombuffer(data, dtype=uint8).reshape(height, width, 4))

    def _init_figure(self):
        """Initialize the figure and axis for plotting.

        Args:
            self (object): The instance to initialize figure and axis attributes.

        """
        self.fig = figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(1, 1, 1, aspect='equal', adjustable='box')
        self.plotArrays = []

    def _load_grid(self):
        """Load grid file and initialize model grid and PlotMapView.

        Returns:
            None

        """
        fname = join(self.model_ws, f'{self.MODELNAME}.dis.grb')
        grd = MfGrdFile(fname, verbose=False)
        mg = grd.modelgrid
        self.modelmap = PlotMapView(modelgrid=mg, ax=self.ax)

    def _plot_base_map(self):
        """Plot the base grid map with light transparency.

        Args:
            self (object): The instance containing the model map to plot.

        """
        self.modelmap.plot_grid(alpha=0.1)

    def _update_title(self):
        """Set the plot title with current step and reward.

        Args:
            self (object): The instance of the class containing plot attributes.

        """
        self.ax.cla()
        self.ax.set_title(
            f'step {self.timeStep}, reward {self.rewardCurrent:.2f}'
        )

    def _load_head_file(self):
        """Load the head file (.hds); silently ignore if loading fails.

        Returns:
            bf.HeadFile or None: HeadFile object if loaded successfully, else None.
        
        """
        try:
            fname = join(self.model_ws, f'{self.MODELNAME}.hds')
            heads = bf.HeadFile(fname)
            return heads
        except Exception:
            return None

    def _plot_storm_centers(self):
        """Plot storm centers if current timestep matches a storm start.

        Args:
            self (object): The instance containing storm data and plot axes.
        
        """
        if self.timeStep in self.stormStarts:
            i = self.stormStarts.index(self.timeStep)
            self.ax.scatter(
                self.stormCentersX[i],
                self.stormCentersY[i],
                color='red',
                s=10,
                zorder=10
            )

    def _plot_head_array(self):
        """Plot the head array on the map with colorbar.

        Args:
            self (object): The instance containing head data and plot attributes.

        Returns:
            matplotlib.colorbar.Colorbar: The colorbar instance for the plot.
        
        """
        head = self.head.reshape((self.nlay, self.nrow, self.ncol))
        pa = self.modelmap.plot_array(
            head, vmax=self.chd_west, vmin=self.chd_east
        )
        cb = self.fig.colorbar(pa, ax=self.ax)
        return cb

    def _plot_helper_wells(self):
        """Plot helper wells with color coding and labels.

        Args:
            self (object): The instance containing helper well data and plot axes.
        
        """
        for i in range(self.nHelperWells):
            x = self.actionsDict[f'well{i}x']
            y = self.actionsDict[f'well{i}y']
            Q = self.actionsDict[f'well{i}Q']
            color = 'blue' if Q > 0 else 'red'
            self.ax.scatter(x, y, color=color, s=5, zorder=10)
            self.ax.text(
                x + 3, y, f'{Q:.2f}', fontsize=12, color=color,
                zorder=12, alpha=0.5
            )

    def _display_plot(self, cb):
        """Display the plot for human viewing and remove the colorbar.

        Args:
            self (object): The instance controlling the plot display.
            cb: The colorbar instance to remove after display.
        
        """
        show(block=False)
        timeout = 2.5 if self.timeStep in self.stormStarts else 0.5
        waitforbuttonpress(timeout=timeout)
        cb.remove()

    def _handle_image_array(self, cb, dpi=120):
        """Convert figure to image array and handle saving if needed.

        Args:
            self (object): The instance containing figure and plot settings.
            cb: The colorbar instance to remove after processing.
            dpi (int, optional): Resolution for saving images. Defaults to 120.

        Returns:
            numpy.ndarray: The RGB image array of the figure.
        
        """
        imarray = self.fig_to_rgb_array(self.fig)

        self._ensure_plot_dirs()
        plotfile = self._get_plot_filepath()
        self.plotArrays.append(imarray)

        if self.SAVEPLOT:
            imsave(plotfile, imarray, dpi=dpi)
            self._save_gif_if_needed()

        cb.remove()
        return imarray

    def _save_gif_if_needed(self):
        """Append image to plotArrays and save GIF if done or at last step.

        Args:
            self (object): The instance containing plot data and configuration.
        
        """
        if self.done or self.timeStep == self.nstp:
            path = join(
                self.wrkspc,
                'runs',
                self.ANIMATIONFOLDER,
                f'{self.MODELNAMEGENCOUNT}.gif'
            )
            self.writeGIFtodisk(path, self.plotArrays, optimizeSize=True)

    def _handle_display_or_return(self, cb, return_figure, dpi):
        """Show plot or return RGB array, handle saving if needed.

        Args:
            self (object): The instance controlling plot display and saving.
            cb: The colorbar instance to remove after processing.
            return_figure (bool): If True, return the RGB array instead of showing.
            dpi (int): Resolution for saving images.

        Returns:
            numpy.ndarray or None: RGB array if return_figure is True, else None.
        
        """
        if not return_figure:
            self._display_plot(cb)
            return None  # Explicitly return None for clarity
        else:
            return self._handle_image_array(cb, dpi)

    def render_envtype_0s_c(self, return_figure, mode, dpi):
        """Render method part for ENVTYPE == '0s-c'.

        Args:
            self (object): The instance containing environment and plot state.
            return_figure (bool): Whether to return the figure as an image array.
            mode: Unused parameter (consider removing if not needed).
            dpi (int): Resolution for saving images.

        Returns:
            numpy.ndarray or None: RGB array if return_figure is True, else None.
        
        """
        if self.timeStep == 0:
            self._init_figure()

        self._update_title()
        self._load_grid()
        self._plot_base_map()

        self.hds = self._load_head_file()  # Optional head file loading
        self._plot_storm_centers()
        cb = self._plot_head_array()
        self._plot_helper_wells()

        return self._handle_display_or_return(cb, return_figure, dpi)

    def render_envtype_other(self, return_figure, dpi):
        """Render method for other environment types.

        Args:
            self (object): The instance managing the environment and plotting.
            return_figure (bool): Whether to return the figure as an image array.
            dpi (int): Resolution for saving images.

        Returns:
            numpy.ndarray or None: RGB array if return_figure is True, else None.
        
        """
        self._init_canvas_if_needed()
        self._plot_environment()

        if return_figure:
            return self._handle_return_figure(dpi)
        else:
            self._handle_display_mode()

    def _init_canvas_if_needed(self):
        """Initialize canvas and reset plotting arrays if first timestep or 
        if not initialized.

        Args:
            self (object): The instance managing canvas and plotting state.
        
        """
        if self.timeStep == 0 or not self.canvasInitialized:
            self.renderInitializeCanvas()
            self.plotfilesSaved = []
            self.plotArrays = []
            self.extent = (
                self.dRow / 2,
                self.extentX - self.dRow / 2,
                self.extentY - self.dCol / 2,
                self.dCol / 2,
            )
            self.fig.canvas.draw()

    def _plot_environment(self):
        """Plot map and all overlays for the non '0s-c' environment.

        Args:
            self (object): The instance containing environment and plotting data.
        
        """
        self.modelmap = PlotMapView(model=self.mf, layer=0)
        self.headsplot = self.modelmap.plot_array(
            self.heads,
            masked_values=[999.0],
            alpha=0.5,
            zorder=2,
            cmap=colormaps['terrain'] # get_cmap('terrain')
        )
        self.quadmesh = self.modelmap.plot_ibound(zorder=3)
        self.renderWellSafetyZone(zorder=3)
        self.renderContourLines(n=30, zorder=4)
        self.renderIdealParticleTrajectory(zorder=5)
        self.renderTextOnCanvasPumpingRate(zorder=10)
        self.renderTextOnCanvasGameOutcome(zorder=10)
        self.renderParticle(zorder=6)
        self.renderParticleTrajectory(zorder=6)
        self.renderTextOnCanvasTimeAndScore(zorder=10)
        self.renderRemoveAxesTicks()
        self.renderSetAxesLimits()
        self.renderAddAxesTextLabels()

        for ax in [self.ax, self.ax2, self.ax3]:
            ax.axis('off')

    def _prepare_figure_for_return(self, dpi):
        """Prepare figure for RGB array return: remove axes, set size and dpi.

        Args:
            self (object): The instance containing the figure to prepare.
            dpi (int): The resolution to set for the figure.

        Returns:
            tuple: Original figure size in inches (width, height).
        
        """
        s = self.fig.get_size_inches()
        ax = self.fig.gca()
        ax.set_axis_off()
        margins(0, 0)
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        self.fig.tight_layout(pad=0)
        self.fig.set_size_inches(7, 7)
        self.fig.set_dpi(dpi)
        return s

    def _restore_figure_size(self, size):
        """Restore figure size after RGB array extraction.

        Args:
            self (object): The instance containing the figure to resize.
            size (tuple or list): The original figure size in inches.
        
        """
        self.fig.set_size_inches(size)

    def _handle_return_figure(self, dpi):
        """Handle rendering when return_figure is True.

        Args:
            self (object): The instance managing figure rendering and saving.
            dpi (int): Resolution for saving images.

        Returns:
            numpy.ndarray: RGB array of the rendered figure.
        
        """
        size = self._prepare_figure_for_return(dpi)
        imarray = self.fig_to_rgb_array(self.fig)
        self._restore_figure_size(size)

        self.plotArrays.append(imarray)

        if self.SAVEPLOT:
            self.renderSavePlot()
            if self.done or self.timeStep == self.NAGENTSTEPS:
                pathGIF = join(
                    self.wrkspc,
                    'runs',
                    self.ANIMATIONFOLDER,
                    f'{self.MODELNAME}.gif'
                )
                self.writeGIFtodisk(pathGIF, self.plotArrays, optimizeSize=True)

        self.renderClearAxes()
        return imarray

    def _handle_display_mode(self):
        """Handle rendering when return_figure is False.

        Args:
            self (object): The instance managing interactive and display rendering.
        
        """
        if self.MANUALCONTROL:
            self.renderUserInterAction()
        elif self.RENDER:
            self.fig.canvas.draw()
            show(block=False)
            pause(self.MANUALCONTROLTIME)
        self.renderClearAxes()

    def renderInitializeCanvas(self):
        """Initialize plot canvas with figure and axes.

        Args:
            self (object): The instance managing the plot canvas initialization.
        
        """
        self._create_figure()
        self._detect_ipython()
        if not self.flagFromIPythonNotebook:
            if self.RENDER or self.MANUALCONTROL:
                self._maximize_figure_window()
        self._setup_axes()
        self._finalize_figure()
        self.canvasInitialized = True

    def _create_figure(self):
        """Create a new figure with a fixed size.

        Args:
            self (object): The instance where the figure will be created.
        
        """
        self.fig = figure(figsize=(7, 7))

    def _detect_ipython(self):
        """Detect if running inside an IPython notebook environment.
        
        Args:
            self (object): The instance where the flag will be set.
        
        """
        self.flagFromIPythonNotebook = 'ipykernel' in modules

    def _maximize_figure_window(self):
        """Maximize the figure window if possible.

        Attempts to maximize the figure window using platform-specific methods.
        Falls back gracefully if methods are unsupported.

        Args:
            self (object): The instance containing the figure manager.
        
        """
        self.figManager = get_current_fig_manager()
        try:
            self.figManager.window.state('zoomed')
        except Exception:
            try:
                self.figManager.full_screen_toggle()
            except Exception:
                pass

    def _setup_axes(self):
        """Set up the main axes and twin axes for the figure.

        Args:
            self (object): The instance containing the figure to set up axes on.
        
        """
        self.ax = self.fig.add_subplot(1, 1, 1, aspect='equal', adjustable='box')
        self.ax3 = self.ax.twinx()
        self.ax2 = self.ax3.twiny()

    def _finalize_figure(self):
        """Finalize the figure by removing axes and adjusting layout.

        Args:
            self (object): The instance containing the figure to finalize.
        
        """
        ax = self.fig.gca()
        ax.set_axis_off()
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        self.fig.tight_layout(pad=0.)

    def renderIdealParticleTrajectory(self, zorder=5):
        """Plot ideal particle trajectory associated with maximum reward.

        Args:
            self (object): The instance containing plotting axes and particle data.
            zorder (int, optional): Drawing order for the plot. Defaults to 5.
        
        """
        self.ax2.plot(
            [self.minX, self.minX + self.extentX],
            [self.particleY, self.particleY],
            lw=1.5,
            c='white',
            ls='--',
            zorder=zorder,
            alpha=0.5
        )

    def renderContourLines(self, n=30, zorder=4):
        """Plot n contour lines of the head field.

        Args:
            self (object): The instance containing head data and plotting methods.
            n (int, optional): Number of contour levels. Defaults to 30.
            zorder (int, optional): Drawing order for contours. Defaults to 4.
        
        """
        self.levels = linspace(min(self.heads), max(self.heads), n)
        self.contours = self.modelmap.contour_array(
            self.heads,
            levels=self.levels,
            alpha=0.5,
            zorder=zorder
        )

    def renderWellSafetyZone(self, zorder=3):
        """Plot well safety zone for main and helper wells.

        Args:
            self (object): The instance containing well data and plotting methods.
            zorder (int, optional): Drawing order for the safety zones. Defaults to 3.
        
        """
        self._draw_well_circle(self.wellCoords, self.wellRadius, self.wellQ, self.minQ, zorder)
        self._render_helper_wells(zorder)

    def _draw_well_circle(self, coords, radius, quantity, min_quantity, zorder=3):
        """Draw a well safety zone circle with outline and filled alpha.

        Args:
            self (object): The instance managing plotting and well data.
            coords (tuple): Coordinates of the well center.
            radius (float): Radius of the safety zone circle.
            quantity (float): Well quantity to determine color and alpha.
            min_quantity (float): Minimum quantity for alpha scaling.
            zorder (int, optional): Drawing order. Defaults to 3.
        
        """
        color = self.renderGetWellColor(quantity)
        center = self._circle_coords(coords)

        outline = self._create_outline_circle(center, radius, color, zorder)
        self.ax2.add_artist(outline)

        alpha = self.renderGetWellCircleAlpha(quantity, min_quantity)
        filled = self._create_filled_circle(center, radius, color, alpha, zorder)
        self.ax2.add_artist(filled)

    def _circle_coords(self, coords):
        """Convert input coordinates to plot coordinates.

        Args:
            self (object): The instance containing extentY for coordinate transformation.
            coords (tuple): Original (x, y) coordinates.

        Returns:
            tuple: Transformed (x, y) coordinates for plotting.
        
        """
        x, y = coords[0], self.extentY - coords[1]
        return x, y

    def _create_outline_circle(self, center, radius, color, zorder):
        """Create an outline circle patch.

        Args:
            self (object): The instance creating the circle patch.
            center (tuple): Coordinates of the circle center.
            radius (float): Radius of the circle.
            color: Color of the circle outline.
            zorder (int): Drawing order.

        Returns:
            matplotlib.patches.Circle: A Circle patch object.
        
        """
        return Circle(
            center,
            radius,
            edgecolor=color,
            facecolor='none',
            fill=False,
            zorder=zorder,
            alpha=1.0,
            lw=1.0
        )

    def _create_filled_circle(self, center, radius, color, alpha, zorder):
        """Create a filled circle patch with transparency.

        Args:
            self (object): The instance creating the circle patch.
            center (tuple): Coordinates of the circle center.
            radius (float): Radius of the circle.
            color: Fill color of the circle.
            alpha (float): Transparency level of the fill.
            zorder (int): Drawing order.

        Returns:
            matplotlib.patches.Circle: A filled Circle patch object.
        
        """
        return Circle(
            center,
            radius,
            edgecolor='none',
            facecolor=color,
            fill=True,
            zorder=zorder,
            alpha=alpha
        )

    def _render_helper_wells(self, zorder=3):
        """Render all helper wells safety zones.

        Args:
            self (object): The instance containing helper wells data and plotting methods.
            zorder (int, optional): Drawing order for the safety zones. Defaults to 3.
        
        """
        valid_envtypes = [
            '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost',
            '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c'
        ]
        if self.ENVTYPE not in valid_envtypes:
            return

        for i in range(self.nHelperWells):
            idx = i + 1
            coords = self.helperWells.get(f'wellCoords{idx}')
            quantity = self.helperWells.get(f'wellQ{idx}')
            if coords is not None and quantity is not None:
                self._draw_well_circle(coords, self.helperWellRadius, quantity, self.minQhelper, zorder)

    def renderTextOnCanvasPumpingRate(self, zorder=10):
        """Plot pumping rate on figure with curved text around wells.

        Args:
            self (object): The instance containing well data and rendering methods.
            zorder (int, optional): Drawing order for the text. Defaults to 10.
        
        """
        color = self.renderGetWellColor(self.wellQ)
        self.render_all_well_labels(color)

    def _generate_circle_coords(self, radius, num_points=500):
        """Generate x, y coordinates of a circle with given radius.

        Args:
            self (object): The instance calling this method.
            radius (float): Radius of the circle.
            num_points (int, optional): Number of points to generate. Defaults to 500.

        Returns:
            tuple: Arrays of x and y coordinates representing the circle.
        
        """
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = -radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y

    def _format_label(self, value, unit='m³', exponent='d⁻¹', delimiter=';'):
        """
        Format label string by inserting delimiters between characters and appending units.

        Example:
            123 -> '1;2;3;   ;m³;   ;d⁻¹'

        Args:
            self (object): The instance calling this method.
            value (int or float): Numeric value to format.
            unit (str, optional): Unit string to append. Defaults to 'm³'.
            exponent (str, optional): Exponent string to append. Defaults to 'd⁻¹'.
            delimiter (str, optional): Delimiter to insert between characters. Defaults to ';'.

        Returns:
            str: Formatted label string with delimiters and units.
        
        """
        val_str = str(int(abs(value)))
        # Insert delimiter between characters
        delimited = delimiter.join(val_str)
        label = f"{delimited}{delimiter}   {delimiter}{unit}{delimiter}   {delimiter}{exponent}"
        return label

    def _add_curved_text(self, x, y, text, va, axes, delimiter,
                         fontsize, color):
        """Create and add a CurvedText object to follow a curve.

        Args:
            self (object): Object instance.
            x (np.ndarray): X coordinates of the curve.
            y (np.ndarray): Y coordinates of the curve.
            text (str): Text to display along the curve.
            va (str): Vertical alignment.
            axes (matplotlib.axes.Axes): Axes to add the text to.
            delimiter (str): Delimiter to split text into characters.
            fontsize (float): Font size for the text.
            color (str): Color of the text.
        
        """
        class CurvedText(mtext.Text):
            """A text object that follows an arbitrary curve."""

            def __init__(self, x, y, text, axes, delimiter, **kwargs):
                """Initialize CurvedText with characters positioned along a curve.

                Args:
                    self (object): Instance of CurvedText.
                    x (np.ndarray): X coordinates of the curve.
                    y (np.ndarray): Y coordinates of the curve.
                    text (str): Text to display along the curve.
                    axes (matplotlib.axes.Axes): Axes to add text artists to.
                    delimiter (str): Delimiter to split text into characters.
                    **kwargs: Additional keyword arguments for text properties.
                
                """
                super().__init__(x[0], y[0], ' ', **kwargs)
                axes.add_artist(self)
                self.axes = axes
                self.__x = x
                self.__y = y
                self.delimiter = delimiter
                self.__Characters = []
                for c in text.split(delimiter):
                    char = 'a' if c == ' ' else c
                    alpha = 0.0 if c == ' ' else 1.0
                    t = mtext.Text(0, 0, char, alpha=alpha, **kwargs)
                    self.__Characters.append((c, t))
                    axes.add_artist(t)

            def set_zorder(self, zorder):
                """Set the z-order for the main object and its character texts.

                Args:
                    self (object): Instance of the class.
                    zorder (int): Base drawing order for the main object.
                
                """
                super().set_zorder(zorder)
                for _, text in self.__Characters:
                    text.set_zorder(zorder + 1)

            def draw(self, renderer, *args, **kwargs):
                """Draw the object by updating character positions along the curve.

                Args:
                    self (object): Instance of the class.
                    renderer (RendererBase): Matplotlib renderer instance.
                    *args: Additional positional arguments.
                    **kwargs: Additional keyword arguments.
                
                """
                self.update_positions(renderer)

            def update_positions(self, renderer):
                """Update character positions and orientations along the curve.

                Args:
                    self (object): Instance of the class.
                    renderer (RendererBase): Matplotlib renderer instance.
                
                """
                aspect = self._compute_aspect_ratio()
                x_fig, y_fig = self._get_curve_figure_coords()
                cum_lengths, seg_angles, seg_degs = self._compute_curve_metrics(
                    x_fig, y_fig
                )
                rel_pos = 10
                offset_dist = 0.6
                extra_space = 2
                for char, txt in self.__Characters:
                    rel_pos = self._process_character(
                        txt, renderer, rel_pos, cum_lengths, seg_angles,
                        seg_degs, aspect, offset_dist, extra_space
                    )

            def _compute_aspect_ratio(self):
                """Compute the axes aspect ratio for coordinate transformation.

                Args:
                    self (object): Instance of the class.

                Returns:
                    float: Aspect ratio for scaling between data and figure coordinates.
                
                """
                xlim = self.axes.get_xlim()
                ylim = self.axes.get_ylim()
                fig_w, fig_h = self.axes.get_figure().get_size_inches()
                _, _, w, h = self.axes.get_position().bounds
                return ((fig_w * w) / (fig_h * h)) * (
                    (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
                )

            def _get_curve_figure_coords(self):
                """Transform curve data coordinates to figure coordinates.

                Args:
                    self (object): Instance of the class.

                Returns:
                    tuple: Arrays of x and y coordinates in figure space.
                
                """
                coords = list(zip(self.__x, self.__y))
                x_fig, y_fig = zip(*self.axes.transData.transform(coords))
                return np.array(x_fig), np.array(y_fig)

            def _compute_curve_metrics(self, x_fig, y_fig):
                """Compute curve segment metrics for positioning text along a curve.

                Args:
                    self (object): Instance of the class.
                    x_fig (np.ndarray): X coordinates in figure space.
                    y_fig (np.ndarray): Y coordinates in figure space.

                Returns:
                    tuple:
                        cum_lengths (np.ndarray): Cumulative lengths along the curve.
                        seg_angles (np.ndarray): Segment angles in radians.
                        seg_degs (np.ndarray): Segment angles in degrees.
                
                """
                seg_dist = np.sqrt(np.diff(x_fig) ** 2 + np.diff(y_fig) ** 2)
                cum_lengths = np.insert(np.cumsum(seg_dist), 0, 0)
                seg_angles = np.arctan2(np.diff(y_fig), np.diff(x_fig))
                seg_degs = np.rad2deg(seg_angles)
                return cum_lengths, seg_angles, seg_degs

            def _advance_rel_pos(self, rel_pos, width):
                """Advance the relative position by the character width.

                Args:
                    self (object): Instance of the class.
                    rel_pos (float): Current relative position along the curve.
                    width (float): Width of the character.

                Returns:
                    float: Updated relative position.
                
                """
                return rel_pos + width

            def _process_character(
                self, txt, renderer, rel_pos, cum_lengths, seg_angles, seg_degs,
                aspect, offset_dist, extra_space
            ):
                """Position and orient a character along the curve.

                Args:
                    self (object): Instance of the class.
                    txt (Text): Matplotlib text object for the character.
                    renderer (RendererBase): Matplotlib renderer instance.
                    rel_pos (float): Current position along the curve.
                    cum_lengths (np.ndarray): Cumulative lengths along the curve.
                    seg_angles (np.ndarray): Segment angles in radians.
                    seg_degs (np.ndarray): Segment angles in degrees.
                    aspect (float): Aspect ratio for scaling.
                    offset_dist (float): Offset distance from the curve.
                    extra_space (float): Extra spacing after the character.

                Returns:
                    float: Updated relative position for the next character.
                
                """
                self._reset_text_rotation_and_va(txt)
                width = self._get_text_width(txt, renderer)
                if self._should_hide_text(rel_pos, width, cum_lengths):
                    self._hide_text(txt)
                    return self._advance_rel_pos(rel_pos, width)
                self._show_text(txt)
                il, ir = self._find_segment_indices(cum_lengths, rel_pos, width)
                used = self._compute_used_length(cum_lengths, il, rel_pos)
                frac = self._compute_fraction(width, used, cum_lengths, il)
                x, y = self._interpolate_position(il, ir, frac)
                txt.set_va(self.get_va())
                bbox1, bbox2 = self._get_adjusted_bbox(txt, renderer)
                dr = self._compute_displacement(bbox1, bbox2)
                drp = self._rotate_displacement(dr, seg_angles[il], aspect)
                base_pos = self._compute_base_position(x, y, drp)
                normal = self._compute_normal(seg_angles[il], aspect)
                pos = self._apply_offset(base_pos, normal, offset_dist)
                self._set_text_properties(txt, pos, seg_degs[il])
                return self._update_rel_pos(rel_pos, width, used, extra_space)

            def _reset_text_rotation_and_va(self, txt):
                """Reset text rotation and vertical alignment to default.

                Args:
                    self (object): Instance of the class.
                    txt (Text): Matplotlib text object to reset.
                
                """
                txt.set_rotation(0)
                txt.set_va('center')

            def _get_text_width(self, txt, renderer):
                """Return the width of a text object in display units.

                Args:
                    self (object): Instance of the class.
                    txt (Text): Matplotlib text object.
                    renderer (RendererBase): Matplotlib renderer instance.

                Returns:
                    float: Width of the text in display units (pixels).
                
                """
                bbox = txt.get_window_extent(renderer=renderer)
                return bbox.width

            def _should_hide_text(self, rel_pos, width, cum_lengths):
                """Determine if the text should be hidden based on position.

                Args:
                    self (object): Instance of the class.
                    rel_pos (float): Current position along the curve.
                    width (float): Width of the character.
                    cum_lengths (np.ndarray): Cumulative lengths along the curve.

                Returns:
                    bool: True if the character should be hidden, False otherwise.
                
                """
                return rel_pos + width / 2 > cum_lengths[-1]

            def _hide_text(self, txt):
                """Hide the given text object by setting its alpha to zero.

                Args:
                    self (object): Instance of the class.
                    txt (Text): Matplotlib text object to hide.
                
                """
                txt.set_alpha(0.0)

            def _show_text(self, txt):
                """Show the given text object by setting its alpha to one.

                Args:
                    self (object): Instance of the class.
                    txt (Text): Matplotlib text object to show.
                
                """
                txt.set_alpha(1.0)

            def _find_segment_indices(self, cum_lengths, rel_pos, width):
                """Find segment indices for placing a character along the curve.

                Args:
                    self (object): Instance of the class.
                    cum_lengths (np.ndarray): Cumulative lengths along the curve.
                    rel_pos (float): Current position along the curve.
                    width (float): Width of the character.

                Returns:
                    tuple: Indices (il, ir) of the curve segment for the character.
                
                """
                center_pos = rel_pos + width / 2
                il = np.where(center_pos >= cum_lengths)[0][-1]
                ir = np.where(center_pos <= cum_lengths)[0][0]
                if ir == il:
                    ir += 1
                return il, ir

            def _compute_used_length(self, cum_lengths, il, rel_pos):
                """Compute the length already used along the curve for character placement.

                Args:
                    self (object): Instance of the class.
                    cum_lengths (np.ndarray): Cumulative lengths along the curve.
                    il (int): Index of the left segment.
                    rel_pos (float): Current position along the curve.

                Returns:
                    float: Length already used along the curve from rel_pos to segment il.
                
                """
                return cum_lengths[il] - rel_pos

            def _compute_fraction(self, width, used, cum_lengths, il):
                """Compute the fractional position of a character within a curve segment.

                Args:
                    self (object): Instance of the class.
                    width (float): Width of the character.
                    used (float): Length already used along the curve.
                    cum_lengths (np.ndarray): Cumulative lengths along the curve.
                    il (int): Index of the left segment.

                Returns:
                    float: Fractional position within the segment for character placement.
                
                """
                seg_len = cum_lengths[il + 1] - cum_lengths[il]
                return (width / 2 - used) / seg_len

            def _interpolate_position(self, il, ir, frac):
                """Interpolate the (x, y) position for a character along the curve.

                Args:
                    self (object): Instance of the class.
                    il (int): Index of the left segment.
                    ir (int): Index of the right segment.
                    frac (float): Fractional distance between il and ir.

                Returns:
                    tuple: Interpolated (x, y) coordinates.
                
                """
                x = self.__x[il] + frac * (self.__x[ir] - self.__x[il])
                y = self.__y[il] + frac * (self.__y[ir] - self.__y[il])
                return x, y

            def _get_adjusted_bbox(self, txt, renderer):
                """Get the bounding boxes of a text object before and after alignment.

                Args:
                    self (object): Instance of the class.
                    txt (Text): Matplotlib text object.
                    renderer (RendererBase): Matplotlib renderer instance.

                Returns:
                    tuple: Bounding boxes (bbox1, bbox2) before and after vertical alignment.
                
                """
                bbox1 = txt.get_window_extent(renderer=renderer)
                txt.set_va(self.get_va())
                bbox2 = txt.get_window_extent(renderer=renderer)
                return bbox1, bbox2

            def _compute_displacement(self, bbox1, bbox2):
                """Compute the displacement vector between two bounding boxes in data coords.

                Args:
                    self (object): Instance of the class.
                    bbox1 (Bbox): First bounding box (before alignment).
                    bbox2 (Bbox): Second bounding box (after alignment).

                Returns:
                    np.ndarray: Displacement vector from bbox1 to bbox2 in data coordinates.
                
                """
                pts1 = bbox1.get_points()
                pts2 = bbox2.get_points()
                pts1_data = self.axes.transData.inverted().transform(pts1)
                pts2_data = self.axes.transData.inverted().transform(pts2)
                return pts2_data[0] - pts1_data[0]

            def _rotate_displacement(self, dr, rad, aspect):
                """Rotate a displacement vector by a given angle and aspect ratio.

                Args:
                    self (object): Instance of the class.
                    dr (np.ndarray): Displacement vector to rotate.
                    rad (float): Rotation angle in radians.
                    aspect (float): Aspect ratio for scaling.

                Returns:
                    np.ndarray: Rotated displacement vector.
                
                """
                rot_mat = self._rotation_matrix(rad, aspect)
                return np.dot(dr, rot_mat)

            def _compute_base_position(self, x, y, drp):
                """Compute the base position for a character along the curve.

                Args:
                    self (object): Instance of the class.
                    x (float): X coordinate along the curve.
                    y (float): Y coordinate along the curve.
                    drp (np.ndarray): Displacement vector to adjust the base position.

                Returns:
                    np.ndarray: Adjusted (x, y) position as a NumPy array.
                
                """
                return np.array([x, y]) + drp

            def _compute_normal(self, rad, aspect):
                """Compute the normal vector at a point on the curve, accounting for aspect.

                Args:
                    self (object): Instance of the class.
                    rad (float): Angle (in radians) of the curve segment.
                    aspect (float): Aspect ratio for scaling the y-component.

                Returns:
                    np.ndarray: Normalized normal vector (2D) at the given angle.
                
                """
                normal = np.array([-np.sin(rad), np.cos(rad) * aspect])
                return normal / np.linalg.norm(normal)

            def _apply_offset(self, base_pos, normal, offset_dist):
                """Apply an offset to a base position along the normal vector.

                Args:
                    self (object): Instance of the class.
                    base_pos (np.ndarray): Base (x, y) position along the curve.
                    normal (np.ndarray): Normalized normal vector at the position.
                    offset_dist (float): Distance to offset from the base position.

                Returns:
                    np.ndarray: Offset (x, y) position as a NumPy array.
                
                """
                return base_pos + offset_dist * normal

            def _set_text_properties(self, txt, pos, deg):
                """Set the position, rotation, and alignment of a text object.

                Args:
                    self (object): Instance of the class.
                    txt (Text): Matplotlib text object to modify.
                    pos (tuple or np.ndarray): (x, y) position for the text.
                    deg (float): Rotation angle in degrees.

                """
                txt.set_position(pos)
                txt.set_rotation(deg)
                txt.set_va('center')
                txt.set_ha('center')

            def _update_rel_pos(self, rel_pos, width, used, extra_space):
                """Update the relative position for the next character along the curve.

                Args:
                    self (object): Instance of the class.
                    rel_pos (float): Current relative position along the curve.
                    width (float): Width of the current character.
                    used (float): Length already used along the curve for this character.
                    extra_space (float): Additional space to add after the character.

                Returns:
                    float: Updated relative position for placing the next character.
                
                """
                return rel_pos + width - used + extra_space

            def _rotation_matrix(self, rad, aspect):
                """Return a 2D rotation matrix with aspect ratio correction.

                Args:
                    self (object): Instance of the class.
                    rad (float): Rotation angle in radians.
                    aspect (float): Aspect ratio (y/x scaling correction).

                Returns:
                    np.ndarray: 2x2 rotation matrix for the given angle and aspect.
                
                """
                return np.array([
                    [np.cos(rad), np.sin(rad) * aspect],
                    [-np.sin(rad) / aspect, np.cos(rad)]
                ])

        CurvedText(
            x=x, y=y, text=text, va=va, axes=axes,
            delimiter=delimiter, fontsize=fontsize, color=color
        )

    def _render_main_well_label(self, color):
        """Render the main well label as curved text along the well circle.

        Args:
            self (object): Instance of the class.
            color (str or tuple): Color for the label text.
        
        """
        # Generate circle coordinates for the well
        x_circle, y_circle = self._generate_circle_coords(self.wellRadius)
        # Adjust coordinates to the well's actual position in the plot
        x = x_circle + self.wellCoords[0]
        y = y_circle + self.extentY - self.wellCoords[1]

        # Format the label for the well (e.g., its discharge)
        label = self._format_label(self.wellQ)

        # Add the curved text label to the plot on ax2
        self._add_curved_text(
            x, y, label,
            va='bottom',
            axes=self.ax2,
            delimiter=';',
            fontsize=8,
            color=color
        )

    def _render_helper_wells_labels(self):
        """Render labels for all helper wells as curved text.

        Only renders labels if the environment type supports helper wells.

        Args:
            self (object): Instance of the FloPyEnv class.
        
        """
        # List of environment types that include helper wells
        helper_envtypes = [
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        ]

        if self.ENVTYPE not in helper_envtypes:
            return

        for i in range(self.nHelperWells):
            # Get helper well coordinates and discharge
            c = self.helperWells[f'wellCoords{i + 1}']
            Q = self.helperWells[f'wellQ{i + 1}']

            # Determine color for the well label
            color = self.renderGetWellColor(Q)

            # Generate circle coordinates for the helper well
            x_circle, y_circle = self._generate_circle_coords(self.helperWellRadius)
            x = x_circle + c[0]
            y = y_circle + self.extentY - c[1]

            # Format label (no units or exponent for helper wells)
            label = self._format_label(Q, unit='', exponent='')

            # Add the curved text label to the plot
            self._add_curved_text(
                x, y, label,
                va='bottom',
                axes=self.ax2,
                delimiter=';',
                fontsize=5,
                color=color
            )

    def render_all_well_labels(self, main_color):
        """Render labels for the main well and all helper wells.

        Args:
            self (object): Instance of the FloPyEnv class.
            main_color (str or tuple): Color for the main well label.
        
        """
        self._render_main_well_label(main_color)
        self._render_helper_wells_labels()

    def render_all_well_labels(self, main_color):
        """Render labels for the main well and all helper wells.

        Args:
            self (object): Instance of the FloPyEnv class.
            main_color (str or tuple): Color for the main well label.
        
        """
        self._render_main_well_label(main_color)
        self._render_helper_wells_labels()

    def renderGetWellColor(self, Q):
        """Return a color representing the well discharge magnitude and sign.

        Args:
            self (object): Instance of the class.
            Q (float): Well discharge (pumping or injection rate).

        Returns:
            str: Color code or name for the well.
        
        """
        # Define color map for positive and negative discharges
        # Positive Q (injection) could be blue shades, negative Q (pumping) red shades
        if Q > 0:
            # Injection well: shades of blue
            return 'blue'
        elif Q < 0:
            # Pumping well: shades of red
            return 'red'
        else:
            # No flow or zero discharge: neutral color
            return 'gray'

    def renderGetWellCircleAlpha(self, Q, maxQ):
        """Return transparency level representing pumping or injection magnitude.

        The alpha value is scaled linearly from 0 to alphaMax (0.9) based on the
        absolute value of Q relative to maxQ. If maxQ is zero, returns 0.

        Args:
            self (object): Instance of the class.
            Q (float): Discharge (pumping or injection rate) of the well.
            maxQ (float): Maximum absolute discharge among all wells.

        Returns:
            float: Alpha value (0 to alphaMax) for the well circle.
        
        """
        alphaMax = 0.9
        if maxQ == 0:
            return 0.0
        return abs(Q) / abs(maxQ) * alphaMax

    def renderTextOnCanvasGameOutcome(self, zorder=10):
        """Plot the final game outcome text on the figure.

        This method checks if the episode is done. If so, it retrieves the game
        result and renders the corresponding text on the canvas.

        Args:
            self (object): Instance of the FloPyEnv class.
            zorder (int, optional): Z-order for the rendered text. Defaults to 10.
        
        """
        if not self.done:
            return
        game_result = self._get_game_result()
        self._render_game_result_text(game_result, zorder)

    def _get_game_result(self) -> str:
        """Return the game outcome string.

        Args:
            self (object): The instance of the class.

        Returns:
            str: 'Success' if the game was successful, otherwise 'Failure'.
        
        """
        return 'Success' if self.success else 'Failure'

    def _render_game_result_text(self, text: str, zorder=10):
        """Render the game result text on the canvas.

        Args:
            self (object): The instance of the class.
            text (str): The game result text to display.
            zorder (int, optional): Drawing order for the text. Defaults to 10.
        
        """
        self.text = self.ax2.text(
            1.25, 65., text,
            fontsize=500,
            color='red',
            zorder=zorder,
            alpha=0.25,
            bbox=dict(facecolor='none', edgecolor='none', pad=0.0)
        )
        self.textSpanAcrossAxis(
            self.text, 100., 80., fig=self.fig, ax=self.ax2
        )

    def textSpanAcrossAxis(self, text, width, height, fig=None, ax=None):
        """
        Auto-decrease and re-expand the fontsize of a text object to match the axis extent.

        Args:
            text (matplotlib.text.Text): Text object to resize.
            width (float): Allowed width in data coordinates.
            height (float): Allowed height in data coordinates.
            fig (matplotlib.figure.Figure): Figure containing the text.
            ax (matplotlib.axes.Axes): Axes containing the text.
        
        """
        size = fig.get_size_inches()
        whratio = size[0] / size[1]

        # 0.98 as a security to avoid buggy overlapses
        self._decrease_font_until_fit(text, 0.98*width, 0.98*height, fig, ax, whratio)
        self._increase_font_until_fit(text, 0.98*width, 0.98*height, fig, ax, whratio)

    def _fits_within_bounds(self, bbox, width, height, whratio):
        """Check if a bounding box fits within given width and height constraints.

        Args:
            self (object): The instance of the class.
            bbox (Bbox): Bounding box to check.
            width (float): Maximum allowed width.
            height (float): Maximum allowed height.
            whratio (float): Width-to-height aspect ratio correction.

        Returns:
            bool: True if bbox fits within the constraints, False otherwise.
        
        """
        fits_w = bbox.width * whratio < width if width else True
        fits_h = bbox.height / whratio < height if height else True
        return fits_w and fits_h

    def _decrease_font_until_fit(self, text, width, height, fig, ax, whratio,
                                dFontDecrease=1.):
        """Decrease font size until text fits within specified bounds.

        Args:
            self (object): The instance of the class.
            text (Text): Matplotlib text object to resize.
            width (float): Maximum allowed width.
            height (float): Maximum allowed height.
            fig (Figure): Matplotlib figure.
            ax (Axes): Matplotlib axes.
            whratio (float): Width-to-height aspect ratio correction.
            dFontDecrease (float, optional): Font size decrement step.
                Defaults to 1.
        
        """
        fontsize = float(text.get_fontsize())
        while fontsize > 1:
            text.draw(fig.canvas.renderer)
            bbox = text.get_window_extent().transformed(ax.transData.inverted())
            if self._fits_within_bounds(bbox, width, height, whratio):
                break
            fontsize = py_max(fontsize - dFontDecrease, 1)
            text.set_fontsize(fontsize)

    def _increase_font_until_fit(self, text, width, height, fig, ax, whratio,
                                dFontIncrease=0.001):
        """Increase font size until text nearly fills the given bounds.

        Args:
            self (object): The instance of the class.
            text (Text): Matplotlib text object to resize.
            width (float): Maximum allowed width.
            height (float): Maximum allowed height.
            fig (Figure): Matplotlib figure.
            ax (Axes): Matplotlib axes.
            whratio (float): Width-to-height aspect ratio correction.
            dFontIncrease (float, optional): Font size increment step.
                Defaults to 0.001.
        
        """
        while True:
            text.draw(fig.canvas.renderer)
            bbox = text.get_window_extent().transformed(ax.transData.inverted())
            if (bbox.width * whratio > width) or (bbox.height / whratio > height):
                break
            new_size = text.get_fontsize() + dFontIncrease
            text.set_fontsize(new_size)

    def renderTextOnCanvasTimeAndScore(self, zorder=10):
        """Plot the current score and time on the figure.

        Args:
            self (object): The instance of the class.
            zorder (int, optional): Drawing order for the text. Defaults to 10.
        
        """
        score_text = self.format_score()
        time_text = self.format_time()
        display_text = f"{score_text}\ntime: {time_text}"
        self.ax2.text(5, 92, display_text, fontsize=12, zorder=zorder)

    def format_time(self) -> str:
        """Return formatted time string in days.

        Args:
            self (object): The instance of the class.

        Returns:
            str: Time in days, formatted as '<days> d'.
        
        """
        time_days = self.timeStep * self.periodLength
        return f"{time_days:.0f} d"

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

    def renderParticle(self, zorder=6):
        """Plot particle at current state.

        Args:
            self (object): The instance of the class.
            zorder (int, optional): Drawing order for the particle. Defaults to 6.
        
        """
        x, y = self._get_particle_coords()
        self._plot_particle(x, y, zorder)

    def _get_particle_coords(self):
        """Return current particle coordinates as (x, y).

        Args:
            self (object): The instance of the class.

        Returns:
            tuple: (x, y) coordinates of the particle.
        
        """
        if self.timeStep == 0:
            return self.minX, self.particleCoords[1]
        return (self.trajectories['x'][-1][-1],
                self.trajectories['y'][-1][-1])

    def _plot_particle(self, x, y, zorder):
        """Plot particle at (x, y) with given zorder.

        Args:
            self (object): The instance of the class.
            x (float): X coordinate of the particle.
            y (float): Y coordinate of the particle.
            zorder (int): Drawing order for the particle.
        
        """
        linewidth = 4 if self.timeStep == 0 else 2
        self.ax2.scatter(x, y, lw=linewidth, c='red', zorder=zorder)

    def renderParticleTrajectory(self, zorder=6):
        """Plot particle trajectory up to the current state.

        Args:
            self (object): The instance of the class.
            zorder (int, optional): Drawing order for the trajectory. Defaults to 6.
        
        """
        if self.timeStep <= 0:
            return
        total, lengths = self._count_trajectory_points()
        colors = self._generate_fade_colors(total)
        self._plot_trajectories(lengths, colors, zorder)

    def _count_trajectory_points(self):
        """Count total points and lengths of each trajectory segment.

        Args:
            self (object): The instance of the class.

        Returns:
            tuple: (total_points, [lengths of each trajectory segment])
        
        """
        lengths = [len(xs) for xs in self.trajectories['x']]
        total = sum(lengths)
        return total, lengths

    def _generate_fade_colors(self, total_points):
        """Generate RGBA colors fading from alpha 0.1 to 1.0 in red.

        Args:
            self (object): The instance of the class.
            total_points (int): Total number of trajectory points.

        Returns:
            np.ndarray: Array of RGBA colors with increasing alpha.
        
        """
        alphas = np.linspace(0.1, 1.0, total_points)
        colors = np.zeros((total_points, 4))
        colors[:, 0] = 1.0  # Red channel
        colors[:, 3] = alphas  # Alpha channel
        return colors

    def _plot_trajectories(self, lengths, colors, zorder):
        """Plot each trajectory segment with corresponding fade color.

        Args:
            self (object): The instance of the class.
            lengths (list): Lengths of each trajectory segment.
            colors (np.ndarray): RGBA colors for each point.
            zorder (int): Drawing order for the trajectories.
        
        """
        idx = 0
        for i, length in enumerate(lengths):
            color = colors[idx + length - 1]
            self.ax2.plot(
                self.trajectories['x'][i],
                self.trajectories['y'][i],
                lw=2, c=color, zorder=zorder
            )
            idx += length

    def renderRemoveAxesTicks(self):
        """Remove axes ticks from all relevant figure axes.

        Args:
            self (object): The instance of the class.
        
        """
        for ax in (self.ax, self.ax2):
            ax.set_xticks([])
            ax.set_yticks([])

        envtypes_with_ax3 = {
            '1s-d', '1s-c', '1r-d', '1r-c',
            '3s-d', '3s-c', '3r-d', '3r-c',
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        }
        if self.ENVTYPE in envtypes_with_ax3:
            self.ax3.set_xticks([])
            self.ax3.set_yticks([])

    def renderSetAxesLimits(self):
        """Set limits of axes from the extents of the environment domain.

        Args:
            self (object): The instance of the class.
        
        """
        for ax in (self.ax, self.ax2, self.ax3):
            ax.set_xlim(self.minX, self.minX + self.extentX)
            ax.set_ylim(self.minY, self.minY + self.extentY)

    def _get_lr_labels(self):
        """Return left and right axis labels.

        Args:
            self (object): The instance of the class.

        Returns:
            tuple: (left_label, right_label)
        
        """
        return (f"Start:   {self.headSpecWest:.2f} m",
                f"Destination:   {self.headSpecEast:.2f} m")

    def _get_tb_labels(self):
        """Return top and bottom axis labels based on ENVTYPE.

        Args:
            self (object): The instance of the class.

        Returns:
            tuple: (top_label, bottom_label)
        
        """
        env = self.ENVTYPE

        group1 = {'1s-d', '1s-c', '1r-d', '1r-c'}
        group2 = {'2s-d', '2s-c', '2r-d', '2r-c'}
        group3 = {
            '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        }

        if env in group1:
            top = f"{getattr(self, 'actionValueNorth', 0):.2f} m"
            bottom = f"{getattr(self, 'actionValueSouth', 0):.2f} m"
        elif env in group2:
            top = ''
            bottom = f"{getattr(self, 'actionValue', 0):.2f} m"
        elif env in group3:
            top = f"{getattr(self, 'headSpecNorth', 0):.2f} m"
            bottom = f"{getattr(self, 'headSpecSouth', 0):.2f} m"
        else:
            top = bottom = ''

        return top, bottom

    def _get_text_labels(self):
        """Return all axis labels: left, right, top, bottom.

        Args:
            self (object): The instance of the class.

        Returns:
            tuple: (left, right, top, bottom) axis labels as strings.
        
        """
        left, right = self._get_lr_labels()
        top, bottom = self._get_tb_labels()
        return left, right, top, bottom

    def renderAddAxesTextLabels(self):
        """Add labeling text to axes.

        Args:
            self (object): The instance of the class.
        
        """
        left, bottom = self.minX, self.minY
        width = abs(self.minX + self.extentX)
        height = abs(self.minY + self.extentY)
        y_center = bottom + height / 2
        x_center = left + width / 2

        text_left, text_right, text_top, text_bottom = self._get_text_labels()

        self._add_vertical_labels(text_left, text_right, y_center)
        self._add_horizontal_labels(text_top, text_bottom, x_center)

    def _add_vertical_labels(self, text_left, text_right, y_center):
        """Add vertical axis labels on left and right sides.

        Args:
            self (object): The instance of the class.
            text_left (str): Label for the left axis.
            text_right (str): Label for the right axis.
            y_center (float): Y-coordinate for vertical label placement.
        
        """
        left_x = self.minX + 2 * self.dCol
        right_x = self.extentX - 2 * self.dCol

        self.ax2.text(left_x, y_center, text_left,
                      ha='left', va='center', rotation='vertical',
                      zorder=10, fontsize=12)
        self.ax2.text(right_x, y_center, text_right,
                      ha='right', va='center', rotation='vertical',
                      zorder=10, fontsize=12)

    def _add_horizontal_labels(self, text_top, text_bottom, x_center):
        """Add horizontal axis labels on top and bottom.

        Args:
            self (object): The instance of the class.
            text_top (str): Label for the top axis.
            text_bottom (str): Label for the bottom axis.
            x_center (float): X-coordinate for horizontal label placement.
        
        """
        top_y = self.extentY - 2 * self.dRow
        bottom_y = self.minY + 2 * self.dRow

        self.ax2.text(x_center, top_y, text_top,
                      ha='center', va='top', rotation='horizontal',
                      zorder=10, fontsize=12)
        self.ax2.text(x_center, bottom_y, text_bottom,
                      ha='center', va='bottom', rotation='horizontal',
                      zorder=10, fontsize=12)

    def renderUserInterAction(self):
        """Enable user control of the environment.

        Connects key press events and waits for user input, adapting behavior
        depending on whether running in an IPython notebook or standard Python.

        Args:
            self (object): The instance of the class.
        
        """
        if self.timeStep == 0:
            self.flagFromIPythonNotebook = 'ipykernel' in modules
        self.fig.canvas.mpl_connect('key_press_event', self.captureKeyPress)
        show(block=False)
        waitforbuttonpress(timeout=self.MANUALCONTROLTIME)

    def renderSavePlot(self, dpi=120):
        """Save plot of the currently rendered timestep.

        Args:
            self (object): The instance of the class.
            dpi (int, optional): Dots per inch for saving. Defaults to 120.
        
        """
        self._ensure_plot_dirs()
        plotfile = self._get_plot_filepath()
        original_size = self.fig.get_size_inches()
        self._prepare_figure_for_saving()

        imarray = self.figToArray(self.fig)
        self.plotArrays.append(imarray)

        self.fig.set_size_inches(original_size)
        if self.SAVEPLOT:
            imsave(plotfile, imarray)

    def _ensure_plot_dirs(self):
        """Ensure the directories for saving plots exist.

        Args:
            self (object): The instance of the class.
        
        """
        if self.timeStep == 0:
            self.plotsfolderpth = join(self.wrkspc, 'runs')
            self.plotspth = join(self.plotsfolderpth, self.ANIMATIONFOLDER)
            makedirs(self.plotsfolderpth, exist_ok=True)
            makedirs(self.plotspth, exist_ok=True)

    def _get_plot_filepath(self):
        """Construct the filepath for the current plot image.

        Args:
            self (object): The instance of the class.

        Returns:
            str: Filepath for the current plot image.
        
        """
        step_str = str(self.timeStep).zfill(
            len(str(abs(self.NAGENTSTEPS))) + 1
        )
        filename = f"{self.MODELNAME}_{step_str}.png"
        return join(self.plotspth, filename)

    def _prepare_figure_for_saving(self):
        """Prepare the figure by hiding axes and margins.

        Args:
            self (object): The instance of the class.
        
        """
        ax = self.fig.gca()
        ax.set_axis_off()
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        self.fig.tight_layout(pad=0)
        self.fig.set_size_inches(7, 7)

    def figToArray(self, fig):
        """Convert Matplotlib figure to a NumPy array.

        Args:
            self (object): The instance of the class.
            fig (Figure): Matplotlib figure to convert.

        Returns:
            np.ndarray: Image array (height, width, 3) in uint8 RGB format.
        
        """
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image_array = frombuffer(
            fig.canvas.tostring_argb(), dtype=uint8
            # fig.canvas.tostring_rgb(), dtype=uint8
        ).reshape(height, width, 4)
        return image_array

    def clear_axis(self, axis):
        """Safely clear a matplotlib axis.

        Args:
            self (object): The instance of the class.
            axis (Axes): Matplotlib axis to clear.
        
        """
        try:
            axis.clear()
        except Exception:
            pass

    def renderClearAxes(self):
        """Clear all axes after a timestep.

        Args:
            self (object): The instance of the class.
        
        """
        self.clear_axis(self.ax)
        self.clear_axis(self.ax2)

        envtypes_with_ax3 = {
            '1s-d', '1s-c', '1r-d', '1r-c',
            '3s-d', '3s-c', '3r-d', '3r-c',
            '4s-d', '4s-c', '4r-d', '4r-c',
            '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c',
            '6s-d', '6s-c', '6r-d', '6r-c'
        }
        if self.ENVTYPE in envtypes_with_ax3:
            self.clear_axis(self.ax3)

    def writeGIFtodisk(self, pathGIF, ims, fps=2, optimizeSize=False):
        """Write a GIF to disk and optionally optimize its size.

        Args:
            self (object): The instance of the class.
            pathGIF (str): Path to save the GIF.
            ims (list): List of image arrays.
            fps (int, optional): Frames per second. Defaults to 2.
            optimizeSize (bool, optional): Whether to optimize GIF size. Defaults to False.
        
        """
        self._write_gif(pathGIF, ims, fps)
        if optimizeSize:
            self._optimize_gif(pathGIF)

    def _write_gif(self, pathGIF, ims, fps):
        """Write images to a GIF file.

        Args:
            self (object): The instance of the class.
            pathGIF (str): Path to save the GIF.
            ims (list): List of image arrays.
            fps (int): Frames per second.
        
        """
        with get_writer(pathGIF, mode='I', fps=fps) as writer:
            for image in ims:
                writer.append_data(image)

    def _optimize_gif(self, pathGIF):
        """Optimize GIF file size using pygifsicle if available.

        Args:
            self (object): The instance of the class.
            pathGIF (str): Path to the GIF file to optimize.
        
        """
        try:
            from pygifsicle import optimize as optimizeGIF
            optimizeGIF(pathGIF)
        except ImportError:
            print('Module pygifsicle not properly installed. '
                  'Returning uncompressed animation.')

    def cellInfoFromCoordinates(self, coords):
        """Determine layer, row, and column corresponding to model location.

        Args:
            self (object): The instance of the class.
            coords (tuple): Coordinates (x, y, z) in model space.

        Returns:
            tuple: (layer, column, row) indices for the model grid.
        
        """
        x, y, z = coords
        layer = self._calculate_index(z + self.zBot, self.dVer, self.nLay)
        column = self._calculate_index(x + self.minX, self.dCol, self.nCol)
        row = self._calculate_index(y + self.minY, self.dRow, self.nRow)
        return (layer, column, row)

    def _calculate_index(self, coord, delta, max_index):
        """Calculate index in a model grid given coordinate, spacing, and max index.

        Args:
            self (object): The instance of the class.
            coord (float): Coordinate value.
            delta (float): Grid spacing.
            max_index (int): Maximum index allowed.

        Returns:
            int: Index value (1-based, clamped to [1, max_index]).
        
        """
        max_index = int(max_index)
        idx = int(ceil(coord / delta))
        idx = py_max(1, py_min(idx, max_index))
        return idx

    def calculatePathLength(self, x, y):
        """Calculate length of advectively traveled path.

        Args:
            self (object): The instance of the class.
            x (list or np.ndarray): X-coordinates of the path.
            y (list or np.ndarray): Y-coordinates of the path.

        Returns:
            float: Total path length.
        
        """
        n = len(x)
        lv = []
        for i in range(1, n):
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            lv.append(sqrt(dx ** 2 + dy ** 2))
        pathLength = sum(lv)
        return pathLength

    def captureKeyPress(self, event):
        """Capture key pressed through manual user interaction.

        Args:
            self (object): The instance of the class.
            event (Event): Matplotlib key press event.

        """
        self.keyPressed = event.key

    def update_action_1d(self, action):
        """Update north and south action values for 1D environments.

        Args:
            self (object): The instance of the class.
            action (str): Action string ('up', 'down', or other).
        
        """
        delta = self._get_delta_1d(action)
        if delta != 0:
            self.actionValueNorth += delta
            self.actionValueSouth += delta

    def _get_delta_1d(self, action):
        """Get action delta for 1D environments.

        Args:
            self (object): The instance of the class.
            action (str): Action string.

        Returns:
            float: Delta value (+/- actionRange or 0).
        
        """
        if action == 'up':
            return self.actionRange
        elif action == 'down':
            return -self.actionRange
        return 0

    def update_action_1c(self, action):
        """Update north and south action values for 1C environments.

        Args:
            self (object): The instance of the class.
            action (list or tuple): [increment, decrement] multipliers.
        
        """
        inc = action[0] * self.actionRange
        dec = action[1] * self.actionRange
        self.actionValueNorth += inc
        self.actionValueSouth += inc
        self.actionValueNorth -= dec
        self.actionValueSouth -= dec

    def update_action_2d(self, action):
        """Update action value for 2D environments.

        Args:
            self (object): The instance of the class.
            action (str): Action string ('up', 'down', or other).
        
        """
        delta = self._get_delta_2d(action)
        if delta != 0:
            self.actionValue += delta

    def _get_delta_2d(self, action):
        """Get action delta for 2D environments.

        Args:
            self (object): The instance of the class.
            action (str): Action string.

        Returns:
            float: Delta value (+/- actionRange or 0).
        
        """
        if action == 'up':
            return self.actionRange
        elif action == 'down':
            return -self.actionRange
        return 0

    def update_action_2c(self, action):
        """Update action value for 2C environments.

        Args:
            self (object): The instance of the class.
            action (list or tuple): [increment, decrement] multipliers.
        
        """
        self.actionValue += action[0] * self.actionRange
        self.actionValue -= action[1] * self.actionRange

    def update_action_3d(self, action):
        """Update well X or Y action value for 3D discrete environments.

        Args:
            self (object): The instance of the class.
            action (str): Action string ('up', 'down', 'left', 'right').
        
        """
        if action == 'up' and self.wellY > self.dRow + self.actionRange:
            self.actionValueY = self.wellY - self.actionRange
        elif (action == 'down' and
              self.wellY < self.extentY - self.dRow - self.actionRange):
            self.actionValueY = self.wellY + self.actionRange
        elif action == 'left' and self.wellX > self.dCol + self.actionRange:
            self.actionValueX = self.wellX - self.actionRange
        elif (action == 'right' and
              self.wellX < self.extentX - self.dCol - self.actionRange):
            self.actionValueX = self.wellX + self.actionRange

    def update_action_3c(self, action):
        """Update well X and Y action values for 3D continuous environments.

        Args:
            self (object): The instance of the class.
            action (list or tuple): [up, down, right, left] increments.
        
        """
        dy = self._calc_delta(action[0], action[1])  # vertical delta
        dx = self._calc_delta(action[2], action[3])  # horizontal delta

        self._update_action_value_y(dy)
        self._update_action_value_x(dx)

    def _calc_delta(self, positive_action, negative_action):
        """Calculate net delta from positive and negative action values.

        Args:
            self (object): The instance of the class.
            positive_action (float): Increment value.
            negative_action (float): Decrement value.

        Returns:
            float: Net delta scaled by actionRange.
        
        """
        return (positive_action - negative_action) * self.actionRange

    def _update_action_value_x(self, dx):
        """Update X action value, ensuring it stays within bounds.

        Args:
            self (object): The instance of the class.
            dx (float): X-direction increment.
        
        """
        new_x = self.wellX + dx
        if self.dCol < new_x < self.extentX - self.dCol:
            self.actionValueX = new_x

    def _update_action_value_y(self, dy):
        """Update Y action value, ensuring it stays within bounds.

        Args:
            self (object): The instance of the class.
            dy (float): Y-direction increment.
        
        """
        new_y = self.wellY + dy
        if self.dRow < new_y < self.extentY - self.dRow:
            self.actionValueY = new_y

    def update_action_multi_wells_d(self, action):
        """Update actions for multiple helper wells in discrete environments.

        Args:
            self (object): The instance of the class.
            action (str): Encoded action string for all wells.
        
        """
        action_list = self._parse_action_list(action)
        for i in range(self.nHelperWells):
            well_id = str(i + 1)
            self._apply_helper_well_action(well_id, action_list[i])

    def _parse_action_list(self, action):
        """Parse an action string into a list of individual well actions.

        Args:
            self (object): The instance of the class.
            action (str): Encoded action string.

        Returns:
            list: List of action strings for each helper well.
        
        """
        action_list = []
        while action:
            for candidate in self.actionSpaceIndividual:
                if action.startswith(candidate):
                    action_list.append(candidate)
                    action = action[len(candidate):]
                    break
            else:
                # No matching action found; optionally raise error or break
                break
        return action_list

    def _apply_helper_well_action(self, well_id, action):
        """Apply an action to a helper well (move or change Q).

        Args:
            self (object): The instance of the class.
            well_id (str): Identifier for the helper well.
            action (str): Action string ('up', 'down', 'left', 'right',
                'moreQ', 'lessQ').
        
        """
        if action == 'up':
            self._move_helper_well_y(well_id, -self.actionRange)
        elif action == 'down':
            self._move_helper_well_y(well_id, self.actionRange)
        elif action == 'left':
            self._move_helper_well_x(well_id, -self.actionRange)
        elif action == 'right':
            self._move_helper_well_x(well_id, self.actionRange)
        elif action == 'moreQ':
            self._change_helper_well_q(well_id, -self.actionRangeQ)
        elif action == 'lessQ':
            self._change_helper_well_q(well_id, self.actionRangeQ)

    def _move_helper_well_x(self, well_id, delta):
        """Move helper well in the X direction if within bounds.

        Args:
            self (object): The instance of the class.
            well_id (str): Identifier for the helper well.
            delta (float): Amount to move in X direction.
        
        """
        key_x = 'wellX' + well_id
        key_action_x = 'actionValueX' + well_id
        new_x = self.helperWells[key_x] + delta
        if self.dCol < new_x < self.extentX - self.dCol:
            self.helperWells[key_action_x] = new_x

    def _move_helper_well_y(self, well_id, delta):
        """Move helper well in the Y direction if within bounds.

        Args:
            self (object): The instance of the class.
            well_id (str): Identifier for the helper well.
            delta (float): Amount to move in Y direction.
        
        """
        key_y = 'wellY' + well_id
        key_action_y = 'actionValueY' + well_id
        new_y = self.helperWells[key_y] + delta
        if self.dRow < new_y < self.extentY - self.dRow:
            self.helperWells[key_action_y] = new_y

    def _change_helper_well_q(self, well_id, delta):
        """Change the discharge (Q) of a helper well within allowed limits.

        Args:
            self (object): The instance of the class.
            well_id (str): Identifier for the helper well.
            delta (float): Amount to change discharge.
        
        """
        key_q = 'wellQ' + well_id
        key_action_q = 'actionValueQ' + well_id
        new_q = self.helperWells[key_q] + delta
        if self.minQhelper < new_q < self.maxQhelper:
            self.helperWells[key_action_q] = new_q

    def update_action_multi_wells_c_position(self, action):
        """Update positions for multiple helper wells in continuous environments.

        Args:
            self (object): The instance of the class.
            action (list): List of action increments for each well.
        
        """
        for i in range(self.nHelperWells):
            w = str(i + 1)
            offset = i * len(self.actionSpaceIndividual)
            if self.ENVTYPE in ['4s-c', '4r-c', '6s-c', '6r-c']:
                dy, dx = self._calculate_displacement(action, offset)
                self._update_well_position(w, dy, dx)

    def _calculate_displacement(self, action, offset):
        """Calculate net displacement in y and x directions based on action values.

        Args:
            self (object): The instance of the class.
            action (list or np.ndarray): Action values for all wells.
            offset (int): Offset index for the current well's action values.

        Returns:
            tuple: (dy, dx) net displacement in y and x directions.
        
        """
        dy_up = action[offset + 0] * self.actionRange
        dy_down = action[offset + 1] * self.actionRange
        dy = dy_up - dy_down

        dx_left = action[offset + 2] * self.actionRange
        dx_right = action[offset + 3] * self.actionRange
        dx = dx_left - dx_right

        return dy, dx

    def _update_well_position(self, well_id, dy, dx):
        """Update the well's position ensuring it stays within bounds.

        Args:
            self (object): The instance of the class.
            well_id (str): Identifier for the helper well.
            dy (float): Displacement in y direction.
            dx (float): Displacement in x direction.
        
        """
        current_y = self.helperWells['wellY' + well_id]
        current_x = self.helperWells['wellX' + well_id]

        # Check and update Y position
        new_y = current_y + dy
        if self.dRow < new_y < self.extentY - self.dRow:
            self.helperWells['actionValueY' + well_id] = new_y

        # Check and update X position
        new_x = current_x + dx
        if self.dCol < new_x < self.extentX - self.dCol:
            self.helperWells['actionValueX' + well_id] = new_x

    def update_action_multi_wells_c_rate(self, action):
        """Update flow rates for multiple helper wells in continuous environments.

        Args:
            self (object): The instance of the class.
            action (list or np.ndarray): Action values for all wells.
        
        """
        for i in range(self.nHelperWells):
            w = str(i + 1)
            offset = i * len(self.actionSpaceIndividual)

            dQ_more, dQ_less = self._extract_dQ_values(action, offset)
            dQ = dQ_less - dQ_more

            self._update_helper_well_Q(w, dQ)

    def _extract_dQ_values(self, action, offset):
        """Extract dQMore and dQLess based on ENVTYPE and action offset.

        Args:
            self (object): The instance of the class.
            action (list or np.ndarray): Action values for all wells.
            offset (int): Offset index for the current well's action values.

        Returns:
            tuple: (dQMore, dQLess) increments for well Q.
        
        """
        if self.ENVTYPE in ['5s-c', '5s-c-cost', '5r-c']:
            dQ_more = action[offset] * self.actionRangeQ
            dQ_less = action[offset + 1] * self.actionRangeQ
        elif self.ENVTYPE in ['6s-c', '6r-c']:
            dQ_more = action[offset + 4] * self.actionRangeQ
            dQ_less = action[offset + 5] * self.actionRangeQ
        else:
            dQ_more = 0
            dQ_less = 0
        return dQ_more, dQ_less

    def _update_helper_well_Q(self, well_id, dQ):
        """Update the helper well flow rate if within allowed bounds.

        Args:
            self (object): The instance of the class.
            well_id (str): Identifier for the helper well.
            dQ (float): Change in discharge (Q).
        
        """
        current_Q = self.helperWells[f'wellQ{well_id}']
        new_Q = current_Q + dQ
        if self.minQhelper < new_Q < self.maxQhelper:
            self.helperWells[f'actionValueQ{well_id}'] = new_Q

    def setActionValue(self, action):
        """Set values for performed actions based on environment type.

        Args:
            self (object): The instance of the class.
            action: Action to perform (type depends on ENVTYPE).
        
        """
        if self.ENVTYPE in ['1s-d', '1r-d']:
            self.update_action_1d(action)
        elif self.ENVTYPE in ['1s-c', '1r-c']:
            self.update_action_1c(action)
        elif self.ENVTYPE in ['2s-d', '2r-d']:
            self.update_action_2d(action)
        elif self.ENVTYPE in ['2s-c', '2r-c']:
            self.update_action_2c(action)
        elif self.ENVTYPE in ['3s-d', '3r-d']:
            self.update_action_3d(action)
        elif self.ENVTYPE in ['3s-c', '3r-c']:
            self.update_action_3c(action)
        elif self.ENVTYPE in [
            '4s-d', '4r-d', '5s-d', '5r-d', '6s-d', '6r-d'
        ]:
            self.update_action_multi_wells_d(action)
        elif self.ENVTYPE in [
            '4s-c', '4r-c', '5s-c', '5s-c-cost', '5r-c', '6s-c', '6r-c'
        ]:
            self.update_action_multi_wells_c_position(action)
            self.update_action_multi_wells_c_rate(action)

    def _append_list(self, base_list, key, observationsDict):
        """Append items from observationsDict[key] to base_list if key exists.

        Args:
            self (object): The instance of the class.
            base_list (list): List to append to.
            key (str): Key to look for in observationsDict.
            observationsDict (dict): Dictionary of observations.
        
        """
        if key in observationsDict:
            base_list.extend(observationsDict[key])

    def _append_helper_wells_perceptron(self, observationsVector, observationsDict):
        """Append helper well data for perceptron OBSPREP.

        Args:
            self (object): The instance of the class.
            observationsVector (list): Vector to append data to.
            observationsDict (dict): Dictionary of observations.
        
        """
        for i in range(self.nHelperWells):
            iStr = str(i)
            if f'wellQ{iStr}' in observationsDict:
                observationsVector.append(observationsDict[f'wellQ{iStr}'])
            if f'wellCoords{iStr}' in observationsDict:
                observationsVector.extend(observationsDict[f'wellCoords{iStr}'])

    def _build_perceptron_vector(self, observationsDict):
        """Build observations vector for perceptron OBSPREP.

        Args:
            self (object): The instance of the class.
            observationsDict (dict): Dictionary of observations.

        Returns:
            list: Observations vector.
        
        """
        observationsVector = []
        self._append_list(observationsVector, 'particleCoords', observationsDict)
        self._append_list(observationsVector, 'heads', observationsDict)
        if 'wellQ' in observationsDict:
            observationsVector.append(observationsDict['wellQ'])
        self._append_list(observationsVector, 'wellCoords', observationsDict)
        self._append_helper_wells_perceptron(observationsVector, observationsDict)
        return observationsVector

    def _append_helper_wells_convolution(self, valsExtra, observationsDict):
        """Append helper well data for convolution OBSPREP.

        Args:
            self (object): The instance of the class.
            valsExtra (list): List to append data to.
            observationsDict (dict): Dictionary of observations.
        
        """
        for i in range(self.nHelperWells):
            iStr = str(i + 1)
            if f'wellQ{iStr}' in observationsDict:
                valsExtra.append(observationsDict[f'wellQ{iStr}'])
            if f'wellCoords{iStr}' in observationsDict:
                valsExtra.extend(observationsDict[f'wellCoords{iStr}'])

    def _build_convolution_vector(self, observationsDict):
        """Build observations vector for convolution OBSPREP.

        Args:
            self (object): The instance of the class.
            observationsDict (dict): Dictionary of observations.

        Returns:
            np.ndarray: Observations array.
        
        """
        valsExtra = []
        if 'particleCoords' in observationsDict:
            valsExtra.extend(observationsDict['particleCoords'])
        observationsVector = np.array(observationsDict.get('heads', []))
        if 'wellQ' in observationsDict:
            valsExtra.append(observationsDict['wellQ'])
        if 'wellCoords' in observationsDict:
            valsExtra.extend(observationsDict['wellCoords'])
        if self.nHelperWells > 0:
            self._append_helper_wells_convolution(valsExtra, observationsDict)

        for val in valsExtra:
            add_ = np.multiply(observationsDict['heads'][0, :, :], 0.)
            add_ = np.add(add_, val)
            add_ = np.expand_dims(add_, axis=0)
            observationsVector = np.concatenate((observationsVector, add_), axis=0)

        return observationsVector

    def observationsDictToVector(self, observationsDict):
        """Convert dictionary of observations to vector.

        Args:
            self (object): The instance of the class.
            observationsDict (dict): Dictionary of observations.

        Returns:
            list or np.ndarray: Vector representation of observations.
        
        """
        if self.OBSPREP == 'perceptron':
            return self._build_perceptron_vector(observationsDict)
        elif self.OBSPREP == 'convolution':
            return self._build_convolution_vector(observationsDict)
        else:
            raise ValueError(f"Unsupported OBSPREP type: {self.OBSPREP}")

    def _parse_helper_wells_perceptron(self, observationsVector, offset):
        """Parse helper wells for 'perceptron' OBSPREP.

        Args:
            self (object): The instance of the class.
            observationsVector (list): Observations vector.
            offset (int): Offset for helper well data.

        Returns:
            dict: Parsed helper well data.
        
        """
        helper_data = {}
        for i in range(self.nHelperWells):
            iStr = str(i)
            start_idx = -(offset + 4 * (i + 1))
            wellQ_idx = start_idx
            wellCoords_idx = slice(start_idx + 1, start_idx + 4)
            helper_data[f'wellQ{iStr}'] = observationsVector[wellQ_idx]
            helper_data[f'wellCoords{iStr}'] = observationsVector[wellCoords_idx]
        return helper_data
    
    def _parse_perceptron(self, observationsVector):
        """Parse observations vector for 'perceptron' OBSPREP.

        Args:
            self (object): The instance of the class.
            observationsVector (list): Observations vector.

        Returns:
            dict: Parsed observations dictionary.
        
        """
        offset = 4 * self.nHelperWells
        obs = {
            'particleCoords': observationsVector[:3],
            'heads': observationsVector[3:-(4 - offset)],
            'wellQ': observationsVector[-(4 - offset)],
            'wellCoords': observationsVector[-(3 - offset):]
        }
        obs.update(self._parse_helper_wells_perceptron(observationsVector, offset))
        return obs

    def _parse_helper_wells_convolution(self, observationsVector):
        """Parse helper wells for 'convolution' OBSPREP.

        Args:
            self (object): The instance of the class.
            observationsVector (np.ndarray): Observations array.

        Returns:
            dict: Parsed helper well data.
        
        """
        helper_data = {}
        for i in range(self.nHelperWells):
            iStr = str(i + 1)
            offset = i * 4
            helper_data[f'wellQ{iStr}'] = observationsVector[8 + offset, 0, 0]
            helper_data[f'wellCoords{iStr}'] = [
                observationsVector[9 + offset, 0, 0],
                observationsVector[10 + offset, 0, 0],
                observationsVector[11 + offset, 0, 0]
            ]
        return helper_data

    def _parse_convolution(self, observationsVector):
        """Parse observations vector for 'convolution' OBSPREP.

        Args:
            self (object): The instance of the class.
            observationsVector (np.ndarray): Observations array.

        Returns:
            dict: Parsed observations dictionary.
        
        """
        obs = {
            'particleCoords': [
                observationsVector[0, 0, 0],
                observationsVector[1, 0, 0],
                observationsVector[2, 0, 0]
            ],
            'heads': observationsVector[3, :, :],
            'wellQ': observationsVector[4, 0, 0],
            'wellCoords': [
                observationsVector[5, 0, 0],
                observationsVector[6, 0, 0],
                observationsVector[7, 0, 0]
            ]
        }
        if self.nHelperWells > 0:
            obs.update(self._parse_helper_wells_convolution(observationsVector))
        return obs

    def observationsVectorToDict(self, observationsVector):
        """Convert list or array of observations to dictionary.

        Args:
            self (object): The instance of the class.
            observationsVector (list or np.ndarray): Observations vector or array.

        Returns:
            dict: Observations dictionary.
        
        """
        if self.OBSPREP == 'perceptron':
            return self._parse_perceptron(observationsVector)
        elif self.OBSPREP == 'convolution':
            return self._parse_convolution(observationsVector)
        else:
            raise ValueError(f"Unsupported OBSPREP type: {self.OBSPREP}")

    def normalize(self, array, zerooffset, diff):
        """Normalize an array by subtracting zerooffset and dividing by diff.

        Args:
            self (object): The instance of the class.
            array (np.ndarray): Array to normalize.
            zerooffset (float or np.ndarray): Offset to subtract.
            diff (float or np.ndarray): Value to divide by.

        Returns:
            np.ndarray: Normalized array.
        
        """
        array = divide(subtract(array, zerooffset), diff)
        return array

    def _scale_data(self, data, key, scale_factor):
        """Scale data[key] by scale_factor if key exists in data.

        Args:
            self (object): The instance of the class.
            data (dict): Data dictionary.
            key (str): Key to scale.
            scale_factor (float): Factor to multiply.
        
        """
        if key in data:
            data[key] = data[key] * scale_factor

    def unnormalize(self, data):
        """Unnormalize various fields in the data dictionary.

        Args:
            self (object): The instance of the class.
            data (dict): Data dictionary to unnormalize.

        Returns:
            dict: Unnormalized data dictionary.
        
        """
        scale_map = {
            'particleCoords': abs(self.minX + self.extentX),
            'heads': self.maxH,
            'wellQ': self.minQ,
            'wellCoords': abs(self.minX + self.extentX),
            'rewards': self.rewardMax,
        }

        for key, factor in scale_map.items():
            self._scale_data(data, key, factor)

        return data


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


class FloPyAgent():
    """Agent to navigate a spawned particle advectively through one of the
    aquifer environments, collecting reward along the way.
    """

    def __init__(self, observationsVector=None, actionSpace=['keep'],
                 hyParams=None, envSettings=None, mode='random',
                 maxTasksPerWorker=1000, maxTasksPerWorkerMutate=1000,
                 maxTasksPerWorkerNoveltySearch=10000, zFill=6):
        """Constructor"""

        self.wrkspc = dirname(abspath(__file__))
        if 'library.zip' in self.wrkspc:
            # changing workspace in case of call from compiled executable
            self.wrkspc = dirname(dirname(self.wrkspc))

        # initializing arguments
        self.observationsVector = observationsVector
        self.hyParams, self.envSettings = hyParams, envSettings
        self.actionSpace = actionSpace
        self.actionSpaceSize = len(self.actionSpace)

        if self.envSettings:
            self.actionType = self.getActionType(self.envSettings['ENVTYPE'])
            # self.actionType = FloPyEnv(initWithSolution=False).getActionType(self.envSettings['ENVTYPE'])
        self.agentMode = mode
        self.maxTasksPerWorker = maxTasksPerWorker
        self.maxTasksPerWorkerMutate = maxTasksPerWorkerMutate
        self.maxTasksPerWorkerNoveltySearch = maxTasksPerWorkerNoveltySearch
        self.zFill = zFill

        # setting seeds
        if self.envSettings is not None:
            self.setSeeds()

        # creating required folders if inexistent
        self.modelpth = join(self.wrkspc, 'models')
        if not exists(self.modelpth):
            makedirs(self.modelpth)

        if self.agentMode == 'DQN':
            # initializing DQN agent
            self.initializeDQNAgent()

        if self.agentMode == 'genetic':
            if self.envSettings['NAGENTSPARALLEL'] == None:
                self.envSettings['NAGENTSPARALLEL'] = cpu_count()

            # creating required folders if inexistent
            self.tempModelpth = join(self.wrkspc, 'temp', 
                self.envSettings['MODELNAME'])

            if not self.envSettings['RESUME']:
                # removing previous models if existing to avoid loading outdated models
                modelFiles = glob(join(self.modelpth, self.envSettings['MODELNAME'] + '_' + '*'))
                for item in modelFiles:
                    try:
                        remove(item)
                    except:
                        files = glob(join(item, '*'))
                        for f in files:
                            remove(f)
                try:
                    rmtree(self.tempModelpth)
                except:
                    pass

            if not exists(self.tempModelpth):
                makedirs(self.tempModelpth)
            if self.envSettings['BESTAGENTANIMATION']:
                runModelpth = join(self.wrkspc, 'runs',
                    self.envSettings['MODELNAME'])
                if not exists(runModelpth):
                    makedirs(runModelpth)
            if self.hyParams is not None:
                if self.hyParams['NOVELTYSEARCH']:
                    self.tempNoveltypth = join(self.tempModelpth, 'novelties')
                    if not exists(self.tempNoveltypth):
                        makedirs(self.tempNoveltypth)
                    else:
                        if self.envSettings is not None:
                            if not self.envSettings['RESUME']:
                                if exists(self.tempNoveltypth):
                                    for f in listdir(self.tempNoveltypth):
                                        remove(join(self.tempNoveltypth, f))

            # initializing seeds for mutation
            self.mutationSeeds = []
            for iGen in range(self.hyParams['NGENERATIONS']):
                seedsInnerGeneration = list(randint(
                    low=1, high=10000000, size=self.hyParams['NAGENTS']))
                self.mutationSeeds.append(seedsInnerGeneration)
                self.setSeeds()

            # initializing seeds for games
            if self.envSettings['SEEDSRANDOM']:
                self.gamesSeeds = []
                self.bestRewards = []
                for iGen in range(self.hyParams['NGENERATIONS']):
                    seedsInnerGeneration = list(randint(
                        low=1, high=10000000, size=self.hyParams['NGAMESAVERAGED']))
                    self.gamesSeeds.append(seedsInnerGeneration)
                self.setSeeds()

            # initializing genetic agents and saving hyperparameters and
            # environment settings or loading them if resuming
            if not self.envSettings['RESUME']:
                self.geneticGeneration = 0
                print('Initializing genetic agents.')
                self.initializeGeneticAgents()
                self.pickleDump(join(self.tempModelpth,
                    self.envSettings['MODELNAME'] + '_hyParams.p'),
                    self.hyParams)
                self.pickleDump(join(self.tempModelpth,
                    self.envSettings['MODELNAME'] + '_envSettings.p'),
                    self.envSettings)
            elif self.envSettings['RESUME']:
                self.hyParams = self.pickleLoad(join(self.tempModelpth,
                    self.envSettings['MODELNAME'] + '_hyParams.p'))

    def initializeDQNAgent(self):
        actionType = FloPyEnv(initWithSolution=False).getActionType(self.envSettings['ENVTYPE'])
        
        # initializing main predictive and target model
        self.mainModel = self.createNNModel(actionType)
        self.targetModel = self.createNNModel(actionType)
        self.targetModel.set_weights(self.mainModel.get_weights())

        # initializing array with last training data of specified length
        self.replayMemory = deque(maxlen=self.hyParams['REPLAYMEMORYSIZE'])
        self.epsilon = self.hyParams['EPSILONINITIAL']

        # initializing counter for updates on target network
        self.targetUpdateCount = 0

    # def initializeGeneticAgents(self):
    #     if self.envSettings['KEEPMODELHISTORY']:
    #         chunksTotal = self.yieldChunks(
    #             arange(self.hyParams['NAGENTS']),
    #             self.envSettings['NAGENTSPARALLEL'] * self.maxTasksPerWorkerMutate
    #         )
    #         for chunk in chunksTotal:
    #             # Sequentially process each index in the chunk
    #             for agent_idx in chunk:
    #                 self.randomAgentGenetic(agent_idx)
    #     else:
    #         self.mutationHistory = {}
    #         for iAgent in range(self.hyParams['NAGENTS']):
    #             creationSeed = iAgent + 1
    #             agentNumber = iAgent + 1
    #             self.mutationHistory = self.updateMutationHistory(
    #                 self.mutationHistory,
    #                 agentNumber,
    #                 creationSeed,
    #                 mutationSeeds=[],
    #                 mutationSeed=None
    #             )

    def initializeGeneticAgents(self):
        if self.envSettings['KEEPMODELHISTORY']:
            chunksTotal = self.yieldChunks(arange(self.hyParams['NAGENTS']),
                self.envSettings['NAGENTSPARALLEL']*self.maxTasksPerWorkerMutate)
            for chunk in chunksTotal:
                _ = self.multiprocessChunks(self.randomAgentGenetic, chunk)
        else:
            self.mutationHistory = {}
            for iAgent in range(self.hyParams['NAGENTS']):
                creationSeed = iAgent+1
                agentNumber = iAgent+1
                self.mutationHistory = self.updateMutationHistory(self.mutationHistory, agentNumber, creationSeed, mutationSeeds=[], mutationSeed=None)

    def setSeeds(self):
        self.SEED = self.envSettings['SEEDAGENT']
        numpySeed(self.SEED)
        randomSeed(self.SEED)
        set_random_seed(self.SEED)

    def getActionType(self, ENVTYPE):
        if '-d' in ENVTYPE:
            actionType = 'discrete'
        elif '-c' in ENVTYPE:
            actionType = 'continuous'
        else:
            print('Environment name is unknown.')
            quit()

        return actionType

    def runDQN(self, env):
        """
        Run main pipeline for Deep Q-Learning optimisation.
        # Inspiration and larger parts of code modified after sentdex
        # https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
        """

        self.actionType = env.actionType

        # generating seeds to generate reproducible cross-validation data
        # note: avoids variability from averaged new games
        numpySeed(self.envSettings['SEEDAGENT'])
        self.seedsCV = randint(self.envSettings['SEEDAGENT'],
            size=self.hyParams['NGAMESCROSSVALIDATED']
            )

        gameRewards = []
        # iterating over games being played
        for iGame in tqdm(range(1, self.hyParams['NGAMES']+1), ascii=True,
            unit='games'):
            env.reset(MODELNAME=env.MODELNAME) # no need for seed?
            # MODELNAME=MODELNAMETEMP, _seed=SEEDTEMP


            # simulating, updating replay memory and training main network
            self.takeActionsUpdateAndTrainDQN(env)
            if env.success:
                self.gameReward = self.gameReward
            elif env.success == False:
                # overwriting simulation memory to zero if no success
                # to test: is it better to give reward an not reset to 0?
                # self.gameReward = 0.0
                self.updateReplayMemoryZeroReward(self.gameStep) # is this better on or off?
            gameRewards.append(self.gameReward)

            # cross validation, after every given number of games
            if not iGame % self.hyParams['CROSSVALIDATEEVERY'] or iGame == 1:
                self.crossvalidateDQN(env)
    
                MODELNAME = self.envSettings['MODELNAME']
                DQNfstring = f'{MODELNAME}{iGame:_>7.0f}ep'\
                    f'{self.max_rewardCV:_>7.1f}max'\
                    f'{self.average_rewardCV:_>7.1f}avg'\
                    f'{self.min_rewardCV:_>7.1f}min'\
                    f'{datetime.now().strftime("%Y%m%d%H%M%S")}datetime.keras'
                if self.average_rewardCV >= self.envSettings['REWARDMINTOSAVE']:
                    # saving model if larger than a specified reward threshold
                    self.mainModel.save(join(self.wrkspc, 'models', DQNfstring))

            # decaying epsilon
            if self.epsilon > self.hyParams['EPSILONMIN']:
                self.epsilon *= self.hyParams['EPSILONDECAY']
                self.epsilon = max([self.hyParams['EPSILONMIN'], self.epsilon])

    def runGenetic(self, env, searchNovelty=False):
        """Run main pipeline for genetic agent optimisation.
        # Inspiration and larger parts of code modified after and inspired by:
        # https://github.com/paraschopra/deepneuroevolution
        # https://arxiv.org/abs/1712.06567
        """

        self.actionType = env.actionType
        if self.hyParams['NAGENTS'] <= self.hyParams['NNOVELTYELITES']:
            raise ValueError('Settings and hyperparameters require changes: ' + \
                             'The number of novelty elites considered during novelty search ' + \
                             'should be lower than the number of agents considered ' + \
                             'to evolve.')

        # setting environment and number of games
        self.env, n = env, self.hyParams['NGAMESAVERAGED']
        if searchNovelty:
            self.searchNovelty, self.noveltyArchive = searchNovelty, {}
            self.noveltyItemCount = 0
            self.agentsUnique, self.agentsUniqueIDs = [], []
            self.agentsDuplicate, self.agentsDuplicateID = [], []
            if env.actionType == 'discrete':
                self.actionsUniqueIDMapping = defaultdict(count().__next__)
        cores = self.envSettings['NAGENTSPARALLEL']
        # generating unique process ID from system time
        self.pid = str(uuid4())
        self.pidList = list(self.pid)
        shuffle(self.pidList)
        self.pid = ''.join(self.pidList)

        agentCounts = [iAgent for iAgent in range(self.hyParams['NAGENTS'])]
        self.rereturnChildrenGenetic = False
        for self.geneticGeneration in range(self.hyParams['NGENERATIONS']):
            self.flagSkipGeneration = False
            self.generatePathPrefixes()

            if self.envSettings['RESUME']:
                if self.searchNovelty:
                    if self.geneticGeneration > 0:
                        self.noveltyArchive = self.pickleLoad(join(
                            self.tempPrevModelPrefix + '_noveltyArchive.p'))
                        if not self.envSettings['KEEPMODELHISTORY']:
                            self.mutationHistory = self.pickleLoad(join(
                                self.tempPrevModelPrefix + '_mutationHistory.p'))
                        self.noveltyItemCount = len(self.noveltyArchive.keys())
                sortedParentIdxs, continueFlag, breakFlag = self.resumeGenetic()
                if continueFlag: continue
                if breakFlag: break

                if self.searchNovelty:
                    # regenerating list of unique and duplicate agents
                    # in case of resume
                    for iAgent in range(self.noveltyItemCount):
                        agentStr = 'agent' + str(iAgent+1)
                        # self.noveltyArchive[agentStr] = {}
                        tempAgentPrefix = self.noveltyArchive[agentStr]['modelFile'].replace('.keras', '')
                        pth = join(tempAgentPrefix + '_results.p')
                        actions = self.pickleLoad(pth)['actions']
                        if env.actionType == 'discrete':
                            actionsAll = [action for actions_ in actions for action in actions_]
                            actionsUniqueID = self.actionsUniqueIDMapping[tuple(actionsAll)]
                            self.noveltyArchive[agentStr]['actionsUniqueID'] = actionsUniqueID
                            if actionsUniqueID not in self.agentsUniqueIDs:
                                # checking if unique ID from actions already exists
                                self.agentsUnique.append(iAgent)
                                self.agentsUniqueIDs.append(actionsUniqueID)
                            else:
                                self.agentsDuplicate.append(iAgent)
                                # self.agentsDuplicateID.apppend()

            print('########## started generation ' + str(self.geneticGeneration+1).zfill(self.zFill) + ' ##########')

            # simulating agents in environment, returning average of n runs
            self.rewards = self.runAgentsRepeatedlyGenetic(agentCounts, n, env)
            print('lowest reward', min(self.rewards))
            print('average reward', mean(self.rewards))
            print('highest reward', max(self.rewards))
            if self.envSettings['SEEDSRANDOM']:
                self.bestRewards.append(max(self.rewards))
                print('highest sliding reward (n=50)', mean(self.bestRewards[-50:]))

            # sorting by rewards in reverse, starting with indices of top reward
            # https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
            sortedParentIdxs = argsort(
                self.rewards)[::-1][:self.hyParams['NAGENTELITES']]
            self.bestAgentReward = self.rewards[sortedParentIdxs[0]]
            self.pickleDump(join(self.tempModelPrefix +
                '_agentsSortedParentIndexes.p'), sortedParentIdxs)

            if self.searchNovelty:
                print('Performing novelty search')
                t0PreparingNoveltySearch = time()
                # iterating through agents and storing with novelty in archive
                # calculating average nearest-neighbor novelty score
                for iAgent in range(self.hyParams['NAGENTS']):
                    noveltiesAgent, actionsAll = [], []
                    itemID = self.noveltyItemCount
                    k = self.noveltyItemCount
                    agentStr = 'agent' + str(k+1)
                    self.noveltyArchive[agentStr] = {}
                    # self.noveltyArchive[agentStr]['novelties'] = {}
                    tempAgentPrefix = join(self.tempModelPrefix + '_agent'
                        + str(iAgent + 1).zfill(self.zFill))
                    modelFile = tempAgentPrefix + '.keras'
                    resultsFile = tempAgentPrefix + '_results.p'

                    if env.actionType == 'discrete':
                        pth = join(tempAgentPrefix + '_results.p')
                        actions = self.pickleLoad(pth)['actions']
                        actionsAll = [action for actions_ in actions for action in actions_]
                        actionsUniqueID = self.actionsUniqueIDMapping[tuple(actionsAll)]
                        # https://stackoverflow.com/questions/38291372/assign-unique-id-to-list-of-lists-in-python-where-duplicates-get-the-same-id
                        self.noveltyArchive[agentStr]['actionsUniqueID'] = actionsUniqueID
                    self.noveltyArchive[agentStr]['itemID'] = itemID
                    self.noveltyArchive[agentStr]['modelFile'] = modelFile
                    self.noveltyArchive[agentStr]['resultsFile'] = resultsFile
                    # self.noveltyArchive[agentStr]['actions'] = actions

                    if env.actionType == 'discrete':
                        # if not self.noveltyItemCount > self.hyParams['NNOVELTYNEIGHBORS']:
                        # removing duplicate novelty calculation only in case neighbour limit is not reached
                        # as otherwise the same novelty might not apply
                        if actionsUniqueID not in self.agentsUniqueIDs:
                            # checking if unique ID from actions already exists
                            self.agentsUnique.append(k)
                            self.agentsUniqueIDs.append(actionsUniqueID)
                        else:
                            self.agentsDuplicate.append(k)

                        # is this necessary?

                        # else:
                        #     self.agentsUnique, self.agentsUniqueIDs, self.agentsDuplicate = [], [], []
                        #     # otherwise computed as if unique to avoid assigning novelty
                        #     # from duplicates with different nearest neighbours
                        #     for iNov in range(self.noveltyItemCount+1):
                        #         self.agentsUnique.append(iNov)
                        #         self.noveltyArchive['agent' + str(iNov+1)]['actionsUniqueID'] = iNov
                        #         # self.agentsUniqueIDs.append(iNov)
                    elif env.actionType == 'continuous':
                            self.agentsUnique, self.agentsUniqueIDs, self.agentsDuplicate = [], [], []
                            for iNov in range(self.noveltyItemCount+1):
                                self.agentsUnique.append(iNov)
                                self.noveltyArchive['agent' + str(iNov+1)]['actionsUniqueID'] = iNov
                    self.noveltyItemCount += 1
                tPreparingNoveltySearch = time() - t0PreparingNoveltySearch
                print('Preparing novelty search took', tPreparingNoveltySearch, 's')

                if not self.noveltyItemCount > self.hyParams['NNOVELTYNEIGHBORS']:
                    print('Novelty search:', len(self.agentsUnique), 'unique agents', len(self.agentsDuplicate), 'duplicate agents')
                else:
                    print('Novelty search (neighbor level reached):', len(self.agentsUnique), 'unique agents', len(self.agentsDuplicate), 'duplicate agents')
                # updating novelty of unique agents
                # Note: This can become a massive bottleneck with increasing
                # number of stored agent information and generations
                # despite parallelization
                noveltiesUniqueAgents, updateFlagsUniqueAgents = [], []

                # # loading all actions
                # # this can be improved by loading only once and then saving to disk between generations
                # t0LoadActionsOnce = time()
                # sharedArrayActions = self.loadActions()
                # tLoadActionsOnce = time() - t0LoadActionsOnce

                t0NoveltySearch = time()
                self.neighborLimitReached = (len(self.agentsUnique) > self.hyParams['NNOVELTYNEIGHBORS'])

                t0CalcNoveltyArgs = time()
                args = []

                for iAgent in range(len(self.agentsUnique)):
                    if self.neighborLimitReached:
                        # includes identification of iAgents needing novelty update
                        rangeLower, rangeHigher, iAgentInCroppedArray, needsUpdate = self.calculateNoveltyNeighborBounds(iAgent)

                        # load only actions needing update here?
                        # so replace sharedArrayActions with a load function
                        # other approaches will still blow up memory

                        # ranges were mapped these to indices considering uniques
                        # arr = sharedArrayActions[rangeLower:rangeHigher]
                    else:
                        needsUpdate = True
                        rangeLower, rangeHigher, iAgentInCroppedArray = 0, len(self.agentsUnique), iAgent

                    # replace arr with generator reference?
                    if needsUpdate:
                        args.append([iAgent, None, rangeLower, rangeHigher, iAgentInCroppedArray])
                    updateFlagsUniqueAgents.append(needsUpdate)
                tCalcNoveltyArgs = time()-t0CalcNoveltyArgs
                print('Calculated novelty search arguments, took', int(tCalcNoveltyArgs), 's')

                chunksTotal = self.yieldChunks(args,
                    cores*self.maxTasksPerWorkerNoveltySearch)

                print('Started novelty search ...')
                for chunk in chunksTotal:
                    # sharing dictionary containing actions to avoid loading
                    # those are only the updated novelties
                    noveltiesPerAgent = self.multiprocessChunks(
                        self.calculateNoveltyPerAgent, chunk)
                    noveltiesUniqueAgents += noveltiesPerAgent

                print('Finished novelty search, took', int(time()-t0NoveltySearch), 's')

                # calculating novelty of unique agents
                count_ = 0
                for i, iUniqueAgent in enumerate(self.agentsUnique):
                    agentStr = 'agent' + str(iUniqueAgent+1)
                    if updateFlagsUniqueAgents[i]:
                        # novelty = noveltiesUniqueAgents[actionsUniqueID]
                        novelty = noveltiesUniqueAgents[count_]
                        self.noveltyArchive[agentStr]['novelty'] = novelty
                        count_ += 1

                # updating novelty of duplicate agents from existing value
                count_ = 0
                for iDuplicateAgent in self.agentsDuplicate:
                    # finding ID of agent representing duplicate agent's actions
                    agentStr = 'agent' + str(iDuplicateAgent+1)
                    actionsUniqueID = self.noveltyArchive[agentStr]['actionsUniqueID']
                    itemID = self.noveltyArchive[agentStr]['itemID']
                    # print(self.agentsUnique, actionsUniqueID)
                    novelty = noveltiesUniqueAgents[actionsUniqueID]
                    self.noveltyArchive[agentStr]['novelty'] = novelty

                self.pickleDump(join(self.tempModelPrefix +
                    '_noveltyArchive.p'), self.noveltyArchive)

                self.novelties, self.noveltyFilenames = [], []
                for k in range(self.noveltyItemCount):
                    agentStr = 'agent' + str(k+1)
                    self.novelties.append(
                        self.noveltyArchive[agentStr]['novelty'])
                    self.noveltyFilenames.append(
                        self.noveltyArchive[agentStr]['modelFile'])

            if self.geneticGeneration+1 >= self.hyParams['ADDNOVELTYEVERY']:
                print('lowest novelty', min(self.novelties))
                print('average novelty', mean(self.novelties))
                print('highest novelty', max(self.novelties))

            # returning best-performing agents
            self.returnChildrenGenetic(sortedParentIdxs)

            if not self.envSettings['KEEPMODELHISTORY']:
                self.pickleDump(join(self.tempModelPrefix +
                    '_mutationHistory.p'), self.mutationHistory)

            MODELNAME = self.envSettings['MODELNAME']
            MODELNAMEGENCOUNT = (MODELNAME + '_gen' +
                str(self.geneticGeneration + 1).zfill(self.zFill) + '_avg' +
                str('%.1f' % (max(self.rewards))))
            # saving best agent of the current generation
            self.saveBestAgent(MODELNAME)

            if self.envSettings['BESTAGENTANIMATION']:
                if not self.flagSkipGeneration:
                    self.saveBestAgentAnimation(env, self.bestAgentFileName,
                        MODELNAMEGENCOUNT, MODELNAME)
            # if not self.envSettings['KEEPMODELHISTORY']:
            #     # removing stored agent models of finished generation
            #     # as the storage requirements can be substantial
            #     for agentIdx in range(self.hyParams['NAGENTS']):
            #         remove(join(self.tempModelPrefix + '_agent' +
            #             str(agentIdx + 1).zfill(self.zFill) + '.keras'))

            if self.envSettings['ENVTYPE'] in ['0s-c']:
                self.env = FloPyEnv(
                    self.envSettings['ENVTYPE'],
                    self.envSettings['PATHMF2005'],
                    self.envSettings['PATHMP6'],
                    MODELNAME=self.envSettings['MODELNAME'],
                    flagRender=self.envSettings['RENDER'],
                    NAGENTSTEPS=self.hyParams['NAGENTSTEPS'],
                    nLay=self.envSettings['NLAY'],
                    nRow=self.envSettings['NROW'],
                    nCol=self.envSettings['NCOL'],
                    OBSPREP=self.hyParams['NNTYPE'],
                    initWithSolution=True,
                    PATHMF6DLL=None)

            print('########## finished generation ' + str(self.geneticGeneration+1).zfill(self.zFill) + ' ##########')

    def loadActions(self, agents=None):

        if agents == None:
            agents = self.agentsUnique

        sharedListActions = []

        # retrieving shape to allow padding with zeros
        # else conversion to array fails with different length of action collections
        # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros/35751834

        # pre-loop to determine the largest shape of actions
        pth = self.noveltyArchive['agent' + str(0+1)]['resultsFile']
        if self.hyParams['NGAMESAVERAGED'] == 1:
            actions = array(self.pickleLoad(pth)['actions'], dtype=object)
            dimShape = len(shape(actions))
        else:
            dimShape = 0
            actions = array(self.pickleLoad(pth)['actions'], dtype=object)
            for iActions in range(len(actions)):
                actions_ = actions[iActions]
                dimShape_ = len(shape(actions_))
                if dimShape_ >= dimShape:
                    dimShape = dimShape_

        maxShape = [0 for _ in range(dimShape)]
        for iAgent in agents:
            agentStr = 'agent' + str(iAgent+1)
            pth = self.noveltyArchive[agentStr]['resultsFile']
            if self.hyParams['NGAMESAVERAGED'] == 1:
                actions = self.pickleLoad(pth)['actions']
            # else:
            #     actions = self.pickleLoad(pth)['actions'][0]
            if self.hyParams['NGAMESAVERAGED'] == 1:
                actions = array(actions, dtype=object)
                for iDim in range(dimShape):
                    if shape(actions)[iDim] > maxShape[iDim]:
                        maxShape[iDim] = shape(actions)[iDim]
            elif self.hyParams['NGAMESAVERAGED'] != 1:
                actions = self.pickleLoad(pth)['actions']
                for iActions in range(shape(actions)[0]):
                    actions_ = array(actions[iActions], dtype=object)
                    dimShape = len(shape(actions_))
                    for iDim in range(dimShape):
                        if shape(actions_)[iDim] > maxShape[iDim]:
                            maxShape[iDim] = shape(actions_)[iDim]

        if self.hyParams['NGAMESAVERAGED'] != 1:
            maxShape = [self.hyParams['NGAMESAVERAGED']] + maxShape
        maxShape = tuple(maxShape)

        # replace this with specific agents?
        for iAgent in agents:
        # for iAgent in range(len(agents)):
            agentStr = 'agent' + str(iAgent+1)
            pth = self.noveltyArchive[agentStr]['resultsFile']
            actions = self.pickleLoad(pth)['actions']
            if self.actionType == 'discrete':
                # sharedListActions_ = chararray(shape=maxShape, itemsize=10)
                sharedListActions_ = empty(maxShape, dtype='S10')  # Byte strings, max length 10
                sharedListActions_[:] = 'keep'
                padValue ='keep'
                # padding action lists to same length
                maxLen = maxShape[1]
                for iActions in range(len(actions)):
                    actions[iActions] = actions[iActions] + [padValue] * (maxLen - len(actions[iActions]))
                sharedListActions_[:len(actions)] = actions
            elif self.actionType == 'continuous':
                sharedListActions_ = zeros(maxShape)
                if self.hyParams['NGAMESAVERAGED'] != 1:
                    for iActions in range(len(actions)):
                        a = array(actions[iActions], dtype=object)
                        sharedListActions_[iActions, :a.shape[0], :a.shape[1]] = array(a, dtype=object)
                        # sharedListActions_[iActions, :a.shape[0], :a.shape[1]] = array([a])
                else:
                    a = array(actions, dtype=object)
                    sharedListActions_[:a.shape[0], :a.shape[1]] = a
            sharedListActions.append(sharedListActions_)

        if self.actionType == 'discrete':
            self.actionsUniqueScheme, sharedListActions = unique(sharedListActions, return_inverse=True)
            sharedListActions = reshape(sharedListActions, tuple([len(agents)] + list(maxShape)))
            sharedArrayActions = array(sharedListActions, dtype=object)
        if self.actionType == 'continuous':
            sharedArrayActions = array(sharedListActions, dtype=object)

        return sharedArrayActions

    def calculateNoveltyNeighborBounds(self, iAgent):
        """Check if half of NNOVELTYNEIGHBORS are available surrounding the given index,
        if not agents are selected until the index boundary and more from the other end."""

        nAgentsUnique = len(self.agentsUnique)

        # defining how many neighbors to consider below and above
        # will be asymmetric in case of odd NNOVELTYNEIGHBORS
        nLower = int(floor(self.hyParams['NNOVELTYNEIGHBORS']/2))
        nHigher = int(ceil(self.hyParams['NNOVELTYNEIGHBORS']/2))
        bottomReached, topReached = False, False
        if iAgent - nLower >= 0:
            rangeLower = iAgent - nLower
            rangeHigher = iAgent + nHigher
        else:
            bottomReached = True
            rangeLower, rangeHigher = 0, self.hyParams['NNOVELTYNEIGHBORS']

        if not bottomReached:
            # if iAgent + nHigher <= self.noveltyItemCount:
            if iAgent + nHigher <= nAgentsUnique:
                rangeLower = iAgent - nLower
                rangeHigher = iAgent + nHigher
            else:
                topReached = True
                rangeLower = nAgentsUnique - self.hyParams['NNOVELTYNEIGHBORS']
                rangeHigher = nAgentsUnique
        if not bottomReached and not topReached:
            iAgentInCroppedArray = iAgent - rangeLower
        elif bottomReached:
            iAgentInCroppedArray = iAgent
        elif topReached:


            # DO UNIT TESTING IF THIS YIELDS CORRECT NOVELTIES

            iAgentInCroppedArray = iAgent+1-nAgentsUnique + nHigher-1
        else:
            print('Error iAgentInCroppedArray could not be defined. This is a bug, please report.')

        # checking if agent needs updating given the setting of NNOVELTYNEIGHBORS to consider
        # considering NAGENTS as an edge case avoids to look up existing novelty when not yet existing
        nMaxUpdates = self.hyParams['NAGENTS'] + self.hyParams['NNOVELTYNEIGHBORS']
        if (nAgentsUnique - (iAgent+1)) >= nMaxUpdates:
            needsUpdate = False
        else:
            needsUpdate = True

        return rangeLower, rangeHigher, iAgentInCroppedArray, needsUpdate

    def createNNModel(self, actionType, seed=None):
        """Create neural network."""
        seed_orig = deepcopy(seed)
        if seed is None:
            seed = self.SEED
        model = Sequential()
        seed = int(seed)  # force Python int
        seed = SeedGenerator(seed=seed)
        if actionType == 'discrete':
            initializer = random_uniform(minval=-2.2, maxval=2.2, seed=seed)
        elif actionType == 'continuous':
            initializer = random_uniform(minval=-2.5, maxval=2.5, seed=seed)
        else:
            print('Chosen normal glorot initializer, as not specified.')
            initializer = glorot_normal(seed=seed)

        nHiddenNodes = copy(self.hyParams['NHIDDENNODES'])
        # resetting numpy seeds to generate reproducible architecture
        numpySeed(seed_orig)

        if self.hyParams['NNTYPE'] == 'perceptron':
            # fully-connected feed-forward multi-layer neural network
            # applying architecture (variable number of nodes per hidden layer)
            if self.agentMode == 'genetic' and self.hyParams['ARCHITECTUREVARY']:
                for layerIdx in range(len(nHiddenNodes)):
                    nHiddenNodes[layerIdx] = randint(2, self.hyParams['NHIDDENNODES'][layerIdx]+1)
            for layerIdx in range(len(nHiddenNodes)):
                inputShape = shape(self.observationsVector) if layerIdx == 0 else []
                model.add(Dense(units=nHiddenNodes[layerIdx],
                    input_shape=inputShape,
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    use_bias=True))
                if self.hyParams['BATCHNORMALIZATION']:
                    model.add(BatchNormalization())
                model.add(Activation(self.hyParams['HIDDENACTIVATIONS'][layerIdx]))
                if 'DROPOUTS' in self.hyParams:
                    if self.hyParams['DROPOUTS'][layerIdx] != 0.0:
                        model.add(Dropout(self.hyParams['DROPOUTS'][layerIdx]))

        elif self.hyParams['NNTYPE'] == 'convolution':
            # convolutional feed-forward neural network
            
            inputShape = shape(self.observationsVector)
            model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                             activation='relu',
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             input_shape=inputShape, padding='same'))
            # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2),
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             activation='relu', padding='same'))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             activation='relu', padding='same'))
            model.add(Flatten())
            model.add(Dense(512,
                      kernel_initializer=initializer,
                      bias_initializer=initializer,
                      activation='relu'))
            
        # adding output layer
        if actionType == 'discrete':
            model.add(Dense(self.actionSpaceSize, activation='linear', # 'softmax'
                kernel_initializer=initializer))
        elif actionType == 'continuous':
            # sigmoid used here as actions are predicted as fraction of actionRange
            model.add(Dense(self.actionSpaceSize, activation='sigmoid',
                kernel_initializer=initializer))

        # compiling to avoid warning while saving agents in genetic search
        # specifics are irrelevant, as genetic models are not optimized
        # along gradients
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001),
                      metrics=['mean_squared_error'])

        return model

    def checkParameterMean(self, agent):
        
        weights, means = agent.get_weights(), []
        for iParam, parameters in enumerate(weights):
            means.append(mean(weights[iParam]))
        
        return mean(means)

    def loadModelFromMutationHistory(self, creationSeed, mutationSeeds):

        agent = self.createNNModel(self.actionType, seed=creationSeed)

        for mutationSeed in mutationSeeds:
            agent = self.mutateGenetic(agent, mutationSeed)
        
        return agent

    def updateMutationHistory(self, mutationHistory, agentNumber, creationSeed, mutationSeeds=[], mutationSeed=None):

        mutationHistory['agent' + str(agentNumber)] = {}
        mutationHistory['agent' + str(agentNumber)]['creationSeed'] = creationSeed
        mutationHistory['agent' + str(agentNumber)]['mutationSeeds'] = mutationSeeds
        if mutationSeed != None:
            mutationHistory['agent' + str(agentNumber)]['mutationSeeds'].append(mutationSeed)
        
        return mutationHistory

    def createMutationRecord(self, creationSeed, mutationSeeds=[], mutationSeed=None):

        record = {}
        record['creationSeed'] = creationSeed
        record['mutationSeeds'] = mutationSeeds
        if mutationSeed != None:
            record['mutationSeeds'].append(mutationSeed)
        
        return record

    def updateReplayMemory(self, transition):
        """Update replay memory by adding a given step's data to a memory
        replay array.
        """
        self.replayMemory.append(transition)

    def updateReplayMemoryZeroReward(self, steps):
        """Update replay memory rewards to zero in case game ended up with zero
        reward.
        """
        for i in range(steps):
            self.replayMemory[-i][2] = 0.0

    def train(self, terminal_state, step):
        """Trains main network every step during a game."""

        # training only if certain number of samples is already saved
        if len(self.replayMemory) < self.hyParams['REPLAYMEMORYSIZEMIN']:
            return

        # retrieving a subset of random samples from memory replay table
        minibatch = randomSample(self.replayMemory,
                                 self.hyParams['MINIBATCHSIZE']
                                 )

        # retrieving current states from minibatch
        # then querying NN model for Q values
        current_states = array([transition[0] for transition in minibatch])
        current_qs_list = self.mainModel.predict(current_states)

        # retrieving future states from minibatch
        # then querying NN model for Q values
        # when using target network, query it, otherwise main network should be
        # queried
        new_current_states = array([transition[3] for transition in minibatch])
        future_qs_list = self.targetModel.predict(new_current_states)

        X, y = [], []
        # enumerating batches
        for index, (current_state, action, reward, new_current_state,
                    done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states,
            # otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation
            if not done:
                max_future_q = max(future_qs_list[index])
                new_q = reward + self.hyParams['DISCOUNT'] * max_future_q
            else:
                new_q = reward

            # updating Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            # appending states to training data
            X.append(current_state)
            y.append(current_qs)

        # fitting on all samples as one batch, logging only on terminal state
        self.mainModel.fit(array(X), array(y),
                           batch_size=self.hyParams['MINIBATCHSIZE'],
                           verbose=0, shuffle=False
                           )

        # updating target network counter every game
        if terminal_state:
            self.targetUpdateCount += 1

        # updating target network with weights of main network,
        # if counter reaches set value
        if self.targetUpdateCount > self.hyParams['UPDATEPREDICTIVEMODELEVERY']:
            self.targetModel.set_weights(self.mainModel.get_weights())
            self.targetUpdateCount = 0

    def takeActionsUpdateAndTrainDQN(self, env):
        """
        Take an action and update as well as train the main network every
        step during a game. Acts as the main operating body of the runDQN
        function.
        """

        # retrieving initial state
        current_state = env.observationsVectorNormalized

        # resetting counters prior to restarting game
        self.gameReward, self.gameStep, done = 0, 1, False
        for game in range(self.hyParams['NAGENTSTEPS']):
            # epsilon defines the fraction of random to queried actions
            if random() > self.epsilon:
                # retrieving action from Q table
                if env.actionType == 'discrete':
                    actionIdx = argmax(self.getqsGivenAgentModel(
                        self.mainModel, env.observationsVectorNormalized))
                    action = self.actionSpace[actionIdx]
                elif env.actionType == 'continuous':
                    action = self.getqsGivenAgentModel(
                        self.mainModel, env.observationsVectorNormalized)
            else:
                # retrieving random action
                if env.actionType == 'discrete':
                    actionIdx = randint(0, self.actionSpaceSize)
                    action = self.actionSpace[actionIdx]
                elif env.actionType == 'continuous':
                    action = list(uniform(low=0.0, high=1.0, size=self.actionSpaceSize))

            new_state, reward, done, truncated, info = env.step(action)
            new_state = env.observationsVectorNormalized

            # updating replay memory
            self.updateReplayMemory(
                [current_state, actionIdx, reward, new_state, done])
            # training main network on every step
            self.train(done, self.gameStep)

            # counting reward
            self.gameReward += reward

            if self.envSettings['RENDER']:
                if not done:
                    if not game+1 % self.envSettings['RENDEREVERY']:
                        env.render()

            # transforming new continous state to new discrete state
            current_state = env.observationsVectorNormalized
            self.gameStep += 1

            if done:
                if env.ENVTYPE in ['0s-c']:
                    env.teardown()
                break

    def crossvalidateDQN(self, env):
        """Simulate a given number of games and cross-validate current DQN
        success.
        """

        # loop to cross-validate on unique set of models
        self.gameRewardsCV = []
        for iGame in range(self.hyParams['NGAMESCROSSVALIDATED']):
            # resetting variables and environment
            self.gameReward, step, done = 0.0, 0, False
            seedCV = self.seedsCV[iGame]
            env.reset(seedCV, initWithSolution=env.initWithSolution)

            current_state = env.observationsVectorNormalized
            # iterating until game ends
            for _ in range(self.hyParams['NAGENTSTEPS']):
                # querying for Q values
                if env.actionType == 'discrete':
                    actionIdx = argmax(self.getqsGivenAgentModel(
                        self.mainModel, env.observationsVectorNormalized))
                    action = self.actionSpace[actionIdx]
                elif env.actionType == 'continuous':
                    action = self.getqsGivenAgentModel(
                        self.mainModel, env.observationsVectorNormalized)
                # simulating and counting total reward
                new_state, reward, done, truncated, info = env.step(action)
                self.gameReward += reward
                if self.envSettings['RENDER']:
                    if not iGame % self.envSettings['RENDEREVERY']:
                        if not done: env.render()
                current_state = new_state
                step += 1
                if done:
                    if env.ENVTYPE in ['0s-c']:
                        env.teardown()
                    break

            # if not env.success:
            #     self.gameReward = 0.0
            self.gameRewardsCV.append(self.gameReward)

        self.average_rewardCV = mean(self.gameRewardsCV)
        self.min_rewardCV = min(self.gameRewardsCV)
        self.max_rewardCV = max(self.gameRewardsCV)

    def runAgentsGenetic(self, agentCounts, env):
        """Run genetic agent optimisation, if opted for with multiprocessing.
        """

        # running in parallel if specified
        # debug: add if available number of CPU cores is exceeded
        cores = self.envSettings['NAGENTSPARALLEL']

        # if __name__ == 'FloPyArcade':
        if 'flopyarcade' in __name__:
            if self.envSettings['SURROGATESIMULATOR'] is not None:
                # removing environment in case of surrogate model
                # as TensorFlow model cannot be pickled
                if hasattr(self, 'env'):
                    del self.env

            t0Runs = time()
            reward_agents, runtimes = [], []
            runtimeGenEstimate, runtimeGensEstimate = None, None
            generationsRemaining = (self.hyParams['NGENERATIONS'] -
                (self.geneticGeneration + 1))
            chunksTotal = self.yieldChunks(agentCounts,
                cores*self.maxTasksPerWorker)
            nChunks = ceil((max(agentCounts)+1)/(cores*self.maxTasksPerWorker))
            nChunks *= (self.hyParams['NGAMESAVERAGED']+1-self.currentGame)
            nChunksRemaining = copy(nChunks)
            # batch processing to avoid memory explosion
            # https://stackoverflow.com/questions/18414020/memory-usage-keep-growing-with-pythons-multiprocessing-pool/20439272
            print('------------------------------')
            for chunk in chunksTotal:
                t0 = time()
                if len(runtimes) == 0:
                    runtimeGenEstimate, runtimeGensEstimate = '?', '?'
                elif len(runtimes) != 0:
                    runtimeGenEstimate = mean(runtimes) * nChunksRemaining
                    runtimeGensEstimate = mean(runtimes) * nChunksRemaining
                    runtimeGensEstimate += ((mean(runtimes) * nChunks) *
                        generationsRemaining)
                    runtimeGenEstimate = '%.3f' % (runtimeGenEstimate/(60*60))
                    runtimeGensEstimate = '%.3f' % (runtimeGensEstimate/(60*60))

                print('Currently: ' + str(min(chunk)) + '/' + 
                      str(max(agentCounts)+1) + ' agents, ' +
                      str(self.currentGame) + '/' +
                      str(self.hyParams['NGAMESAVERAGED']) + ' games, ' +
                      str(self.geneticGeneration + 1) + '/' +
                      str(self.hyParams['NGENERATIONS']) + ' generations\n' +
                      runtimeGenEstimate + ' h for generation, ' +
                      runtimeGensEstimate + ' h for all generations')

                if self.envSettings['ENVTYPE'] in ['0s-c']:
                    # necessary as some ctypes object related to the dll used with BMI is not pickleable
                    try:
                        self.env.teardown()
                    except:
                        pass
                    try:
                        del self.env
                    except:
                        pass

                reward_chunks = self.multiprocessChunks(
                    self.runAgentsGeneticSingleRun, chunk)
                reward_agents += reward_chunks
                runtimes.append(time() - t0)
                nChunksRemaining -= 1
            tRuns = time()-t0Runs
            print('Processing agents took', tRuns, 's')

        return reward_agents

    def runAgentsGeneticSingleRun(self, agentCount):
        """Run single game within genetic agent optimisation."""

        tempAgentPrefix = join(self.tempModelPrefix + '_agent'
            + str(agentCount + 1).zfill(self.zFill))

        # loading specific agent and weights with given ID
        t0load_model = time()

        if self.envSettings['KEEPMODELHISTORY']:
            agent = load_model(join(tempAgentPrefix + '.keras'), compile=False)
        
        else:
            t0RecreateFromMutationHistory = time()
            agentOffset = self.hyParams['NAGENTS'] * (self.geneticGeneration)
            agentNumber = agentCount + 1 + agentOffset
            creationSeed = self.mutationHistory['agent' + str(agentNumber)]['creationSeed']
            mutationSeeds = self.mutationHistory['agent' + str(agentNumber)]['mutationSeeds']
            agent = self.loadModelFromMutationHistory(creationSeed, mutationSeeds)
            tRecreateFromMutationHistory = (self.hyParams['NAGENTS']/self.envSettings['NAGENTSPARALLEL']) * (t0RecreateFromMutationHistory - time())
            if agentCount == 1:
                print('model recreation took about', tRecreateFromMutationHistory, 's')

        MODELNAMETEMP = ('Temp' + self.pid +
            '_' + str(agentCount + 1))

        iGame = self.currentGame-1
        if self.envSettings['SEEDSRANDOM']:
            SEEDTEMP = self.gamesSeeds[self.geneticGeneration][iGame]
        else:
            SEEDTEMP = self.envSettings['SEEDENV'] + iGame
        if self.envSettings['SURROGATESIMULATOR'] is None:
            if self.envSettings['ENVTYPE'] in ['0s-c']:

                lenCount = len(str(agentCount + 1))

                MODELNAMETEMP = self.pid[0:15-lenCount] + str(agentCount + 1)

                MODELNAMETEMP_ = list(MODELNAMETEMP)
                shuffle(MODELNAMETEMP_)
                MODELNAMETEMP = ''.join(MODELNAMETEMP_)

                # loading anew with BMI mode, as environment was not pickleable
                env = FloPyEnv(
                    self.envSettings['ENVTYPE'],
                    self.envSettings['PATHMF2005'],
                    self.envSettings['PATHMP6'],
                    MODELNAME=MODELNAMETEMP,
                    _seed=SEEDTEMP,
                    flagRender=self.envSettings['RENDER'],
                    NAGENTSTEPS=self.hyParams['NAGENTSTEPS'],
                    nLay=self.envSettings['NLAY'],
                    nRow=self.envSettings['NROW'],
                    nCol=self.envSettings['NCOL'],
                    OBSPREP=self.hyParams['NNTYPE'],
                    initWithSolution=True,
                    PATHMF6DLL=None)

            else:
                env = self.env
            # resetting to unique temporary folder to enable parallelism
            # Note: This will resimulate the initial environment state
            env.reset(MODELNAME=MODELNAMETEMP, _seed=SEEDTEMP)
        elif self.envSettings['SURROGATESIMULATOR'] is not None:
            # this must be initialized here as surrogate TensorFlow models
            # cannot be pickled for use in parallel operation
            env = FloPyEnvSurrogate(self.envSettings['SURROGATESIMULATOR'],
                self.envSettings['ENVTYPE'],
                MODELNAME=MODELNAMETEMP, _seed=SEEDTEMP,
                NAGENTSTEPS=self.hyParams['NAGENTSTEPS'])
    
        results, keys = {}, ['trajectories', 'actions', 'rewards', 'wellCoords']
        if self.currentGame == 1:
            trajectories = [[] for _ in range(self.hyParams['NGAMESAVERAGED'])]
            actions = [[] for _ in range(self.hyParams['NGAMESAVERAGED'])]
            rewards = [[] for _ in range(self.hyParams['NGAMESAVERAGED'])]
            wellCoords = [[] for _ in range(self.hyParams['NGAMESAVERAGED'])]
        elif self.currentGame > 1:
            pth = join(tempAgentPrefix + '_results.p')
            results = self.pickleLoad(pth)
            trajectories, actions = results[keys[0]], results[keys[1]]
            rewards, wellCoords = results[keys[2]], results[keys[3]]

        r = 0
        for step in range(self.hyParams['NAGENTSTEPS']):
            if env.actionType == 'discrete':
                actionIdx = argmax(self.getqsGivenAgentModel(agent,
                    env.observationsVectorNormalized))
                action = self.actionSpace[actionIdx]
            if env.actionType == 'continuous':
                action = self.getqsGivenAgentModel(
                    agent, env.observationsVectorNormalized)
            
            # note: need to feed normalized observations
            new_observation, reward, done, truncated, info = env.step(action)
            actions[self.currentGame-1].append(action)
            rewards[self.currentGame-1].append(reward)
            if not env.ENVTYPE in ['0s-c']:
                wellCoords[self.currentGame-1].append(env.wellCoords)
            r += reward
            if self.envSettings['RENDER']:
                env.render()

            if done or (step == self.hyParams['NAGENTSTEPS']-1): # or if reached end
                # saving specific simulation results pertaining to agent
                if not env.ENVTYPE in ['0s-c']:
                    trajectories[self.currentGame-1].append(env.trajectories)
                objects = [trajectories, actions, rewards, wellCoords]
                for i, objectCurrent in enumerate(objects):
                    results[keys[i]] = objectCurrent
                pth = join(tempAgentPrefix + '_results.p')
                self.pickleDump(pth, results)

                if env.ENVTYPE in ['0s-c']:
                    env.teardown()
                break

        return r

    def runAgentsRepeatedlyGenetic(self, agentCounts, n, env):
        """Run all agents within genetic optimisation for a defined number of
        games.
        """

        reward_agentsMin = zeros(len(agentCounts))
        reward_agentsMax = zeros(len(agentCounts))
        reward_agentsMean = zeros(len(agentCounts))
        for game in range(n):
            self.currentGame = game + 1
            print('Currently: ' + str(game + 1) + '/' + str(n) + ' games, ' +
                  str(self.geneticGeneration + 1) + '/' +
                  str(self.hyParams['NGENERATIONS']) + ' generations')
            rewardsAgentsCurrent = self.runAgentsGenetic(agentCounts, env)
            reward_agentsMin = minimum(reward_agentsMin, rewardsAgentsCurrent)
            reward_agentsMax = maximum(reward_agentsMax, rewardsAgentsCurrent)
            reward_agentsMean = add(reward_agentsMean, rewardsAgentsCurrent)
        reward_agentsMean = divide(reward_agentsMean, n)

        prefix = self.tempModelPrefix
        self.pickleDump(prefix + '_agentsRewardsMin.p', reward_agentsMin)
        self.pickleDump(prefix + '_agentsRewardsMax.p', reward_agentsMax)
        self.pickleDump(prefix + '_agentsRewardsMean.p', reward_agentsMean)

        return reward_agentsMean

    def randomAgentGenetic(self, agentIdx, generation=1):
        """Creates an agent for genetic optimisation and saves
        it to disk individually.
        """

        actionType = FloPyEnv(initWithSolution=False).getActionType(self.envSettings['ENVTYPE'])
        agent = self.createNNModel(actionType, seed=self.envSettings['SEEDAGENT']+agentIdx)
        agent.save(join(self.tempModelpth, self.envSettings['MODELNAME'] +
            '_gen' + str(generation).zfill(self.zFill) + '_agent' +
            str(agentIdx + 1).zfill(self.zFill) + '.keras'))

    def returnChildrenGenetic(self, sortedParentIdxs):
        """Mutate best parents, keep elite child and save them to disk
        individually.
        """

        if self.rereturnChildrenGenetic:
            generation = self.geneticGeneration
            tempModelPrefixBefore = self.tempModelPrefix
            tempNextModelPrefixBefore = self.tempNextModelPrefix
            self.tempModelPrefix = self.tempPrevModelPrefix
            self.tempNextModelPrefix = tempModelPrefixBefore
            self.rewards = self.pickleLoad(join(self.tempModelPrefix +
                '_agentsRewardsMean.p'))
        elif not self.rereturnChildrenGenetic:
            # self.tempModelPrefix = self.tempModelPrefix
            generation = self.geneticGeneration + 1

        tempNextModelPrefix = join(self.tempModelpth,
            self.envSettings['MODELNAME'] + '_gen' +
            str(generation+1).zfill(self.zFill))

        if self.searchNovelty:
            recalculateNovelties = False
            try:
                self.novelties
            except Exception as e:
                recalculateNovelties = True
            if len(self.novelties) == 0:
                recalculateNovelties = True

            if recalculateNovelties:
                self.novelties, self.noveltyFilenames = [], []
                for k in range(self.noveltyItemCount):
                    agentStr = 'agent' + str(k+1)
                    self.novelties.append(
                        self.noveltyArchive[agentStr]['novelty'])
                    self.noveltyFilenames.append(
                        self.noveltyArchive[agentStr]['modelFile'])
            self.candidateNoveltyParentIdxs = argsort(
                self.novelties)[::-1][:self.hyParams['NNOVELTYELITES']]

        if self.envSettings['KEEPMODELHISTORY']:
            bestAgent = load_model(join(self.tempModelPrefix + '_agent' +
                str(sortedParentIdxs[0] + 1).zfill(self.zFill) + '.keras'),
                compile=False)
        else:
            agentOffsetBest = self.hyParams['NAGENTS'] * (self.geneticGeneration)
            agentNumberBest = sortedParentIdxs[0] + 1 + agentOffsetBest
            creationSeedBest = self.mutationHistory['agent' + str(agentNumberBest)]['creationSeed']
            mutationSeedsBest = self.mutationHistory['agent' + str(agentNumberBest)]['mutationSeeds']
            bestAgent = self.loadModelFromMutationHistory(creationSeedBest, mutationSeedsBest)

        if not self.rereturnChildrenGenetic:
            bestAgent.save(join(self.tempModelPrefix + '_agentBest.keras'))
        if generation < self.hyParams['NGENERATIONS']:
            if self.envSettings['KEEPMODELHISTORY']:
                bestAgent.save(join(tempNextModelPrefix + '_agent' +
                    str(self.hyParams['NAGENTS']).zfill(self.zFill) + '.keras'))
            nAgentElites = self.hyParams['NAGENTELITES']
            nNoveltyAgents = self.hyParams['NNOVELTYAGENTS']
            self.candidateParentIdxs = sortedParentIdxs[:nAgentElites]

            records = []
            chunksTotal = self.yieldChunks(arange(self.hyParams['NAGENTS']-1),
                self.envSettings['NAGENTSPARALLEL']*self.maxTasksPerWorkerMutate)
            for chunk in chunksTotal:
                records += self.multiprocessChunks(self.returnChildrenGeneticSingleRun,
                    chunk)

            if not self.envSettings['KEEPMODELHISTORY']:
                agentOffset = self.hyParams['NAGENTS'] * (self.geneticGeneration+1)
                # agentNumber = sortedParentIdxs[0] + 1 + agentOffset
                # storing updates to mutation history
                for iRecord, record in enumerate(records):
                    self.mutationHistory['agent' + str(iRecord+1+agentOffset)] = record
                # saving best agent to mutation history
                creationSeed = self.mutationHistory['agent' + str(agentNumberBest)]['creationSeed']
                mutationSeeds = self.mutationHistory['agent' + str(agentNumberBest)]['mutationSeeds']
                recordBest = self.createMutationRecord(creationSeed, mutationSeeds, None)
                self.mutationHistory['agent' + str(self.hyParams['NAGENTS']+agentOffset)] = recordBest

        if self.rereturnChildrenGenetic:
            # resetting temporarily changed prefixes
            self.tempModelPrefix = tempModelPrefixBefore
            self.tempNextModelPrefix = tempNextModelPrefixBefore

    def returnChildrenGeneticSingleRun(self, childIdx):
        """
        """

        len_ = len(self.candidateParentIdxs)
        selected_agent_index = self.candidateParentIdxs[randint(len_)]
        agentPth = join(self.tempModelPrefix + '_agent' +
            str(selected_agent_index + 1).zfill(self.zFill) + '.keras')

        if self.searchNovelty:
            if ((self.geneticGeneration+1) % self.hyParams['ADDNOVELTYEVERY']) == 0:
                remainingElites = self.hyParams['NAGENTS'] - (childIdx+1)
                if self.rereturnChildrenGenetic:
                    generation = self.geneticGeneration
                else:
                    generation = self.geneticGeneration + 1
                if (childIdx+1 ==
                    # remainingElites - self.hyParams['NNOVELTYELITES']):
                    remainingElites - self.hyParams['NNOVELTYAGENTS']):

                    print('Performing novelty evolution after generation',
                        generation)
                # if remainingElites <= self.hyParams['NNOVELTYELITES']:
                if remainingElites <= self.hyParams['NNOVELTYAGENTS']:
                    # selecting a novelty parent randomly
                    len_ = len(self.candidateNoveltyParentIdxs)
                    selected_agent_index = self.candidateNoveltyParentIdxs[randint(len_)]
                    agentPth = self.noveltyFilenames[selected_agent_index]
                if remainingElites == 1:
                    # Keeping agent with highest novelty
                    selected_agent_index = self.candidateNoveltyParentIdxs[int(
                        remainingElites)-1]
                    agentPth = self.noveltyFilenames[selected_agent_index]

        # loading given parent agent, current with retries in case of race
        # condition: https://bugs.python.org/issue36773
        success = False
        while not success:
            try:
                if self.envSettings['KEEPMODELHISTORY']:
                    agent = load_model(agentPth,
                        compile=False)
                else:
                    # agentOffset = self.hyParams['NAGENTS'] * (self.geneticGeneration)
                    # agentNumber = selected_agent_index + 1 + agentOffset
                    agentNumber = selected_agent_index + 1
                    creationSeed = self.mutationHistory['agent' + str(agentNumber)]['creationSeed']
                    mutationSeeds = self.mutationHistory['agent' + str(agentNumber)]['mutationSeeds']
                    agent = self.loadModelFromMutationHistory(creationSeed, mutationSeeds)
                success = True
            except Exception as e:
                success = False
                print('Retrying loading parent agent, possibly due to lock.')
                print(e)
                sleep(1)

        # activating later part makes mutationSeed different every run (not reproducible)
        mutationSeed = self.mutationSeeds[self.geneticGeneration][childIdx]
        # altering parent agent to create child agent
        childrenAgent = self.mutateGenetic(agent, childIdx)

        if self.envSettings['KEEPMODELHISTORY']:
            childrenAgent.save(join(self.tempNextModelPrefix +
                '_agent' + str(childIdx + 1).zfill(self.zFill) + '.keras'))
            return ''
        else:
            creationSeed = self.mutationHistory['agent' + str(agentNumber)]['creationSeed']
            mutationSeeds = self.mutationHistory['agent' + str(agentNumber)]['mutationSeeds']
            record = self.createMutationRecord(creationSeed, mutationSeeds, mutationSeed)
            return record

    def countParameters(self, agent):
        """Count number of policy model parameters."""

        parameters = agent.get_weights()
        nParameters = 0
        for iParam, parameters in enumerate(parameters):
            nParameters += parameters.size

        return nParameters

    def mutateGenetic(self, agent, seed):
        """Mutate single agent model.
        Mutation power is a hyperparameter. Find example values at:
        https://arxiv.org/pdf/1712.06567.pdf
        """

        numpySeed(seed)

        mProb = self.hyParams['MUTATIONPROBABILITY']
        mPower = self.hyParams['MUTATIONPOWER']
        parameters = agent.get_weights()
        for iParam, parameters_ in enumerate(parameters):
            randn_ = mPower * randn(*shape(parameters_))
            mutateDecisions_ = self.mutateDecisions(mProb, shape(parameters_))
            permutation = randn_*mutateDecisions_
            parameters[iParam] = add(parameters_, permutation)
        agent.set_weights(parameters)

        return agent

    def mutateDecision(self, probability):
        """Return boolean defining whether to mutate or not."""

        return random() < probability

    def mutateDecisions(self, probability, shape):
        """Return boolean defining whether to mutate or not."""

        return random(shape) < probability

    def getqsGivenAgentModel(self, agentModel, state):
        """ Query given model for Q values given observations of state.
        predict_on_batch robust in parallel operation?
        """

        state = array(state).reshape(-1, *shape(state))
        # state = array(state).reshape(-1, (*shape(state)))
        prediction = agentModel.predict_on_batch(state)[0]
        return prediction

    def loadAgentModel(self, modelNameLoad=None, compiled=False):
        """Load an agent model."""

        # Note: change model load and save as json mode is faster
        # this is a temporary quickfix

        modelPrefix = join(self.modelpth, modelNameLoad)
        if exists(modelPrefix + '.json'):
            with open(modelPrefix + '.json') as json_file:
                json_config = json_file.read()
            agentModel = model_from_json(json_config)
            agentModel.load_weights(modelPrefix + '.weights.h5')
        if not exists(modelPrefix + '.json'):
            try: agentModel = load_model(modelPrefix + '.model', compile=compiled)
            except: agentModel = load_model(modelPrefix + '.keras', compile=compiled)
            json_config = agentModel.to_json()
            with open(modelPrefix + '.json', 'w') as json_file:
                json_file.write(json_config)
            agentModel.save_weights(modelPrefix + '.weights.h5')

        return agentModel

    def getAction(self, mode='random', keyPressed=None, agent=None,
            modelNameLoad=None, state=None, actionType='discrete'):
        """Determine an action given an agent model.
        Either the action is determined from the player pressing a button or
        chosen randomly. If the player does not press a key within the given
        timeframe for a game, the action remains unchanged.
        Note: mode 'modelNameLoad' can massively throttleneck in loops from
        recurring model loading overhead.
        """

        if actionType == 'discrete':
            if mode == 'manual':
                if keyPressed in self.actionSpace:
                    self.action = keyPressed
                else:
                    self.action = 'keep'
            if mode == 'random':
                actionIdx = randint(0, high=len(self.actionSpace))
                self.action = self.actionSpace[actionIdx]
            if mode == 'modelNameLoad':
                agentModel = self.loadAgentModel(modelNameLoad)
                actionIdx = argmax(self.getqsGivenAgentModel(agentModel, state))
                self.action = self.actionSpace[actionIdx]
            if mode == 'model':
                actionIdx = argmax(self.getqsGivenAgentModel(agent, state))
                self.action = self.actionSpace[actionIdx]

        elif actionType == 'continuous':
            if mode == 'manual':
                print('Regression mode is currently not supported in manual control.')
                quit()
            if mode == 'random':
                self.action = []
                for i in range(self.actionSpaceSize):
                    actionVal = uniform(low=0.0, high=1.0)
                    self.action.append(actionVal)
            if mode == 'modelNameLoad':
                agentModel = self.loadAgentModel(modelNameLoad)
                self.action = self.getqsGivenAgentModel(agentModel, state)
            if mode == 'model':
                self.action = self.getqsGivenAgentModel(agent, state)

        return self.action

    def resumeGenetic(self):
        # checking if bestModel already exists for current generation
        # skipping calculations then to to resume at a later stage
        continueFlag, breakFlag = False, False
        bestAgentpth = join(self.tempModelPrefix + '_agentBest.keras')
        if exists(bestAgentpth):
            indexespth = join(self.tempModelPrefix +
                '_agentsSortedParentIndexes.p')
            noveltyArchivepth = join(self.tempModelPrefix +
                '_noveltyArchive.p')
            self.sortedParentIdxs = self.pickleLoad(indexespth)
            self.noveltyArchive = self.pickleLoad(noveltyArchivepth)
            if not self.envSettings['KEEPMODELHISTORY']:
                self.mutationHistory = self.pickleLoad(join(self.tempModelPrefix + '_mutationHistory.p'))
            self.flagSkipGeneration, continueFlag = True, True

            self.novelties, self.noveltyFilenames = [], []
            for k in range(self.noveltyItemCount):
                agentStr = 'agent' + str(k+1)
                self.novelties.append(
                    self.noveltyArchive[agentStr]['novelty'])
                self.noveltyFilenames.append(
                    self.noveltyArchive[agentStr]['modelFile'])

        # regenerating children for generation to resume at
        else:
            if self.envSettings['KEEPMODELHISTORY']:
                with Pool(1) as executor:

                    if self.envSettings['ENVTYPE'] in ['0s-c']:
                        # necessary as some ctypes object related to the dll used with BMI is not pickleable
                        try:
                            self.env.teardown()
                        except:
                            pass
                        try:
                            del self.env
                        except:
                            pass

                    self.rereturnChildrenGenetic = True
                    self.returnChildrenGenetic(self.sortedParentIdxs)
                    self.rereturnChildrenGenetic = False
            elif not self.envSettings['KEEPMODELHISTORY']:
                print('Resuming impossible with missing model history.')
                breakFlag = True
            # changing resume flag if resuming
            self.envSettings['RESUME'] = False

        return self.sortedParentIdxs, continueFlag, breakFlag

    def actionNoveltyMetric(self, actions1, actions2, actionType):
        # finding largest object, or determining equal length
        # do this conversion while loading or saving?
        actions1 = array(actions1)
        actions2 = array(actions2)

        if actionType == 'discrete':
            if len(actions1) > len(actions2):
                shorterObj, longerObj = actions2, actions1[:len(actions2)]
            elif len(actions1) <= len(actions2):
                shorterObj, longerObj = actions1, actions2[:len(actions1)]
            diffsCount = len(shorterObj) - numpySum(1 for x, y in zip(longerObj, shorterObj) if x == y)

            # enabling this might promote agents having acted longer but not
            # too different to begin with
            # diffsCount += float(numpyAbs(len(longerObj) - len(shorterObj)))

            # dividing by the length of it, to avoid rewarding longer objects
            novelty = diffsCount/len(shorterObj)

        elif actionType == 'continuous':
            # there are values per action in the action space

            # ln = min([len(actions1), len(actions2)])
            # nActionLists = len(actions1[0])
            # diffsCount = ln*nActionLists
            # actions1_, actions2_ = [], []
            # for iStep in range(ln):
            #     actions1_.append(actions1[iStep])
            #     actions2_.append(actions2[iStep])
            # actions1 = list(chain.from_iterable(actions1_))
            # actions2 = list(chain.from_iterable(actions2_))
            # diffs = numpySum(numpyAbs(subtract(actions1, actions2)))
            # novelty = diffs/diffsCount

            # alternative way
            ln = min([len(actions1[:,0]), len(actions2[:,0])])
            nActionLists = len(actions1)
            diffsCount = ln*nActionLists
            diffs = numpySum(numpyAbs(subtract(
                actions1[:ln,:].flatten(),
                actions2[:ln,:].flatten())
                ))
            novelty = diffs/diffsCount

        return novelty

    def calculateNoveltyPerPair(self, args):

        actions = args[0]
        actions2 = args[1]

        # loading per agent becomes a real bottleneck if done repeatedly
        # better to know which needed beforehand
        # tempAgentPrefix = self.noveltyArchive[agentStr]['modelFile'].replace('.keras', '')
        # tempAgentPrefix2 = self.noveltyArchive[agentStr2]['modelFile'].replace('.keras', '')
        # pth = join(tempAgentPrefix + '_results.p')
        # pth2 = join(tempAgentPrefix2 + '_results.p')
        # actions = self.pickleLoad(pth)['actions']
        # actions2 = self.pickleLoad(pth2)['actions']

        # having shared this object across processes will increase used memory too much
        # actions = self.noveltyArchive[agentStr]['actions']
        # actions2 = self.noveltyArchive[agentStr2]['actions']

        novelty = 0.
        for g in range(len(actions)):
            novelty += self.actionNoveltyMetric(actions[g],
                actions2[g], self.actionType)

        return novelty

    def calculateNoveltyPerAgent(self, iAgent, actionsDict=None):

        # args.append([iAgent, sharedArrayActions[rangeLower:rangeHigher], rangeLower, rangeHigher, iAgentInCroppedArray])
        iAgentOriginal, arr, rangeLower, rangeHigher, iAgentInCroppedArray = iAgent[0], iAgent[1], iAgent[2], iAgent[3], iAgent[4]

        arr = self.loadActions(agents=list(range(rangeLower, rangeHigher)))

        agentStr = 'agent' + str(iAgentOriginal+1)
        iAgent = iAgentInCroppedArray

        # loading noveties for specific agent from disk
        noveltyFile = join(self.tempNoveltypth, agentStr + '_novelties.p')
        if exists(noveltyFile):
            agentNovelties = self.pickleLoad(noveltyFile, compressed=None)
        else:
            agentNovelties = {}

        # determine which ones need update?
        # pass necessary actionsDict?
        # load only parts of the noveltyArchive

        # load all actions in main process
        # is this not the last batch missing?
        # collect iAgent2 beforehand and pass corresponding actions in actionsDict
        # load large noveltyArchive
        # iActionsNeeded = []
        # select actions from iActionsNeeded
        # dump the rest of the novelty Archive
        # or use these indices to request from main process?

        # i1 = self.agentsUnique.index(iAgent)
        actions = arr[iAgent]

        novelties = []
        if not self.neighborLimitReached:
            for iAgent2 in range(len(arr)):
                if iAgent != iAgent2:
                    agentStr2 = 'agent' + str(iAgent2+1)
                    actions2 = arr[iAgent2]
                    try:
                        # calculate novelties only if unavailable
                        # as keys mostly exists, try/except check should be performant here
                        novelty = agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+rangeLower+1)]
                    except:
                        novelty = self.calculateNoveltyPerPair([actions, actions2])
                        agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+rangeLower+1)] = novelty
                    novelties.append(novelty)

        elif self.neighborLimitReached:
            iArr = 0
            for iAgent2 in range(rangeLower, rangeHigher):
                if iAgentOriginal != iAgent2:
                    agentStr2 = 'agent' + str(iAgent2+1)
                    # actionsUniqueID2 = self.noveltyArchive[agentStr2]['actionsUniqueID']
                    actions2 = arr[iArr]
                    try:
                        # calculate novelties only if unavailable
                        # as keys mostly exists, try/except check should be performant here
                        novelty = agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+1)]
                    except:
                        novelty = self.calculateNoveltyPerPair([actions, actions2])
                        agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+1)] = novelty
                    novelties.append(novelty)
                iArr += 1

        dumpPath = join(self.tempNoveltypth, agentStr + '_novelties.p')
        self.pickleDump(dumpPath, agentNovelties, compress=None)

        novelty = mean(novelties)

        return novelty

    def saveBestAgent(self, MODELNAME):
        # saving best agent of the current generation
        bestAgent = load_model(join(self.tempModelPrefix + '_agentBest.keras'),
                compile=False)
        self.bestAgentFileName = (f'{MODELNAME}' + '_gen' +
            str(self.geneticGeneration+1).zfill(self.zFill) + '_avg' +
            f'{self.bestAgentReward:_>7.1f}')
        bestAgent.save(join(self.modelpth, self.bestAgentFileName + '.keras'))

    def saveBestAgentAnimation(self, env, bestAgentFileName, MODELNAMEGENCOUNT,
        MODELNAME):
        # playing a game with best agent to visualize progress

        game = FloPyArcade(modelNameLoad=bestAgentFileName,
            ENVTYPE = env.ENVTYPE,
            modelName=MODELNAMEGENCOUNT,
            animationFolder=MODELNAME,
            NAGENTSTEPS=self.hyParams['NAGENTSTEPS'],
            PATHMF2005=self.envSettings['PATHMF2005'],
            PATHMP6=self.envSettings['PATHMP6'],
            flagSavePlot=True,
            flagManualControl=False,
            flagRender=False,
            nLay=env.nLay, nRow=env.nRow, nCol=env.nCol,
            OBSPREP=self.hyParams['NNTYPE'])

        game.play(
            ENVTYPE=self.envSettings['ENVTYPE'],
            seed=self.envSettings['SEEDENV'] + self.currentGame-1)

    def generatePathPrefixes(self):
        self.tempModelPrefix = join(self.tempModelpth,
            self.envSettings['MODELNAME'] + '_gen' +
            str(self.geneticGeneration + 1).zfill(self.zFill))
        self.tempNextModelPrefix = join(self.tempModelpth,
            self.envSettings['MODELNAME'] + '_gen' +
            str(self.geneticGeneration + 2).zfill(self.zFill))
        self.tempPrevModelPrefix = join(self.tempModelpth,
            self.envSettings['MODELNAME'] + '_gen' +
            str(self.geneticGeneration).zfill(self.zFill))

    def yieldChunks(self, lst, n):
        """Yield successive n-sized chunks from a given list.
        Taken from: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def init_arr(self, arr, arr_lock=None, shape=None):
        arr = array(list(arr))
        globals()['arr'] = frombuffer(arr, dtype='float64').reshape(shape)
        globals()['arr_lock'] = arr_lock

    def multiprocessChunks(self, function, chunk, parallelProcesses=None, wait=False, async_=True, sharedArr=None, sharedDict=None):
        """Process function in parallel given a chunk of arguments."""

        # Pool object from pathos instead of multiprocessing library necessary
        # as tensor.keras models are currently not pickleable
        # https://github.com/tensorflow/tensorflow/issues/32159
        if parallelProcesses == None:
            parallelProcesses = self.envSettings['NAGENTSPARALLEL']
        # print('debug parallelProcesses', parallelProcesses)

        if sharedDict == None and sharedArr is None:
            p = Pool(processes=parallelProcesses)
            if async_:
                # imap to use map with a generator and reduce memory impact
                # chunksize = int(max([int(len(chunk)/parallelProcesses), 1]))
                # pasync = p.imap(function, chunk, chunksize=chunksize)
                # pasync = p.imap(function, chunk)
                pasync = p.map_async(function, chunk)
                pasync = pasync.get()
                try:
                    pasync = list(pasync)
                except Exception as e:
                    print('Error in multiprocessing:', e)
            else:
                pasync = p.map(function, chunk)
                # pasync = pasync.get()
            # waiting is important to order results correctly when running
            # -- really?
            # in asynchronous mode (correct reward order is validated)
            if wait:
                pasync.wait()
            p.close()
            p.join()
            p.terminate()

        if sharedDict != None:
            with pathosHelpers.mp.Manager() as manager:
                sharedMgrDict = manager.dict()
                # https://stackoverflow.com/questions/35353934/python-manager-dict-is-very-slow-compared-to-regular-dict
                sharedMgrDict.update(sharedDict)
                with manager.Pool(processes=parallelProcesses) as p:
                    input_ = zip(chunk, repeat(sharedDict, len(chunk)))
                    pasync = p.starmap(function, input_)

        if sharedArr is not None:
            # https://stackoverflow.com/questions/39322677/python-how-to-use-value-and-array-in-multiprocessing-pool
            # https://stackoverflow.com/questions/64222805/how-to-pass-2d-array-as-multiprocessing-array-to-multiprocessing-pool
            # https://stackoverflow.com/questions/1675766/combine-pool-map-with-shared-memory-array-in-python-multiprocessing/58208695#58208695
            # https://stackoverflow.com/questions/11652288/slower-execution-with-python-multiprocessing
            arr = pathosHelpers.mp.Array('d', sharedArr.flatten(), lock=False)
            arr_lock = pathosHelpers.mp.Lock()
            shape_ = shape(sharedArr)
            with Pool(processes=parallelProcesses, initializer=self.init_arr, initargs=(arr, arr_lock, shape_)) as p:
                input_ = zip(chunk, repeat(sharedDict, len(chunk)))
                pasync = p.starmap(function, input_)

        return pasync

    def multiprocessChunks_OLD(self, function, chunk, parallelProcesses=None,
        wait=False, async_=True, sharedArr=None, sharedDict=None, threadPool=False):
        """Process function in parallel given a chunk of arguments."""

        # Pool object from pathos instead of multiprocessing library necessary
        # as tensor.keras models are currently not pickleable
        # https://github.com/tensorflow/tensorflow/issues/32159
        if parallelProcesses == None:
            parallelProcesses = self.envSettings['NAGENTSPARALLEL']

        if sharedDict == None and sharedArr is None:
            if threadPool == False:
                p = Pool(processes=parallelProcesses)
            else:
                p = ThreadPool(processes=parallelProcesses)
            if async_:
                pasync = p.map_async(function, chunk) # , chunksize=10)
                pasync = pasync.get()
            else:
                pasync = p.map(function, chunk)
            if wait:
                pasync.wait()
            p.close()
            p.join()
            p.terminate()

        if sharedDict != None:
            with pathosHelpers.mp.Manager() as manager:
                sharedMgrDict = manager.dict()
                # https://stackoverflow.com/questions/35353934/python-manager-dict-is-very-slow-compared-to-regular-dict
                sharedMgrDict.update(sharedDict)
                with manager.Pool(processes=parallelProcesses) as p:
                    input_ = zip(chunk, repeat(sharedDict, len(chunk)))
                    pasync = p.starmap(function, input_)

        if sharedArr is not None:
            # https://stackoverflow.com/questions/39322677/python-how-to-use-value-and-array-in-multiprocessing-pool
            # https://stackoverflow.com/questions/64222805/how-to-pass-2d-array-as-multiprocessing-array-to-multiprocessing-pool
            # https://stackoverflow.com/questions/1675766/combine-pool-map-with-shared-memory-array-in-python-multiprocessing/58208695#58208695
            # https://stackoverflow.com/questions/11652288/slower-execution-with-python-multiprocessing
            arr = pathosHelpers.mp.Array('d', sharedArr.flatten(), lock=False)
            arr_lock = pathosHelpers.mp.Lock()
            shape_ = shape(sharedArr)
            with Pool(processes=parallelProcesses, initializer=self.init_arr, initargs=(arr, arr_lock, shape_)) as p:
                input_ = zip(chunk, repeat(sharedDict, len(chunk)))
                pasync = p.starmap(function, input_)

        return pasync

    def pickleLoad(self, path, compressed=None):
        """Load pickled object from file."""
        if compressed == None:
            with open(path, 'rb') as f:
                objectLoaded = load(f)
        elif compressed in ['lz4', 'lzma']:
            objectLoaded = joblibLoad(path)
        else:
            print('Unknown compression algorithm specified.')

        return objectLoaded

    def pickleDump(self, path, objectToDump, compress=None, compressLevel=3):
        """Store object to file using pickle."""
        if compress == None:
            with open(path, 'wb') as f:
                dump(objectToDump, f)
        elif compress == 'lz4':
            with open(path, 'wb') as f:
                joblibDump(objectToDump, f, compress=('lz4', compressLevel))
        elif compress == 'lzma':
            with open(path, 'wb') as f:
                joblibDump(objectToDump, f, compress=('lzma', compressLevel))
        else:
            print('Unknown compression algorithm specified.')

    # def GPUAllowMemoryGrowth(self):
    #     """Allow GPU memory to grow to enable parallelism on a GPU."""
    #     config = ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = TFSession(config=config)
    #     K.set_session(sess)

    def GPUAllowMemoryGrowth(self):
        """Allow GPU memory to grow to enable parallelism on a GPU."""
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Enabled memory growth for {len(gpus)} GPU(s).")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"Error setting memory growth: {e}")

    def suppressTensorFlowWarnings(self):
        # suppressing TensorFlow output on import, except fatal errors
        # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
        from logging import getLogger, FATAL
        environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        getLogger('tensorflow').setLevel(FATAL)