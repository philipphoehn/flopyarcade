#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - env_base
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
getLogger('flopy').setLevel(FATAL)
logger = getLogger(__name__)

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




class EnvBaseMixin:

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
        self._init_env_vars(self.env_config.get('_seed', None))
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
            '_seed': local_vars.get('_seed'),
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
        self._blit_background = None
        self._blit_enabled = False
        self._blit_artists = {}
        self._fps = 0.0
        from collections import deque as _deque
        self._frame_times = _deque(maxlen=60)
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

        # self.mf6dll_orig = deepcopy(self.mf6dll)
        # self.mf6dll = self.load_temporary_dll(self.mf6dll_orig)

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
            low=-25.0,
            high=25.0,
            shape=obs_array.shape,
            dtype=np.float64
        )

        if self.actionType == 'discrete':
            self.action_space = gymnasium.spaces.Discrete(len(self.actionSpace))
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
        if seed is not None:
            self._SEED = seed
            np.random.seed(seed)
            import random as _pyrandom
            _pyrandom.seed(seed)
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
            except ImportError:
                pass
        else:
            np.random.seed(seed)
        # Example: to seed other random generators, add here
        # import random
        # random.seed(seed)

    def close(self):
        """Mark the resource as closed and release accumulated memory.

        Sets the `closed` attribute to True and runs garbage collection
        to free numpy arrays and MODFLOW objects no longer referenced.

        Args:
            self (object): The instance of the class.
        
        """
        self.closed = True
        gc.collect()

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

        # self._finalize_model()
        self.finalize_simulation
        self._remove_model_workspace()
        assert self.teardownSuccess, f"Failed to remove directory {self.model_ws}"

    def _change_to_workspace(self):
        """Change the current working directory to the workspace directory.

        Args:
            self (object): The instance of the class.
        
        """
        chdir(self.wrkspc)

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

        # only specific ens
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

    def _setup_general_env(self):
        """Set up general (non-0s-c) environment configuration.

        Initializes spatial extents, layering, head specifications, well parameters,
        and observation settings.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        self.rewardMax = 1000
        self.distanceMax = 98
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
        info = {}

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
            self._obs_hasher = hashlib.sha256()
            self._feed_obs_hash(self.observationsVectorNormalized)
            if self.renderFlag or self.SAVEPLOT:
                self.render()
        else:
            self.runMODFLOW()
            self.update_state()
            self.get_normalized_observations()
            self.get_normalized_stresses()
            self.observations = np.array(self.observationsVectorNormalized)
            self._obs_hasher = hashlib.sha256()
            self._feed_obs_hash(self.observations)
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

        self._prepare_time_step()
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
        self._feed_obs_hash(self.observationsVectorNormalized)

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

        return self.observationsVectorNormalized, self.reward, self.done, self.truncated, self.info

    def _step_apply_action(self, action):
        """Pipeline stage 1: apply action to model."""
        self.setActionValue(action)
        self.initializeState(self.state)
        self.updateModel()
        self.updateWellRate()
        self.updateWell()

    def _step_run_simulation(self):
        """Pipeline stage 2: run MODFLOW + MODPATH."""
        self.runMODFLOW()
        self.runMODPATH()
        self.evaluateParticleTracking()

    def _step_evaluate_reward(self):
        """Pipeline stage 3: calculate reward."""
        self.calculate_reward()

    def _step_update_state(self):
        """Pipeline stage 4: update state + observations."""
        self.update_state()
        self.get_normalized_observations()
        self.get_normalized_stresses()
        self.check_particle_state()

    def _step_finalize(self):
        """Pipeline stage 5: hash, render, bookkeeping."""
        self.rewardCurrent += self.reward
        if self.RENDER or self.MANUALCONTROL or self.SAVEPLOT:
            self.render()
        self.observations = np.array(self.observationsVectorNormalized)
        self._feed_obs_hash(self.observations)
        self.timeStep += 1
        self.info = {
            'timestep': self.timeStep,
            'max_steps': self.maxSteps,
            'action': self._last_action,
        }
        self.update_envhash()

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

        # --- Step pipeline (named stages) ---
        self._last_action = action
        self._step_apply_action(action)    # 1. action to model
        self._step_run_simulation()         # 2. solve + track
        self._step_evaluate_reward()        # 3. reward
        self._step_update_state()           # 4. obs + state
        self._step_finalize()               # 5. hash + bookkeeping

        if self.done and teardownOnFinish:
            self.tempDir.cleanup()

        return self.observations, self.reward, self.done, self.truncated, self.info

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

    def _prepare_time_step(self):
        """Prepare the model for the current time step.

        Args:
            self (object): Instance of the class containing the model and time step.
        
        """
        self.mf6.prepare_time_step(self.dt)
        self.current_time = self.mf6.get_current_time()

    def collect_Qs(self):
        """Collect Q values for all helper wells.

        Args:
            self (object): Instance containing helper wells and action values.

        Returns:
            list[float]: List of Q values for each helper well.
        
        """
        Qs = [self.get_action_value(i, 'Q') for i in range(self.nHelperWells)]
        return Qs

    def get_max_iterations(self):
        """Retrieve maximum iterations from MF6 solver settings. Sets self.max_iter to the maximum number of solver iterations.
        
        Args:
            self (object): Instance containing MF6 model interface.

        """
        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        self.max_iter = int(self.mf6.get_value(mxit_tag).item())

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
        self.envhash = self._obs_hasher.hexdigest()

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

        # try:
        #     unlink(self.mf6dll)
        #     remove(self.mf6dll)
        # except Exception as e:
        #     print('dll not removed or not removable?')

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

    def setup_rch_0sc(self, recharge):
        """Set up the recharge (RCH) package.

        Applies recharge to the groundwater flow model.

        Args:
            self (object): Instance containing the groundwater flow model.
            recharge (float or array-like): Recharge rate(s) to apply.

        """
        ModflowGwfrcha(self.gwf, recharge=recharge)

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


    def _sum_flows(self):
        """Sum pump and recharge flows from helper wells.

        Returns:
            tuple: (pump, recharge) total flow values.

        """
        pump = recharge = 0.0
        hw = self.helperWells
        for i in range(self.nHelperWells):
            Q = hw[f'wellQ{i + 1}']
            if Q <= 0:
                pump += -Q
            else:
                recharge += Q
        return pump, recharge

    def reset(self, seed=None, MODELNAME=None, initWithSolution=None, options=None):
        """Reset environment with same settings and optional new seed.

        Args:
            _seed (int, optional): Random seed for environment reset.
            MODELNAME (str, optional): Model name for reset.
            initWithSolution (optional): Initial solution to use.

        Returns:
            tuple: (observationsVectorNormalized as numpy array, empty dict)

        """
        if seed is None:
            seed = getattr(self, '_SEED', None)

        initWithSolution = self._get_init_solution(initWithSolution)

        if self.ENVTYPE == '0s-c':
            self._reset_0s_c(seed)
        else:
            self._reset_other(seed, MODELNAME, initWithSolution)

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

        self._setup_workspace()
        self._init_env_vars(_seed)
        self._init_env_specific()

        # self.mf6dll = self.load_temporary_dll(self.mf6dll_orig)

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

    def _handle_display_mode(self):
        """Handle rendering when return_figure is False.

        Args:
            self (object): The instance managing interactive and display rendering.
        
        """
        if self.MANUALCONTROL:
            self.renderUserInterAction()
        elif self.RENDER:
            if self._blit_enabled:
                self.render_blit_frame()
            else:
                self.fig.canvas.draw()
            self.renderFPS()
            show(block=False)
            pause(self.MANUALCONTROLTIME)
        self.renderClearAxes()

    def _detect_ipython(self):
        """Detect if running inside an IPython notebook environment.
        
        Args:
            self (object): The instance where the flag will be set.
        
        """
        self.flagFromIPythonNotebook = 'ipykernel' in modules

    def _setup_axes(self):
        """Set up the main axes and twin axes for the figure.

        Args:
            self (object): The instance containing the figure to set up axes on.
        
        """
        self.ax = self.fig.add_subplot(1, 1, 1, aspect='equal', adjustable='box')
        self.ax3 = self.ax.twinx()
        self.ax2 = self.ax3.twiny()

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

    def _create_outline_circle(self, center, radius, color, zorder, lw=1.0):
        """Create an outline circle patch.

        Args:
            self (object): The instance creating the circle patch.
            center (tuple): Coordinates of the circle center.
            radius (float): Radius of the circle.
            color: Color of the circle outline.
            zorder (int): Drawing order.
            lw (float): Line width, scaled by pumping rate.

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
            lw=lw
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

    def _get_game_result(self) -> str:
        """Return the game outcome string.

        Args:
            self (object): The instance of the class.

        Returns:
            str: 'Success' if the game was successful, otherwise 'Failure'.
        
        """
        return 'Success' if self.success else 'Failure'

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

    def format_time(self) -> str:
        """Return formatted time string in days.

        Args:
            self (object): The instance of the class.

        Returns:
            str: Time in days, formatted as '<days> d'.
        
        """
        time_days = self.timeStep * self.periodLength
        return f"{time_days:.0f} d"

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

