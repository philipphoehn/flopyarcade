#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - observations
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




class ObservationsMixin:

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

    def update_state(self):
        """Update the environment state based on the environment type.

        Args:
            self (object): The instance of the class.
        
        """
        self.state = {'heads': self.heads}

        from .env_config import ENV_CONFIG
        cfg = ENV_CONFIG.get(self.ENVTYPE)
        if cfg is None:
            return
        if cfg['helper_keys'] is not None:
            self._update_helper_wells(cfg['helper_keys'])
        else:
            getattr(self, cfg['state_update'])()

    def _init_observations(self):
        """Initialize observation dictionaries.

        Args:
            self (object): The instance of the class.
        
        """
        self.observations = {}
        self.observationsNormalized = {}
        self.observationsNormalizedHeads = {}

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
            self.observations['heads'] += self.observations['headsSampledField'].flatten().tolist()
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

    def _normalize_helper_wells(self):
        """Normalize parameters for all helper wells and return as a flat list.

        Args:
            self (object): The instance of the class.

        Returns:
            list: Normalized values of wellQ, wellX, wellY, wellZ for all helper wells.
        
        """
        hw = self.helperWells
        stresses = []
        extend = stresses.extend
        for i in range(self.nHelperWells):
            idx = str(i + 1)
            extend(self._normalize_well(hw[f'wellQ{idx}'], hw[f'wellX{idx}'], hw[f'wellY{idx}'], hw[f'wellZ{idx}']))
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
        parts = [
            self.normalize(self.head, self.chd_east, self.chd_diff).flatten(),
            self.normalize(new_recharge, self.avg_rch_min, self.rch_diff).flatten(),
        ]
        if observations_wells:
            parts.append(np.array(observations_wells))
        if observations_storms:
            parts.append(np.array(observations_storms))
        return np.concatenate(parts).tolist()

    def _feed_obs_hash(self, observation):
        """Feed a single observation's normalized bytes into the incremental hasher.

        Args:
            self (object): Instance with _obs_hasher attribute.
            observation (np.ndarray): Single observation array.
        
        """
        feature = self.example_feature_extractor(observation)
        self._obs_hasher.update(feature.ravel().tobytes())

    def _append_helper_wells_perceptron(self, observationsVector, observationsDict):
        """Append helper well data for perceptron OBSPREP.

        Args:
            self (object): The instance of the class.
            observationsVector (list): Vector to append data to.
            observationsDict (dict): Dictionary of observations.
        
        """
        obsDict = observationsDict
        for i in range(self.nHelperWells):
            iStr = str(i)
            kQ = f'wellQ{iStr}'
            if kQ in obsDict:
                observationsVector.append(obsDict[kQ])
            kC = f'wellCoords{iStr}'
            if kC in obsDict:
                observationsVector.extend(obsDict[kC])

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

        if valsExtra:
            shape = observationsDict['heads'].shape[1:]
            extra = np.zeros((len(valsExtra),) + shape, dtype=observationsVector.dtype)
            for idx, val in enumerate(valsExtra):
                extra[idx, 0, 0] = val
            observationsVector = np.concatenate((observationsVector, extra), axis=0)

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
        return (array - zerooffset) / diff

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


