#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - actions
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




class ActionsMixin:

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

    def _setup_action_space(self):
        """Configure the action space based on environment and action type.

        Sets `actionSpace` and `actionSpaceSize` according to `ENVTYPE` and `actionType`.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        
        """
        from .env_config import ENV_CONFIG
        cfg = ENV_CONFIG.get(self.ENVTYPE)
        if cfg is None:
            pass
        elif self.actionType == 'discrete':
            if cfg['action_discrete'] is not None:
                self.actionSpace = cfg['action_discrete']
            else:
                self._setup_discrete_multiwell_actions()
        elif self.actionType == 'continuous':
            if cfg['action_continuous'] is not None:
                self.actionSpace = cfg['action_continuous']
            else:
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

    def _update_helper_wells(self, keys):
        """Update state for helper wells with specified keys.

        Args:
            self (object): The instance of the class.
            keys (list of str): List of keys to update the state with.
        
        """
        hw = self.helperWells
        state = self.state
        for i in range(self.nHelperWells):
            w = str(i + 1)
            for key in keys:
                sk = f'actionValue{key}{w}'
                state[sk] = hw[sk]

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

    def update_well_Q(self):
        """Update the well Q values.

        Args:
            self (object): Instance containing well data and Q values.
        
        """
        self.well[:, 0] = self.collect_Qs()

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
        env = self.ENVTYPE
        if env in ('5s-c', '5s-c-cost', '5r-c'):
            dQ_idx_more, dQ_idx_less = 0, 1
        elif env in ('6s-c', '6r-c'):
            dQ_idx_more, dQ_idx_less = 4, 5
        else:
            dQ_idx_more, dQ_idx_less = -1, -1

        hw = self.helperWells
        for i in range(self.nHelperWells):
            w = str(i + 1)
            offset = i * len(self.actionSpaceIndividual)

            if dQ_idx_more >= 0:
                dQ = action[offset + dQ_idx_less] * self.actionRangeQ - action[offset + dQ_idx_more] * self.actionRangeQ
            else:
                dQ = 0

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
        env = self.ENVTYPE
        method = self._ACTION_SINGLE.get(env)
        if method:
            getattr(self, method)(action)
        elif env in self._MULTI_C:
            self.update_action_multi_wells_c_position(action)
            self.update_action_multi_wells_c_rate(action)
        else:
            self.update_action_multi_wells_d(action)

