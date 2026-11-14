#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - modflow_runner
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




class ModflowMixin:

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
        self.MODELNAME = f'{str(uuid4())[:15]}'

    def load_temporary_dll(self, original_dll_path):
        """Copy a DLL to a temporary file and return its path.

        Creates a temporary file with a `.dll` suffix, copies the content of the
        original DLL into it, and returns the temporary file's path. The temporary
        file is not deleted automatically and requires manual cleanup.

        Currently unused, but keeping it in case different thread safety may be
        required later.

        Args:
            self (object): The instance calling this method.
            original_dll_path (str): Path to the original DLL file.

        Returns:
            str: Path to the temporary DLL file.
        
        """

        # Create a temporary file with .dll suffix, delete=False to keep it until manual cleanup
        temp_file = tempfile.NamedTemporaryFile(suffix=".dll", delete=False)
        temp_file.close()  # Close so we can write to it

        # Copy the original DLL content to the temporary file
        shutil.copy2(original_dll_path, temp_file.name)

        return temp_file.name

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
            self.tempDir.cleanup()
            # rmtree(self.model_ws)
            self.teardownSuccess = True
        except Exception as e:
            print(f"Could not remove test {self.model_ws}\nError: {e}")
            self.teardownSuccess = False

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

    def initializeModel(self):
        """Initialize groundwater flow model by constructing its components.

        Args:
            self (object): The instance of the class.

        Calls `constructModel` to build and prepare the model for simulation.

        Assumes:
            `constructModel` initializes all necessary model attributes.
        
        """
        self.constructModel()

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

    def _normalize_basic_stresses(self, vals):
        """Normalize a list of stress values using minH and maxH.

        Args:
            self (object): The instance of the class.
            vals (list or array-like): Stress values to normalize.

        Returns:
            list: Normalized stress values.
        
        """
        return [self._normalize(v, self.minH, self.maxH) for v in vals]

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
            self.mf6 = deepcopy(XmiWrapper(exe))
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
            return self.bmi_return(self.success, self.model_ws, modelname)

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

    def finalize_simulation(self):
        """Finalize the MF6 simulation and mark success.

        Handles exceptions gracefully without raising.

        Args:
            self (object): Instance containing MF6 model and success flag.
        
        """
        self.success = True
        try:
            self.mf6.finalize()
        except Exception as e:
            pass
        # except Exception:
        #     _ = self.bmi_return(self.success, self.model_ws, self.name)

    def post_simulation_teardown(self, teardownOnFinish):
        """Perform post-simulation cleanup and teardown if requested.

        Args:
            self (object): Instance containing model workspace and teardown method.
            teardownOnFinish (bool): Whether to run teardown after finishing.
        
        """

        chdir(self.model_ws)
        if teardownOnFinish:
            self.teardown()

    def updateModel(self):
        """
        Update the model domain for a transient simulation.

        This method reconstructs or refreshes the model setup to reflect changes
        in the simulation domain or parameters for the current time step.

        Args:
            self (object): Instance of the class.
        
        """
        self.constructModel()

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
            # transient convergence failure under memory pressure — retry with backoff
            gc.collect()
            sleep(0.5)
            success, buff = self.mf.run_model(silent=True)
            if not success:
                gc.collect()
                sleep(1.0)
                success, buff = self.mf.run_model(silent=True)
                if not success:
                    raise RuntimeError(
                        'MODFLOW did not terminate normally after retries. '
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

    def _load_grid(self):
        """Load grid file and initialize model grid and PlotMapView.

        Returns:
            None

        """
        fname = join(self.model_ws, f'{self.MODELNAME}.dis.grb')
        grd = MfGrdFile(fname, verbose=False)
        mg = grd.modelgrid
        self.modelmap = PlotMapView(modelgrid=mg, ax=self.ax)

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

