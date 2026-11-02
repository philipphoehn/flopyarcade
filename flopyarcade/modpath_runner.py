#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - modpath_runner
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




class ModpathMixin:

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

    def cleanup_temp_dir(self):
        """Clean up the temporary directory.

        Args:
            self (object): Instance containing tempDir attribute.
        
        """
        self.tempDir.cleanup()

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


    def _updateParticleCoords(self, data, valid):
        """Update particle coordinates to last valid position.

        Args:
            data (pandas.DataFrame): Particle data.
            valid (pandas.Series): Boolean mask for valid data points.

        """
        self.particleCoords = copy([data[c][valid][-1] for c in 'xyz'])

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

