#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - render
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




class RenderMixin:

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

    def _plot_base_map(self):
        """Plot the base grid map with light transparency.

        Args:
            self (object): The instance containing the model map to plot.

        """
        self.modelmap.plot_grid(alpha=0.1)

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
        # Plot head array directly on self.ax using pcolormesh
        # (avoids PlotMapView creating a separate figure which breaks blitting)
        heads_raw = np.asarray(self.heads, dtype=np.float64)
        heads_raw = np.squeeze(heads_raw)
        heads_2d = np.ma.masked_where(heads_raw == 999.0, heads_raw)
        x_coords = np.linspace(self.minX, self.minX + self.extentX, self.nCol + 1)
        y_coords = np.linspace(self.minY, self.minY + self.extentY, self.nRow + 1)
        self.headsplot = self.ax.pcolormesh(
            x_coords, y_coords, heads_2d,
            shading='flat', alpha=0.5, zorder=2, cmap=colormaps['terrain']
        )
        # Plot ibound boundary (masked cells) as black patches
        if hasattr(self, 'ibound'):
            from matplotlib.patches import Rectangle
            ibound_arr = np.asarray(self.ibound)
            ibound_2d = ibound_arr[0] if ibound_arr.ndim == 3 else ibound_arr
            for r in range(self.nRow):
                for c in range(self.nCol):
                    if ibound_2d[r, c] <= 0:
                        x0 = self.minX + c * self.dCol
                        y0 = self.minY + r * self.dRow
                        rect = Rectangle(
                            (x0, y0), self.dCol, self.dRow,
                            facecolor='black', alpha=0.7, zorder=3
                        )
                        self.ax.add_patch(rect)
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
        # B15: ideal (max-reward) line is now a solid, more prominent guide
        # instead of a faint dashed reference.
        self.ax2.plot(
            [self.minX, self.minX + self.extentX],
            [self.particleY, self.particleY],
            lw=2.5,
            c='white',
            ls='-',
            zorder=zorder,
            alpha=0.85
        )

    def renderContourLines(self, n=30, zorder=4):
        """Plot n contour lines of the head field.

        Args:
            self (object): The instance containing head data and plotting methods.
            n (int, optional): Number of contour levels. Defaults to 30.
            zorder (int, optional): Drawing order for contours. Defaults to 4.
        
        """
        heads_arr = np.squeeze(np.asarray(self.heads, dtype=np.float64))
        self.levels = linspace(float(np.min(heads_arr)), float(np.max(heads_arr)), n)
        x_coords = np.linspace(self.minX, self.minX + self.extentX, self.nCol)
        y_coords = np.linspace(self.minY, self.minY + self.extentY, self.nRow)
        heads_plot = np.ma.masked_where(heads_arr == 999.0, heads_arr)
        self.contours = self.ax.contour(
            x_coords, y_coords, heads_plot,
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
        # B14: scale outline width with pumping rate
        lw = 1.0 + 4.0 * (abs(quantity) / py_max(abs(min_quantity), 1e-6))
        outline = self._create_outline_circle(center, radius, color, zorder, lw=lw)
        self.ax2.add_artist(outline)

        alpha = self.renderGetWellCircleAlpha(quantity, min_quantity)
        filled = self._create_filled_circle(center, radius, color, alpha, zorder)
        self.ax2.add_artist(filled)

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

    def renderParticle(self, zorder=6):
        """Plot particle at current state.

        Args:
            self (object): The instance of the class.
            zorder (int, optional): Drawing order for the particle. Defaults to 6.
        
        """
        x, y = self._get_particle_coords()
        self._plot_particle(x, y, zorder)

    def _plot_particle(self, x, y, zorder):
        """Plot particle at (x, y) with given zorder.

        Args:
            self (object): The instance of the class.
            x (float): X coordinate of the particle.
            y (float): Y coordinate of the particle.
            zorder (int): Drawing order for the particle.
        
        """
        # B15: the initial spawn marker (timeStep==0) recedes visually
        # (thin + translucent) so the live particle + trajectory dominate.
        if self.timeStep == 0:
            linewidth = 1.5
            alpha = 0.35
        else:
            linewidth = 2
            alpha = 1.0
        self.ax2.scatter(x, y, lw=linewidth, c='red', zorder=zorder, alpha=alpha)

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
            x_raw = np.asarray(self.trajectories['x'][i], dtype=np.float64)
            y_raw = np.asarray(self.trajectories['y'][i], dtype=np.float64)
            if len(x_raw) >= 3:
                # Cumulative arc length as parameter (non-uniform spacing)
                dx = np.diff(x_raw)
                dy = np.diff(y_raw)
                seg = np.sqrt(dx**2 + dy**2)
                s_orig = np.concatenate([[0.0], np.cumsum(seg)])
                if s_orig[-1] < 1e-10:
                    # Zero-length trajectory (well didn't move)
                    xs, ys = x_raw, y_raw
                else:
                    # Deduplicate points to ensure strictly increasing s_orig
                    keep = np.concatenate([[True], np.diff(s_orig) > 1e-12])
                    x_dedup = x_raw[keep]
                    y_dedup = y_raw[keep]
                    s_dedup = s_orig[keep]
                    if len(s_dedup) >= 3:
                        s_new = np.linspace(0.0, s_dedup[-1], len(s_dedup) * 10)
                        from scipy.interpolate import PchipInterpolator
                        xs = PchipInterpolator(s_dedup, x_dedup)(s_new)
                        ys = PchipInterpolator(s_dedup, y_dedup)(s_new)
                    else:
                        xs = np.linspace(x_dedup[0], x_dedup[-1], 20)
                        ys = np.linspace(y_dedup[0], y_dedup[-1], 20)
            elif len(x_raw) == 2:
                # Too few points for spline — linear is fine
                xs = np.linspace(x_raw[0], x_raw[1], 20)
                ys = np.linspace(y_raw[0], y_raw[1], 20)
            else:
                xs, ys = x_raw, y_raw
            self.ax2.plot(xs, ys, lw=2, c=color, zorder=zorder)
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

        # Reset blit artists so next frame creates fresh ones
        if self._blit_enabled:
            self._blit_artists = {'particle': None, 'trajectory': [], 'score_text': None}

    def render_blit_init(self):
        """Initialize blitting by capturing the static background buffer."""
        self.fig.canvas.draw()
        self._blit_background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._blit_artists = {
            'particle': None,
            'trajectory': [],
            'score_text': None,
        }

    def render_blit_frame(self):
        """Render a single frame using blitting."""
        if self._blit_background is None:
            self.render_blit_init()
        # Update dynamic artists on top of the already-drawn environment
        self._blit_update_particle()
        self._blit_update_trajectory()
        self._blit_update_score_text()
        # Full canvas draw since the head array changes every timestep,
        # making a static-background blit approach inapplicable.
        self.fig.canvas.draw()
        # Refresh the cached background so next frame's restore is current
        self._blit_background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _blit_update_particle(self):
        """Update the particle artist for blitting mode."""
        try:
            x, y = self._get_particle_coords()
        except (IndexError, TypeError):
            return
        linewidth = 4 if self.timeStep == 0 else 2
        if self._blit_artists.get('particle') is None:
            self._blit_artists['particle'] = self.ax2.scatter(
                x, y, lw=linewidth, c='red', zorder=6
            )
        else:
            self._blit_artists['particle'].set_offsets(
                self.ax2.transData.transform([(x, y)])
            )

    def _blit_update_trajectory(self):
        """Update the trajectory artists for blitting mode."""
        if self.timeStep <= 0:
            return
        for line in self._blit_artists.get('trajectory', []):
            line.remove()
        self._blit_artists['trajectory'] = []
        total, lengths = self._count_trajectory_points()
        colors = self._generate_fade_colors(total)
        idx = 0
        for i, length in enumerate(lengths):
            color = colors[idx + length - 1]
            line, = self.ax2.plot(
                self.trajectories['x'][i],
                self.trajectories['y'][i],
                lw=2, c=color, zorder=6
            )
            self._blit_artists['trajectory'].append(line)
            idx += length

    def _blit_update_score_text(self):
        """Update the score/time text for blitting mode."""
        score_text = self.format_score()
        time_text = self.format_time()
        display_text = f"{score_text}\ntime: {time_text}"
        if self._blit_artists.get('score_text') is None:
            self._blit_artists['score_text'] = self.ax2.text(
                5, 92, display_text, fontsize=12, zorder=10
            )
        else:
            self._blit_artists['score_text'].set_text(display_text)

    def renderFPS(self, zorder=10):
        """Render FPS counter text in the upper-right corner."""
        import time as _time
        self._frame_times.append(_time.time())
        if len(self._frame_times) < 2:
            return
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed > 0:
            self._fps = (len(self._frame_times) - 1) / elapsed
        else:
            self._fps = 0.0
        fps_text = f"FPS: {self._fps:.1f}"
        if not hasattr(self, '_fps_artist') or self._fps_artist is None:
            self._fps_artist = self.ax2.text(
                95, 92, fps_text, fontsize=12,
                ha='right', va='top', zorder=zorder
            )
        else:
            self._fps_artist.set_text(fps_text)

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

