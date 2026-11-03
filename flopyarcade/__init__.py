"""
FloPyArcade.

Simulated groundwater flow environments to test reinforcement learning algorithms.
"""

__version__ = '0.7.0'
__author__ = 'Philipp Hoehn'


from os import chdir, remove
from os.path import dirname, exists, join, realpath
from zipfile import ZipFile


# unarchiving simulators
wrkspc = dirname(realpath(__file__))
if not exists(join(wrkspc, 'simulators', 'linux')):
	print('Unarchiving simulation engines on first import. This might take a moment.')
	chdir(join(wrkspc, 'simulators'))
	with ZipFile(join(wrkspc, 'simulators', 'simulators_linux.zip'), 'r') as zipObj:
		zipObj.extractall()
	remove(join(wrkspc, 'simulators', 'simulators_linux.zip'))

if not exists(join(wrkspc, 'simulators', 'mac')):
	chdir(join(wrkspc, 'simulators'))
	with ZipFile(join(wrkspc, 'simulators', 'simulators_mac.zip'), 'r') as zipObj:
		zipObj.extractall()
	remove(join(wrkspc, 'simulators', 'simulators_mac.zip'))

if not exists(join(wrkspc, 'simulators', 'win32')):
	chdir(join(wrkspc, 'simulators'))
	with ZipFile(join(wrkspc, 'simulators', 'simulators_win32.zip'), 'r') as zipObj:
		zipObj.extractall()
	remove(join(wrkspc, 'simulators', 'simulators_win32.zip'))

if not exists(join(wrkspc, 'simulators', 'win64')):
	chdir(join(wrkspc, 'simulators'))
	with ZipFile(join(wrkspc, 'simulators', 'simulators_win64.zip'), 'r') as zipObj:
		zipObj.extractall()
	remove(join(wrkspc, 'simulators', 'simulators_win64.zip'))


from .flopyarcade import FloPyEnv
from .arcade import FloPyArcade

# Solvers are optional — require tensorflow (pip install flopyarcade[solvers])
try:
    from flopyarcade_solvers import FloPyAgent
except ImportError:
    try:
        from .agent import FloPyAgent
    except ImportError:
        FloPyAgent = None

__all__ = ['FloPyEnv', 'FloPyArcade', 'FloPyAgent']