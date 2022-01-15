"""
FloPyArcade.

Simulated groundwater flow environments to test reinforcement learning algorithms.
"""

__version__ = '0.2.0'
__author__ = 'Philipp Hoehn'


from os import chdir, remove
from os.path import dirname, exists, join, realpath
from zipfile import ZipFile


# unarchiving simulators
wrkspc = dirname(realpath(__file__))
chdir(wrkspc)
if not exists(join(wrkspc, 'simulators')):
	print('Unarchiving simulation engines on first import. This might take a moment.')
	with ZipFile(join(wrkspc, 'simulators.zip'), 'r') as zipObj:
		zipObj.extractall()
	remove(join(wrkspc, 'simulators.zip'))


from .flopyarcade import FloPyAgent
from .flopyarcade import FloPyEnv
from .flopyarcade import FloPyArcade