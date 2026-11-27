"""
FloPyArcade.

Simulated groundwater flow environments to test reinforcement learning algorithms.
"""

__version__ = "0.7.26"
__author__ = 'Philipp Hoehn'


from os import chdir, remove
from os.path import dirname, exists, join, realpath
from zipfile import ZipFile
import traceback


# unarchiving simulators (skip on read-only installs)
wrkspc = dirname(realpath(__file__))
for _zip_name, _subfolder in [
    ('simulators_linux.zip', 'linux'),
    ('simulators_mac.zip', 'mac'),
    ('simulators_win32.zip', 'win32'),
    ('simulators_win64.zip', 'win64'),
]:
    _target = join(wrkspc, 'simulators', _subfolder)
    _zip_path = join(wrkspc, 'simulators', _zip_name)
    if not exists(_target) and exists(_zip_path):
        try:
            chdir(join(wrkspc, 'simulators'))
            with ZipFile(_zip_path, 'r') as zipObj:
                zipObj.extractall()
            remove(_zip_path)
        except (OSError, PermissionError) as e:
            print(f'flopyarcade: could not extract {_zip_name}: {e}')


try:
    from .flopyarcade import FloPyEnv
except ImportError as e:
    print(f'flopyarcade WARNING: FloPyEnv unavailable: {e}')
    traceback.print_exc()
    FloPyEnv = None

try:
    from .arcade import FloPyArcade
except ImportError as e:
    print(f'flopyarcade WARNING: FloPyArcade unavailable: {e}')
    FloPyArcade = None

# Solvers are optional
FloPyAgent = None
try:
    from flopyarcade_solvers import FloPyAgent
except ImportError:
    try:
        from .agent import FloPyAgent
    except ImportError as e:
        print(f'flopyarcade WARNING: FloPyAgent unavailable: {e}')
        FloPyAgent = None

__all__ = ['FloPyEnv', 'FloPyArcade', 'FloPyAgent']
