# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all
import matplotlib

platform = sys.platform  # 'linux', 'darwin', 'win32'

# Dynamic matplotlib data path (platform-agnostic)
mpl_data = matplotlib.get_data_path()

datas = [
    ('simulators', 'simulators'),
    (mpl_data, 'mpl-data'),
]

binaries = []
hiddenimports = [
    'tensorflow', 'tensorflow.keras', 'tensorflow.keras.layers',
    'tensorflow.keras.models', 'tensorflow.keras.optimizers',
    'tensorflow.keras.initializers', 'tensorflow.compat.v1',
    'flopy', 'flopy.modflow', 'flopy.modpath', 'flopy.plot', 'flopy.utils',
    'pathos', 'pathos.pools', 'pathos.helpers',
    'imageio', 'matplotlib.backends.backend_agg',
    'pandas', 'pandas.io.formats.style'
]

# Collect TensorFlow (handles platform-specific binaries automatically)
tmp_ret = collect_all('tensorflow')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Excludes (reduce size)
excludes = [
    'nbformat', 'notebook', 'torch', 'cupy', 'cupyx', 'cupy_backends',
    'sklearn', 'pyarrow', 'xarray', 'numba', 'llvmlite', 'dask',
    'distributed', 'bokeh', 'IPython', 'ipykernel', 'jupyter'
]
# On Linux headless, exclude tkinter; on Mac keep it for optional display
if platform == 'linux':
    excludes += ['tkinter', 'matplotlib.backends.backend_tkagg', 'matplotlib.backends._backend_tk']

a = Analysis(
    ['FloPyArcadePlay.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FloPyArcadePlay',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
