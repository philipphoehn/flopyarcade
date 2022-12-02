from os.path import join
from setuptools import setup
from setuptools import find_packages

# https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/

# compiling with mfpymake currently buggy (as of 07.01.2022)
# use https://github.com/MODFLOW-USGS/executables

# https://stackoverflow.com/questions/24727709/do-python-projects-need-a-manifest-in-and-what-should-be-in-it

# inofficial for testing
# python setup.py sdist
# twine upload --repository-url https://test.pypi.org/legacy/ dist\flopyarcade-0.5.32.tar.gz
# while flopy 3.3.5a not official
# python -m pip install -i https://test.pypi.org/project/ flopyarcade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple --no-cache-dir
# as soon as flopy 3.3.5a not official
# python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple flopyarcade --no-cache-dir

# official
# python setup.py sdist
# twine upload dist\flopyarcade-0.3.31.tar.gz
# python -m pip install flopyarcade

# using dependency-links did not work (anymore with this pip version)
# https://stackoverflow.com/questions/12518499/pip-ignores-dependency-links-in-setup-py
# https://github.com/pypa/pip/pull/5571

import subprocess
import sys

def post_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

setup(
	name='flopyarcade',
	version='0.3.33',
	description='Simulated groundwater flow environments for reinforcement learning.',
	url='https://github.com/philipphoehn/flopyarcade',
	author='Philipp Hoehn',
	author_email='philipp.hoehn@yahoo.com',
	license='GNU GPLv3',
	packages=find_packages(),
	include_package_data=True,
	install_requires=[
					  'flopy==3.3.5',
					  # https://stackoverflow.com/questions/71411045/how-to-solve-module-gym-wrappers-has-no-attribute-monitor
					  'gym==0.22.0', # update breaks playbenchmark
					  'imageio==2.22.4',
					  'ipython==7.34.0', # 8.7.0 on newer Python versions
					  'joblib==1.2.0',
					  'lz4==4.0.2',
					  'matplotlib==3.5.3', # 3.6.2 on newer Python versions
					  # https://stackoverflow.com/questions/73929564/entrypoints-object-has-no-attribute-get-digital-ocean
					  'importlib-metadata==4.13.0', # update breaks
					  'numpy==1.21.6', # 1.22.0 on newer Python versions
					  'pandas==1.3.5', # 1.5.2 on newer Python versions
					  'pathos==0.3.0',
					  'pillow==9.3.0',
					  'pygame==2.1.2',
					  'ray==1.9.2', # 2.1.0 requires new benchmark checkpoints
					  'ray[tune]==1.9.2', # 2.1.0 requires new benchmark checkpoints
					  'ray[rllib]==1.9.2', # 2.1.0 requires new benchmark checkpoints
					  'scikit-image==0.19.3',
					  'tensorflow==2.11.0',
					  'tqdm==4.64.1',
					  'xmipy==1.0.0', # 1.2.0 # update breaks
					  ],

	classifiers=[
		'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Education',
        ]
	)

# https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
# not necessary anymore, as currently flopy==3.3.5 became available on PyPi
# post_install('https://test-files.pythonhosted.org/packages/07/30/b2d5af6d652016bee97b2da5351ab5a49f5c7361f94bc63c49b43cd0bd8e/flopy-3.3.5a3.zip#sha256=98a972f60fd955d4a92802050ce6867afefea74a80deda0568438e2499ab0578')
# post_install('https://test-files.pythonhosted.org/packages/18/c3/2763b5ca540233456b5ec31622df0de7853e2e97a37396e3d125ae1a4345/flopy-3.3.5a2.zip#sha256=34a11333301ecc44c5bfd95c84f9e969d8eb39267a44582130e9a5b083a876af')
