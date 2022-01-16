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
# twine upload dist\flopyarcade-0.2.19.tar.gz
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
	version='0.3.19',
	description='Simulated groundwater flow environments for reinforcement learning.',
	url='https://github.com/philipphoehn/flopyarcade',
	author='Philipp Hoehn',
	author_email='philipp.hoehn@yahoo.com',
	license='GNU GPLv3',
	packages=find_packages(),
	include_package_data=True,
	install_requires=[
					  'gym==0.21.0',
					  'imageio==2.9.0',
					  'ipython==7.12.0',
					  'joblib==1.0.0',
					  'lz4==3.1.1',
					  'matplotlib==3.2.2',
					  'numpy==1.21.0',
					  'pandas==1.3.5',
					  'pathos==0.2.7',
					  'pillow==9.0.0',
					  'pygame==2.0.1',
					  'ray==1.9.2',
					  'ray[tune]==1.9.2',
					  'ray[rllib]==1.9.2',
					  'scikit-image==0.17.2',
					  'tensorflow==2.7.0', # 2.5.2
					  'tqdm==4.25.0',
					  'xmipy==1.0.0',
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
# necessary as quiet flopy version is not on PyPi, yet
post_install('https://test-files.pythonhosted.org/packages/07/30/b2d5af6d652016bee97b2da5351ab5a49f5c7361f94bc63c49b43cd0bd8e/flopy-3.3.5a3.zip#sha256=98a972f60fd955d4a92802050ce6867afefea74a80deda0568438e2499ab0578')
# post_install('https://test-files.pythonhosted.org/packages/18/c3/2763b5ca540233456b5ec31622df0de7853e2e97a37396e3d125ae1a4345/flopy-3.3.5a2.zip#sha256=34a11333301ecc44c5bfd95c84f9e969d8eb39267a44582130e9a5b083a876af')