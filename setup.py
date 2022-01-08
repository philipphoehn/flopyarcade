from os.path import join
from setuptools import setup
from setuptools import find_packages

# https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/
# https://github.com/MODFLOW-USGS/executables
# compiling with mfpymake currently buggy (as of 07.01.2022)

# testing
# python setup.py sdist
# twine upload --repository-url https://test.pypi.org/legacy/ dist\flopyarcade-0.4.4.tar.gz
# python -m pip install -i https://test.pypi.org/project/ flopyarcade==0.4.5 --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple --no-cache-dir


setup(
	name='flopyarcade',
	version='0.1.0',
	description='simulated groundwater flow environments to test reinforcement learning algorithms',
	url='https://github.com/philipphoehn/flopyarcade',
	author='Philipp Hoehn',
	author_email='philipp.hoehn@yahoo.com',
	license='GNU GPLv3',
	packages=find_packages(),
	package_data={'flopyarcade': [
								  join('*'),
								  join('simulators', '*'),
								  join('simulators', 'linux', '*'),
								  join('simulators', 'mac', '*'),
								  join('simulators', 'win32', '*'),
								  join('simulators', 'win64', '*'),
								  ]},
	install_requires=['flopy==3.3.5a3',
					  'imageio==2.9.0',
					  'ipython==7.12.0',
					  'joblib==1.0.0',
					  'lz4==3.1.1',
					  'matplotlib==3.2.2',
					  'numpy==1.19.4',
					  'pandas==1.3.5',
					  'pathos==0.2.7',
					  'pillow==8.3.2',
					  'pygame==2.0.1',
					  'scikit-image==0.17.2',
					  'tensorflow==2.5.2',
					  'tqdm==4.25.0',
					  'xmipy==1.0.0'
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