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

setup(
	name='flopyarcade',
	version='0.2.19',
	description='Simulated groundwater flow environments for reinforcement learning.',
	url='https://github.com/philipphoehn/flopyarcade',
	author='Philipp Hoehn',
	author_email='philipp.hoehn@yahoo.com',
	license='GNU GPLv3',
	packages=find_packages(),
	include_package_data=True,
	install_requires=['flopy==3.3.5a3',
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
					  'scikit-image==0.17.2',
					  'tensorflow==2.7.0', # 2.5.2
					  'tqdm==4.25.0',
					  'xmipy==1.0.0'
					  ],
	dependency_links=['https://pypi.org/', # simple/
                      'https://test.pypi.org/'
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