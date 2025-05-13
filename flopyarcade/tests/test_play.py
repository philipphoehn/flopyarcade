#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com

# currently on tested stable using python 3.8
# set PYTHONWARNINGS=ignore::ResourceWarning
# python -m coverage run --source=flopyarcade -m unittest

import play

from flopyarcade import FloPyArcade
from glob import glob
from os import remove
from os.path import abspath, dirname, isdir, isfile, join
from shutil import rmtree
import unittest


class TestFloPyEnvPlay(unittest.TestCase):

    def test_FloPyEnvPlay_noExceptionRaised(self):
        """Test a game run of every existing environment."""

        hashes_validity = {
            '0s-c': 'df6244f5e2173e6f0cc5ace5e2986b9b76dd53663bdf35d4ed80d201d72f46ea',
            '1s-d': 'c66de7d08c5fca727ab446d9fefe1345c7af58c5b2e19dc2d7d3b56e36f62542',
            '1s-c': '056cf9b12a64aa7126010c1307dc5e2cb2c7d02f91859ab5faa0db7391e97c79',
            '1r-d': '07adccda23fb6ab043918194796e85700ed1f3214750c6bbad18954515baa1fc',
            '1r-c': 'e0ef7559be635c25f54a96fff9b1cfaacbfbb2c8762fe16d0f2f67fe43bdc751',
            '2s-d': '2a76079382fb3d17e0c462002cb43cbf9a4a042b60db1e80443227dde07f396f',
            '2s-c': '29fd6dc749c82d5ba1ff0e2a5d8d08082ee2da3fa986a73f4d1b89fbb4c34819',
            '2r-d': '3d658564a3d521a1c96de11d79163729cbf13fab8bdbcfd9bb40efcde4d8f393',
            '2r-c': 'eb6fab15915dfcfef44c2dd49004ae089e79c228118a3a0c9bad940eb5ea776c',
            '3s-d': '4cdd2d87e069e1e11ceaae5433dcb8edd8a435009e41cc84b50d2730cc7db759',
            '3s-c': 'f2cc9be7544e11c51554fc1840d98713e1945dcdc91249b302d27407c6c0cf67',
            '3r-d': '1901bc3bea0590b8a856e4d05de1088e8c5ae3db2f627e6659e077f398bc0286',
            '3r-c': '7280ee7dcdab1afb084488f3da4b318a528030338a411eea22d70b017ab395d9',
            '4s-c': 'bf9dd9579331da2695194917bc120ac2ba3120dbab152cd3fa5669cd747f31b2',
            '4r-c': '5767220292952fafdbce1b1bb7c72c213d6f6daa357f802b252097c562d9bb92',
            '5s-c': '3a99015ca0b0d7063453af337cb70b6595d99cbbcd5a23f9574edabfec447994',
            '5s-c-cost': '3a99015ca0b0d7063453af337cb70b6595d99cbbcd5a23f9574edabfec447994',
            '5r-c': '09b4554f60d04e0fec55e04009af94c3816f41343efad184b5a7ae5e2a459a86',
            '6s-c': '6f0d5b979c36f375ea3cf3cff72df9c8daaf8ba3013d3d115a4eea1f75a3061c',
            '6r-c': 'b90dfac9d2d42d8f105209634550ab5a4e0375a6d4d62b0f6803666169355bda'
            }

        failed = []
        # for ENVTYPE in ['0s-c']:
        # for ENVTYPE in ['1s-d']:
        # for ENVTYPE in ['0s-c', '6r-c']:
        # for ENVTYPE in ['1s-d', '6r-c']:
        # for ENVTYPE in ['3r-d']:
        # for ENVTYPE in ['6r-c']:
        for ENVTYPE in FloPyArcade().ENVTYPES:
            raised = False
            # try:
            envSettings = {}
            gameSettings = {}
            # environment settings
            envSettings['ENVTYPE'] = ENVTYPE
            envSettings['MODELNAMELOAD'] = None
            envSettings['MODELNAME'] = 'unittest'
            # self.SAVEPLOT supresses render to be shown properly, as currently mode=='human' and mode=='rgb_array' counteract each other
            envSettings['SAVEPLOT'] = False
            envSettings['SAVEPLOTALLAGENTS'] = False
            envSettings['MANUALCONTROL'] = False
            envSettings['RENDER'] = True
            envSettings['PATHMF2005'] = None
            envSettings['PATHMP6'] = None
            envSettings['SEEDENV'] = 1
            envSettings['NLAY'] = 1
            envSettings['NROW'] = 100
            envSettings['NCOL'] = 100
            # game settings
            gameSettings['NGAMES'] = 1
            gameSettings['NAGENTSTEPS'] = 200 # 2
            envhash = play.main(envSettings, gameSettings)
            # FloPyArcadePlay.main(envSettings, gameSettings)
            assert(envhash == hashes_validity[ENVTYPE])
            print('Test of environment ' + ENVTYPE + ' succeeded.')
            # except Exception as e:
            #     raised = True
            #     failed.append(ENVTYPE)
            #     print('error FloPyArcadePlay', e)
        for ENVTYPE in failed:
            self.assertFalse(ENVTYPE, f'Test of environment {ENVTYPE} failed.')


if __name__ == '__main__':
    unittest.main(exit=False)

    # detected bug: with gameSettings['NAGENTSTEPS'] = 2 reward may be gained from lost games

    # deleting all test-related files, except unittestAgent.h5
    wrkspc = dirname(abspath(__file__))
    unittestFilesAndFolders = glob(join(wrkspc, 'models', 'unittest*'))
    for obj in unittestFilesAndFolders:
        if isdir(obj):
            rmtree(obj)
        elif isfile(obj):
            remove(obj)
    unittestFilesAndFolders = glob(join(wrkspc, 'temp', 'unittest*'))
    for obj in unittestFilesAndFolders:
        if isdir(obj):
            rmtree(obj)
        elif isfile(obj):
            remove(obj)