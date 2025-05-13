#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com

# currently on tested stable using python 3.8
# set PYTHONWARNINGS=ignore::ResourceWarning
# python -m coverage run --source=flopyarcade -m unittest

import train_neuroevolution

from flopyarcade import FloPyArcade
from glob import glob
from os import remove
from os.path import abspath, dirname, isdir, isfile, join
from shutil import rmtree
import unittest


class TestFloPyAgentGenetic(unittest.TestCase):

    def test_FloPyAgentGenetic_noExceptionRaised(self):
        """Test the genetic algorithm on a computationally simple case."""

        envSettings, hyParams = train_neuroevolution.envSettings, train_neuroevolution.hyParams
        # from train_genetic import envSettings, hyParams
        from matplotlib import use as matplotlibBackend

        failed = []
        for KEEPMODELHISTORY in [True, False]:
            # for NNTYPE in ['perceptron']:
            # for NNTYPE in ['convolution']:
            for NNTYPE in ['convolution', 'perceptron']:
                # for ENVTYPE in FloPyArcade().ENVTYPES:
                # for ENVTYPE in ['1s-d', '2r-d', '3s-c', '4r-c', '5r-c', '6r-c']:
                for ENVTYPE in ['1s-d', '6r-c']:
                # for ENVTYPE in ['1s-d', '1s-c']:
                    raised = False
                    # try:
                    matplotlibBackend('Agg')
                    # environment settings
                    envSettings['ENVTYPE'] = ENVTYPE
                    envSettings['MODELNAME'] = 'unittest_' + ENVTYPE
                    envSettings['NAGENTSPARALLEL'] = 2
                    envSettings['RENDER'] = False
                    envSettings['BESTAGENTANIMATION'] = True
                    envSettings['KEEPMODELHISTORY'] = KEEPMODELHISTORY
                    envSettings['RESUME'] = False
                    # hyperparameters
                    hyParams['NAGENTS'] = 4
                    hyParams['NAGENTELITES'] = 3
                    hyParams['NGENERATIONS'] = 2
                    hyParams['NGAMESAVERAGED'] = 2
                    hyParams['NAGENTSTEPS'] = 2
                    hyParams['NNTYPE'] = NNTYPE
                    hyParams['NHIDDENNODES'] = [5] * 2
                    hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
                    hyParams['BATCHNORMALIZATION'] = True
                    hyParams['NOVELTYSEARCH'] = True
                    hyParams['ADDNOVELTYEVERY'] = 1
                    hyParams['NNOVELTYELITES'] = 2
                    hyParams['NNOVELTYNEIGHBORS'] = 5
                    train_neuroevolution.main(envSettings, hyParams)
                    # FloPyArcadeGeneticNetwork.main(envSettings, hyParams)
                    print('Genetic agent in environment ' + ENVTYPE + ' succeeded.')
                    # except Exception as e:
                    #     raised = True
                    #     failed.append(ENVTYPE)
                    #     print('error train_neuroevolution', e)
        for ENVTYPE in failed:
            self.assertFalse(ENVTYPE, 'Genetic agent in environment ' + ENVTYPE + ' failed.')


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