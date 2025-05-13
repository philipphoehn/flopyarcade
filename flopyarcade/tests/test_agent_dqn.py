#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com

# currently on tested stable using python 3.8
# set PYTHONWARNINGS=ignore::ResourceWarning
# python -m coverage run --source=flopyarcade -m unittest

import train_dqn

from flopyarcade import FloPyArcade
from glob import glob
from os import remove
from os.path import abspath, dirname, isdir, isfile, join
from shutil import rmtree
import unittest


class TestFloPyAgentDQN(unittest.TestCase):

    def test_FloPyAgentDQN_noExceptionRaised(self):
        """Test an agent using the deep Q-Learning algorithm on a
        computationally simple case.
        """

        # from train_dqn import envSettings, hyParams
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams

        failed = []
        # for ENVTYPE in FloPyArcade().ENVTYPES:
        for ENVTYPE in ['1s-d']:
            raised = False
            # try:
            # environment settings
            envSettings['ENVTYPE'] = '1s-d'
            envSettings['MODELNAME'] = 'unittest'
            envSettings['RENDER'] = False
            # hyperparameters
            hyParams['NGAMES'] = 4
            hyParams['NAGENTSTEPS'] = 3
            hyParams['REPLAYMEMORYSIZE'] = 10000
            hyParams['REPLAYMEMORYSIZEMIN'] = 3
            hyParams['MINIBATCHSIZE'] = 3
            hyParams['UPDATEPREDICTIVEMODELEVERY'] = 1
            hyParams['NHIDDENNODES'] = [5] * 2
            hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
            hyParams['DROPOUTS'] = [0.0] * 2
            hyParams['CROSSVALIDATEEVERY'] = 2
            hyParams['NGAMESCROSSVALIDATED'] = 2
            train_dqn.main(envSettings, hyParams)
            # FloPyArcadeDQN.main(envSettings, hyParams)
            print('Test of deep Q-network agent in environment ' + ENVTYPE + ' succeeded.')
            # except Exception as e:
            #     raised = True
            #     failed.append(ENVTYPE)
            #     print('error train_dqn:', e)
        for ENVTYPE in failed:
            self.assertFalse(ENVTYPE, 'Test of deep Q-network agent in environment ' + ENVTYPE + ' failed.')


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