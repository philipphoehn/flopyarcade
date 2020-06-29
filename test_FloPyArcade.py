#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


import FloPyArcadePlay
import FloPyArcadeDQN
import FloPyArcadeGeneticNetwork
from glob import glob
from os import remove
from os.path import abspath, dirname, isdir, isfile, join
from shutil import rmtree
import unittest


class TestFloPyEnvPlay(unittest.TestCase):

    def test_FloPyEnvPlay_noExceptionRaised(self):
        """Test a game run of every existing environment."""

        failed = []
        for i in range(1, 4):
            raised = False
            try:
                from FloPyArcadePlay import envSettings, gameSettings
                envSettings['ENVTYPE'] = str(i)
                # envSettings['MODELNAMELOAD'] = 'unittestAgent' + str(i)
                envSettings['MODELNAMELOAD'] = None
                envSettings['SAVEPLOT'] = False
                envSettings['MANUALCONTROL'] = False
                envSettings['RENDER'] = False
                gameSettings['NGAMES'] = 1
                gameSettings['NAGENTSTEPS'] = 2
                FloPyArcadePlay.main(envSettings, gameSettings)
                print('Successfully tested environment FloPyEnv' + str(i) + '.')
            except Exception as e:
                raised = True
                failed.append(i)
                print('error FloPyArcadePlay', e)
        for i in failed:
            self.assertFalse(raised, 'FloPyEnv' + str(i) + ' failed.')


class TestFloPyAgentDQN(unittest.TestCase):

    def test_FloPyAgentDQN_noExceptionRaised(self):
        """Test an agent using the deep Q-Learning algorithm on a
        computationally simple case.
        """

        raised = False
        try:
            from FloPyArcadeDQN import envSettings, hyParams
            # environment settings
            envSettings['ENVTYPE'] = '2'
            envSettings['MODELNAME'] = 'unittestDQN'
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
            FloPyArcadeDQN.main(envSettings, hyParams)
            print('Successfully tested deep Q-network agent.')
        except Exception as e:
            raised = True
            print('error FloPyArcadeDQN:', e)
        self.assertFalse(raised, 'Deep Q-network agent test failed.')


class TestFloPyAgentGenetic(unittest.TestCase):

    def test_FloPyAgentGenetic_noExceptionRaised(self):
        """Test the genetic algorithm on a computationally simple case."""

        raised = False
        try:
            from FloPyArcadeGeneticNetwork import envSettings, hyParams
            # environment settings
            envSettings['ENVTYPE'] = '1'
            envSettings['MODELNAME'] = 'unittestGenetic'
            envSettings['SURROGATESIMULATOR'] = None
            envSettings['NAGENTSPARALLEL'] = 2
            envSettings['RENDER'] = False
            envSettings['BESTAGENTANIMATION'] = True
            envSettings['KEEPMODELHISTORY'] = True
            envSettings['RESUME'] = False
            # hyperparameters
            hyParams['NAGENTS'] = 4
            hyParams['NAGENTELITES'] = 2
            hyParams['NGENERATIONS'] = 2
            hyParams['NGAMESAVERAGED'] = 2
            hyParams['NAGENTSTEPS'] = 2
            hyParams['NHIDDENNODES'] = [5] * 2
            hyParams['HIDDENACTIVATIONS'] = ['relu'] * 2
            hyParams['NOVELTYSEARCH'] = True
            hyParams['ADDNOVELTYEVERY'] = 1
            hyParams['NNOVELTYELITES'] = 1
            FloPyArcadeGeneticNetwork.main(envSettings, hyParams)
            print('Successfully tested genetic agent.')
        except Exception as e:
            raised = True
            print('error FloPyArcadeGeneticNetwork', e)
        self.assertFalse(raised, 'Genetic agent test failed')


if __name__ == '__main__':
    unittest.main(exit=False)

    # deleting all test-related files, except unittestAgent.h5
    wrkspc = dirname(abspath(__file__))
    unittestFilesAndFolders = glob(join(wrkspc, 'models', 'unittest*'))
    for obj in unittestFilesAndFolders:
        if 'unittestAgent1.model' not in obj:
            if 'unittestAgent2.model' not in obj:
                if 'unittestAgent3.model' not in obj:
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