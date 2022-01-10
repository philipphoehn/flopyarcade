#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from flopyarcade import play
from flopyarcade import train_dqn
from flopyarcade import train_neuroevolution
# import FloPyArcadePlay
# import FloPyArcadeDQN
# import FloPyArcadeGeneticNetwork
from flopyarcade import FloPyArcade
from glob import glob
from os import remove
from os.path import abspath, dirname, isdir, isfile, join
from shutil import rmtree
import unittest
# unittest.main(warnings="ignore")


class TestFloPyEnvPlay(unittest.TestCase):

    def test_FloPyEnvPlay_noExceptionRaised(self):
        """Test a game run of every existing environment."""

        # from FloPyArcadePlay import envSettings, gameSettings
        envSettings, gameSettings = play.envSettings, play.gameSettings

        failed = []
        for ENVTYPE in FloPyArcade().ENVTYPES:
            raised = False
            try:
                # environment settings
                envSettings['ENVTYPE'] = ENVTYPE
                envSettings['MODELNAMELOAD'] = None
                envSettings['MODELNAME'] = 'unittest'
                envSettings['SAVEPLOT'] = False
                envSettings['SAVEPLOTALLAGENTS'] = False
                envSettings['MANUALCONTROL'] = False
                envSettings['RENDER'] = False
                # game settings
                gameSettings['NGAMES'] = 1
                gameSettings['NAGENTSTEPS'] = 2
                play.main(envSettings, gameSettings)
                # FloPyArcadePlay.main(envSettings, gameSettings)
                print('Test of environment ' + ENVTYPE + ' succeeded.')
            except Exception as e:
                raised = True
                failed.append(ENVTYPE)
                print('error FloPyArcadePlay', e)
        for ENVTYPE in failed:
            self.assertFalse(raised, 'Test of environment ' + ENVTYPE + ' failed.')


class TestFloPyAgentDQN(unittest.TestCase):

    def test_FloPyAgentDQN_noExceptionRaised(self):
        """Test an agent using the deep Q-Learning algorithm on a
        computationally simple case.
        """

        # from train_dqn import envSettings, hyParams
        envSettings, hyParams = train_dqn.envSettings, train_dqn.hyParams

        failed = []
        for ENVTYPE in FloPyArcade().ENVTYPES:
            raised = False
            try:
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
            except Exception as e:
                raised = True
                failed.append(ENVTYPE)
                print('error train_dqn:', e)
        for ENVTYPE in failed:
            self.assertFalse(raised, 'Test of deep Q-network agent in environment ' + ENVTYPE + ' failed.')


class TestFloPyAgentGenetic(unittest.TestCase):

    def test_FloPyAgentGenetic_noExceptionRaised(self):
        """Test the genetic algorithm on a computationally simple case."""

        envSettings, hyParams = train_neuroevolution.envSettings, train_neuroevolution.hyParams
        # from train_genetic import envSettings, hyParams
        from matplotlib import use as matplotlibBackend

        failed = []
        for KEEPMODELHISTORY in [True, False]:
            for NNTYPE in ['convolution', 'perceptron']:
                # for ENVTYPE in FloPyArcade().ENVTYPES:
                for ENVTYPE in ['1s-d', '2r-d', '3s-c', '4r-c', '5r-c', '6r-c']:
                    raised = False
                    try:
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
                    except Exception as e:
                        raised = True
                        failed.append(ENVTYPE)
                        print('error train_neuroevolution', e)
        for ENVTYPE in failed:
            self.assertFalse(raised, 'Genetic agent in environment ' + ENVTYPE + ' failed.')


if __name__ == '__main__':
    unittest.main(exit=False)

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
