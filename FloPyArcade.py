#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


# imports for environments
from flopy.modflow import Modflow, ModflowBas, ModflowDis, ModflowLpf
from flopy.modflow import ModflowOc, ModflowPcg, ModflowWel
from flopy.modpath import Modpath, ModpathBas
from flopy.plot import PlotMapView
from flopy.utils import CellBudgetFile, HeadFile, PathlineFile
from imageio import get_writer, imread
from matplotlib.cm import get_cmap
from matplotlib.pyplot import Circle, close, figure, pause, show
from matplotlib.pyplot import waitforbuttonpress
from numpy import add, argmax, argsort, array, ceil, copy, divide, extract
from numpy import float32, int32, linspace, max, maximum, min, minimum, mean
from numpy import ones, shape, sqrt, sum, unique, zeros
from numpy.random import randint, random, randn, seed, uniform
from os import environ, listdir, makedirs, remove, rmdir
from os.path import abspath, dirname, exists, join
from platform import system
from sys import modules
from time import sleep, time

# additional imports for agents
from collections import deque
from datetime import datetime
from gc import collect as garbageCollect
from pathos.pools import _ProcessPool as Pool
from pickle import dump, load
from keras.initializers import glorot_uniform
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from keras.models import clone_model, load_model, save_model, Sequential
from keras.optimizers import Adam
from random import sample as randomSample, seed as randomSeed
from tensorflow.compat.v1 import ConfigProto, set_random_seed
from tensorflow.compat.v1 import Session as TFSession
from tensorflow.compat.v1.keras import backend as K
from tqdm import tqdm
from uuid import uuid4

# potentially new imports of Keras within TensorFlow 2.0, but yet to debug
# from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.layers import Activation, BatchNormalization, Dense
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.models import clone_model, load_model, save_model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam

# suppressing tensorflow backend usage message
# import logging
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)

# ignoring AVX2 support
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_folder_structure(wrkspc):
    modelspth = join(wrkspc, 'models')
    if not exists(modelspth):
        makedirs(modelspth)
    runspth = join(wrkspc, 'runs')
    if not exists(runspth):
        makedirs(runspth)
    simulatorspth = join(wrkspc, 'simulators')
    if not exists(simulatorspth):
        makedirs(simulatorspth)
    temppth = join(wrkspc, 'temp')
    if not exists(temppth):
        makedirs(temppth)


class FloPyAgent():
    """Agent to navigate a spawned particle advectively through one of the
    aquifer environments, collecting reward along the way.
    """

    def __init__(self, observationsVector=None, actionSpace=['keep'],
                 hyParams=None, envSettings=None, mode='random'
                 ):
        """Constructor"""

        self.wrkspc = dirname(abspath(__file__))
        if 'library.zip' in self.wrkspc:
            # changing workspace in case of call from compiled executable
            self.wrkspc = dirname(dirname(self.wrkspc))

        create_folder_structure(self.wrkspc)

        self.actionSpace = actionSpace
        self.actionSpaceSize = len(self.actionSpace)

        self.observationsVector = observationsVector
        self.hyParams = hyParams
        self.envSettings = envSettings
        self.agentMode = mode
        if self.envSettings is not None:
            self.SEED = self.envSettings['SEED']
            seed(self.SEED)
            randomSeed(self.SEED)
            set_random_seed(self.SEED)

        if self.agentMode == 'DQN':
            self.initializeDQNAgent()
        if self.agentMode == 'genetic':
            self.initializeGeneticAgents()

    def initializeDQNAgent(self):
        """Initialize agent to perform Deep Double Q-Learning.

        Bug fix for predict function:
        https://github.com/keras-team/keras/issues/6462
        """

        # initializing main predictive and target model
        self.mainModel = self.createNNModel()
        self.mainModel._make_predict_function()
        self.targetModel = self.createNNModel()
        self.targetModel._make_predict_function()
        self.targetModel.set_weights(self.mainModel.get_weights())

        # initializing array with last training data of specified length
        self.replayMemory = deque(maxlen=self.hyParams['REPLAYMEMORYSIZE'])
        self.epsilon = self.hyParams['EPSILONINITIAL']

        # initializing counter for updates on target network
        self.targetUpdateCounter = 0

    def initializeGeneticAgents(self):
        """Initialize genetic ensemble of agents."""

        # calling the initialization in a child process to avoid drastic memory
        # increase of parent process that might hamper later parallelisation
        with Pool(1) as executor:
            executor.map(
                self.randomAgentsGenetic, [self.hyParams['NAGENTS']])

    # def runGame(self, env, current_state, actionsVal):
    #     """Run a single instance of a game."""

    #     # iterating until game ends
    #     for timeStep in range(self.hyParams['NAGENTSTEPS']):

    #         # querying training model for q values and corresponding action
    #         actionIdx = argmax(self.getqsGivenAgentModel(
    #             self.mainModel, env.observationsVectorNormalized))
    #         action = self.actionSpace[actionIdx]
    #         actionsVal.append(action)

    #         # running current timestep of the game
    #         new_state, reward, done, info = env.step(
    #             env.observationsVectorNormalized, action, self.gameReward)
    #         # adding timestep reward
    #         self.gameReward += reward

    #         # rendering if required
    #         if self.envSettings['RENDER']:
    #             if not gameVal % self.envSettings['RENDEREVERY']:
    #                 if not done:
    #                     env.render()

    #         current_state = new_state
    #         step += 1
    #         if done:
    #             break

    def runDQN(self, env):
        """
        # generating seeds to generate recurring cross-validation dataset with
        # simplifies judgement of training progress as random noise will be reduced
        # and results will be reproducible

        # It will be important to cross-validate on the same set
        # of models here to see if consistently learning or not
        # or will this lead to overfitting?
        # therefore also no training during cross-validation
        # because this way the reward on training data is reported
        # where is the step to train the target network?
        """

        seed(self.envSettings['SEEDCROSSVALIDATION'])
        seedsCV = randint(self.envSettings['SEEDCROSSVALIDATION'],
                          size=self.hyParams['NGAMESCROSSVALIDATED']
                          )
        seed(self.envSettings['SEED'])

        gameRewards = []
        # iterating over games being played
        for game in tqdm(
                range(
                    1,
                    self.hyParams['NGAMES'] +
                    1),
                ascii=True,
                unit='games'):

            # resetting environment
            env.reset()

            # updating replay memory and training main network
            self.takeActionsUpdateAndTrain(env)

            if env.success:
                self.gameReward = self.gameReward
            elif env.success == False:
                # overwriting rewards in memory to zero in case of lack of success
                # is it not better to give reward an not reset all rewards to 0?
                # overwrites entire memory from game
                self.gameReward = 0.0

                # is this better on or off?
                self.updateReplayMemoryZeroReward(self.gameStep)

            # cross validation
            # append game reward to a list and log stats (every given number
            # of games)
            gameRewards.append(self.gameReward)
            if not game % self.hyParams['CROSSVALIDATEEVERY'] or game == 1:

                # loop to cross-validate on unique set of models
                gameRewardsCrossValidation = []
                actionsVal = []
                for gameVal in range(self.hyParams['NGAMESCROSSVALIDATED']):

                    # resetting environment and variables
                    self.gameReward, step, done = 0.0, 0, False
                    seedCV = seedsCV[gameVal]
                    env.reset(_seed=seedCV)
                    current_state = env.observationsVectorNormalized
                    # iterating until game ends
                    for _ in range(self.hyParams['NAGENTSTEPS']):

                        # This part stays mostly the same, the change is to
                        # query a model for Q values
                        actionIdx = argmax(self.getqsGivenAgentModel(
                            self.mainModel, env.observationsVectorNormalized))
                        action = self.actionSpace[actionIdx]
                        actionsVal.append(action)

                        new_state, reward, done, info = env.step(
                            env.observationsVectorNormalized, action, self.gameReward)

                        # Transform new continous state to new discrete state
                        # and count reward
                        self.gameReward += reward

                        if self.envSettings['RENDER']:
                            if not gameVal % self.envSettings['RENDEREVERY']:
                                if not done:
                                    env.render()

                        current_state = new_state
                        step += 1

                        if done:
                            break

                    if env.success:
                        self.gameReward = self.gameReward
                    elif env.success == False:
                        self.gameReward = 0.0

                    gameRewardsCrossValidation.append(self.gameReward)

                average_reward = mean(gameRewardsCrossValidation)
                min_reward = min(gameRewardsCrossValidation)
                max_reward = max(gameRewardsCrossValidation)

                # print('debug gameRewardsCrossValidation', gameRewardsCrossValidation)
                # average_reward = mean(gameRewardsCrossValidation[-self.hyParams['CROSSVALIDATEEVERY']:])
                # min_reward = min(gameRewardsCrossValidation[-self.hyParams['CROSSVALIDATEEVERY']:])
                # max_reward = max(gameRewardsCrossValidation[-self.hyParams['CROSSVALIDATEEVERY']:])

                # print('game', game, 'average reward', average_reward, 'min reward', min_reward, 'max reward', max_reward, 'epsilon', self.epsilon)
                # print('debug uniques', unique(actionsVal))

                # saving model if larger than a specified reward threshold
                if average_reward >= self.envSettings['REWARDMINTOSAVE']:
                    MODELNAME = self.envSettings['MODELNAME']
                    self.mainModel.save(
                        join(
                            self.wrkspc,
                            'models',
                            f'{MODELNAME}{game:_>7.0f}ep{max_reward:_>7.1f}max{average_reward:_>7.1f}avg{min_reward:_>7.1f}min{datetime.now().strftime("%Y%m%d%H%M%S")}datetime' +
                            '.model'))

            # decaying epsilon
            if self.epsilon > self.hyParams['EPSILONMIN']:
                self.epsilon *= self.hyParams['EPSILONDECAY']
                self.epsilon = max([self.hyParams['EPSILONMIN'], self.epsilon])

    def runGenetic(self, env):
        """Run main pipeline for genetic agent optimisation.

        # Inspiration and larger parts of code modified after and inspired by:
        # https://github.com/paraschopra/deepneuroevolution
        # https://arxiv.org/abs/1712.06567
        """

        # setting environment and number of games
        self.env = env
        n = self.hyParams['NGAMESAVERAGED']

        # generating unique process ID from system time
        self.pid = datetime.now().strftime('%Y%m%d%H%M%S-') + str(uuid4())

        agentCounts = []
        for iAgent in range(self.hyParams['NAGENTS']):
            agentCounts.append(iAgent)

        for generation in range(self.hyParams['NGENERATIONS']):
            self.geneticGeneration = generation

            # simulating agents in environment, returning average of n runs
            self.rewards = self.runAgentsRepeatedlyGenetic(agentCounts, n, env)

            # sorting by rewards in reverse, starting with indices of top reward
            # https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
            # sorting rewards to isolate best agents
            sorted_parent_indexes = argsort(
                self.rewards)[::-1][:self.hyParams['NAGENTELITES']]

            # setting up an empty list as a children agents container
            # returning best-performing agents
            filehandler = open(
                join(
                    self.wrkspc,
                    'temp',
                    self.envSettings['MODELNAME'] +
                    '_gen' +
                    str(
                        self.geneticGeneration +
                        1).zfill(3) +
                    '_agentsSortedParentIndexes.p'),
                'wb')
            dump(sorted_parent_indexes, filehandler)
            filehandler.close()

            # calling in a child process avoids memory rising in parent process
            with Pool(1) as executor:
                executor.map(self.returnChildrenGenetic, 
                    [sorted_parent_indexes])

            bestAgent = load_model(join(
                    self.wrkspc,
                    'temp',
                    self.envSettings['MODELNAME'] +
                    '_gen' +
                    str(
                        self.geneticGeneration +
                        1).zfill(3) +
                    '_agentBest.model'))

            self.bestAgentReward = self.rewards[sorted_parent_indexes[0]]
            # saving best agent model
            MODELNAME = self.envSettings['MODELNAME']
            bestAgentFileName = f'{MODELNAME}' + '_gen' + str(generation+1).zfill(3) + f'_avg{self.bestAgentReward:_>7.1f}' + '_' + datetime.now().strftime('%Y%m%d%H%M%S')
            bestAgent.save(join(self.wrkspc, 'models', bestAgentFileName + '.model'))

            if self.envSettings['BESTAGENTANIMATION']:
                # playing a game with best agent to visualize progress
                game = FloPyArcade(
                    modelNameLoad=bestAgentFileName,
                    modelName=bestAgentFileName,
                    NAGENTSTEPS=self.hyParams['NAGENTSTEPS'],
                    PATHMF2005=self.envSettings['PATHMF2005'],
                    PATHMP6=self.envSettings['PATHMP6'],
                    flagSavePlot=True,
                    flagManualControl=False,
                    flagRender=False)
                game.play(
                    ENVTYPE=self.envSettings['ENVTYPE'],
                    seed=self.envSettings['SEED'] + self.currentGame)

            if not self.envSettings['KEEPMODELHISTORY']:
                # removing stored agent models of finished generation
                for agentIdx in range(self.hyParams['NAGENTS']):
                    remove(join(self.wrkspc, 'temp', self.envSettings['MODELNAME'] + '_gen' + str(
                        self.geneticGeneration + 1).zfill(3) + '_agent' + str(agentIdx + 1).zfill(4) + '.model'))

            del bestAgent

    def createNNModel(self, seed=None):
        """Create fully-connected feed-forward multi-layer neural network."""
        if seed is None:
            seed = self.SEED
        model = Sequential()
        initializer = glorot_uniform(seed=seed)

        nHiddenNodes = copy(self.hyParams['NHIDDENNODES'])

        if self.agentMode == 'genetic' and self.hyParams['ARCHITECTUREVARY']:
            for layerIdx in range(len(nHiddenNodes)):
                # if self.mutateDecision(self.hyParams['MUTATIONPROBABILITY']):
                nHiddenNodes[layerIdx] = randint(2, self.hyParams['NHIDDENNODES'][layerIdx]+1)
        # print('debug architecture', nHiddenNodes)

        # adding hidden layers
        for layerIdx in range(len(nHiddenNodes)):
            model.add(
                Dense(units=nHiddenNodes[layerIdx],
                    input_shape=shape(self.observationsVector) if layerIdx == 0 else [],
                    # input_shape=shape(self.observationsVector) if layerIdx == 0 else [],
                    kernel_initializer=initializer,
                    use_bias=True
                    )
                )
            if self.hyParams['BATCHNORMALIZATION']:
                model.add(BatchNormalization())
            model.add(Activation(self.hyParams['HIDDENACTIVATIONS'][layerIdx]))
            if 'DROPOUTS' in self.hyParams:
                if self.hyParams['DROPOUTS'][layerIdx] != 0.0:
                    model.add(Dropout(self.hyParams['DROPOUTS'][layerIdx]))

        # adding output layer
        # changed from activation='linear' for regression to 'sigmoid' for classification
        # but qs ate regression
        model.add(Dense(self.actionSpaceSize, activation='sigmoid',
                        kernel_initializer=initializer
                        )
                  )

        # compiling to avoid warning while saving agents in genetic search
        model.compile(loss="mse",
                      optimizer=Adam(lr=self.hyParams['LEARNINGRATE'] if 'LEARNINGRATE' in self.hyParams else 0.0001),
                      metrics=['accuracy']
                      )

        return model

    def updateReplayMemory(self, transition):
        """Update replay memory by adding a given step's data to a memory
        replay array.
        """
        self.replayMemory.append(transition)

    def updateReplayMemoryZeroReward(self, steps):
        """Update replay memory rewards to zero in case game ended up with zero
        reward.
        """
        for i in range(steps):
            self.replayMemory[-i][2] = 0.0

    def train(self, terminal_state, step):
        """Trains main network every step during a game."""

        # training only if certain number of samples is already saved
        if len(self.replayMemory) < self.hyParams['REPLAYMEMORYSIZEMIN']:
            return

        # retrieving a subset of random samples from memory replay table
        minibatch = randomSample(self.replayMemory,
                                 self.hyParams['MINIBATCHSIZE']
                                 )

        # retrieving current states from minibatch
        # then querying NN model for Q values
        current_states = array([transition[0] for transition in minibatch])
        current_qs_list = self.mainModel.predict(current_states)

        # retrieving future states from minibatch
        # then querying NN model for Q values
        # when using target network, query it, otherwise main network should be
        # queried
        new_current_states = array([transition[3] for transition in minibatch])
        future_qs_list = self.targetModel.predict(new_current_states)

        X, y = [], []

        # enumerating batches
        for index, (current_state, action, reward, new_current_state,
                    done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states,
            # otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation
            if not done:
                max_future_q = max(future_qs_list[index])
                new_q = reward + self.hyParams['DISCOUNT'] * max_future_q
            else:
                new_q = reward

            # updating Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # appending states to training data
            X.append(current_state)
            y.append(current_qs)

        # fitting on all samples as one batch, logging only on terminal state
        self.mainModel.fit(array(X), array(y),
                           batch_size=self.hyParams['MINIBATCHSIZE'],
                           verbose=0, shuffle=False
                           )

        # updating target network counter every game
        if terminal_state:
            self.targetUpdateCounter += 1

        # updating target network with weights of main network,
        # if counter reaches set value
        if self.targetUpdateCounter > self.hyParams['UPDATEPREDICTIVEMODELEVERY']:
            self.targetModel.set_weights(self.mainModel.get_weights())
            self.targetUpdateCounter = 0

    def takeActionsUpdateAndTrain(self, env):
        """Take an action and update as well as train the main network every
        step during a game. Acts as the main operating body of the runDQN
        function.
        """

        # retrieving initial state
        current_state = env.observationsVectorNormalized

        # resetting counters prior to restarting game
        self.gameReward, self.gameStep, done = 0, 1, False
        for game in range(self.hyParams['NAGENTSTEPS']):
            # epsilon defines the fraction of random to queried actions
            if random() > self.epsilon:
                # retrieving action from Q table
                actionIdx = argmax(self.getqsGivenAgentModel(
                    self.mainModel, env.observationsVectorNormalized))
            else:
                # retrieving random action
                actionIdx = randint(0, self.actionSpaceSize)
            action = self.actionSpace[actionIdx]

            new_state, reward, done, info = env.step(
                env.observationsVectorNormalized, action, self.gameReward)
            new_state = env.observationsVectorNormalized

            # updating replay memory
            self.updateReplayMemory(
                [current_state, actionIdx, reward, new_state, done])
            # training main network on every step
            self.train(done, self.gameStep)

            # counting reward
            self.gameReward += reward

            if self.envSettings['RENDER']:
                if not done:
                    if not game+1 % self.envSettings['RENDEREVERY']:
                        env.render()

            # transforming new continous state to new discrete state
            current_state = env.observationsVectorNormalized
            self.gameStep += 1

            if done:
                break

    def runAgentsGenetic(self, agentCounts, env):
        """Run genetic agent optimisation, if opted for with multiprocessing.
        """

        # running in parallel if specified
        # debug: add if available number of CPU cores is exceeded
        cores = self.envSettings['NAGENTSPARALLEL']

        # why was this necessary?
        if __name__ == 'FloPyArcade':

            chunksTotal = self.yieldChunks(agentCounts, cores)
            # batching parallel processes
            reward_agents = []
            # batch processing to avoid memory explosion
            # https://stackoverflow.com/questions/18414020/memory-usage-keep-growing-with-pythons-multiprocessing-pool/20439272
            for chunk in chunksTotal:
                # Pool object from pathos instead of multiprocessing library necessary
                # as tensor.keras models are currently not pickleable
                # https://github.com/tensorflow/tensorflow/issues/32159
                p = Pool(processes=len(chunk), maxtasksperchild=1)
                reward_chunks = p.map(self.runAgentsGeneticSingleRun, chunk)
                p.close()
                p.join()
                p.terminate()
                reward_agents += reward_chunks

        return reward_agents

    def runAgentsGeneticSingleRun(self, agentCount):
        """Run single game within genetic agent optimisation.
        """

        # load specific agent and weights with given ID
        agent = load_model(join(self.wrkspc, 'temp', self.envSettings['MODELNAME'] + '_gen' + str(
            self.geneticGeneration + 1).zfill(3) + '_agent' + str(agentCount + 1).zfill(4) + '.model'))
        # agent.load_weights(join(self.wrkspc, 'temp', self.envSettings['MODELNAME'] + '_gen' + str(
        #     self.geneticGeneration + 1).zfill(3) + '_agent' + str(agentCount + 1).zfill(4) + 'Weights.h5'))

        # inspecting parameters on the fly, if they stay consistent
        # parameters = agent.get_weights()
        # for iParameters, layer in enumerate(parameters):
        # 	if iParameters == 2:
        # 		print('debug weight 100-0', layer[100][0], 'agentCount', agentCount)

        MODELNAMEUNIQUE = self.envSettings['MODELNAME'] + 'Temp' + self.pid + '_' + str(agentCount + 1)

        env = self.env
        # resetting with temporary name to enable parallelism
        env.reset(
            MODELNAME=self.envSettings['MODELNAME'] +
            'Temp' +
            self.pid +
            '_' +
            str(
                agentCount +
                1),
            _seed=self.envSettings['SEED'] +
            self.currentGame)

        r = 0
        actionsDebug = []
        for _ in range(self.hyParams['NAGENTSTEPS']):

            actionIdx = argmax(
                self.getqsGivenAgentModel(
                    agent, env.observationsVectorNormalized))

            action = self.actionSpace[actionIdx]
            actionsDebug.append(action)
            # needs to feed observationsVectorNormalized as observations are
            # otherwise not normalized
            new_observation, reward, done, info = env.step(
                env.observationsVectorNormalized, action, r)
            r += reward

            if self.envSettings['RENDER']:
                env.render()

            if done:
                if env.success == False:
                    r = 0
                # print('debug north south', env.headSpecNorth, env.headSpecSouth)
                # print('debug reward', r, 'debug wellQ', env.wellQ, 'debug agentCount', agentCount, 'debug actions', unique(actionsDebug))
                del env, agent, agentCount, actionIdx, action,
                del new_observation, done, info
                garbageCollect()
                break

        return r


    def runAgentsRepeatedlyGenetic(self, agentCounts, n, env):
        """Run all agents within genetic optimisation for a defined number of
        games.
        """

        reward_agentsMin = zeros(len(agentCounts))
        reward_agentsMax = zeros(len(agentCounts))
        reward_agentsMean = zeros(len(agentCounts))
        for game in range(n):
            self.currentGame = game + 1
            print('Currently: ' +
                  str(game +
                      1) +
                  '/' +
                  str(n) +
                  ' games, ' +
                  str(self.geneticGeneration +
                      1) +
                  '/' +
                  str(self.hyParams['NGENERATIONS']) +
                  ' generations')
            rewardsAgentsCurrent = self.runAgentsGenetic(agentCounts, env)
            reward_agentsMin = minimum(reward_agentsMin, rewardsAgentsCurrent)
            reward_agentsMax = maximum(reward_agentsMax, rewardsAgentsCurrent)
            reward_agentsMean = add(reward_agentsMean, rewardsAgentsCurrent)
        reward_agentsMean = divide(reward_agentsMean, n)

        filehandler = open(
            join(
                self.wrkspc,
                'temp',
                self.envSettings['MODELNAME'] +
                '_gen' +
                str(
                    self.geneticGeneration +
                    1).zfill(3) +
                '_agentsRewardsMin.p'),
            'wb')
        dump(reward_agentsMin, filehandler)
        filehandler.close()
        filehandler = open(
            join(
                self.wrkspc,
                'temp',
                self.envSettings['MODELNAME'] +
                '_gen' +
                str(
                    self.geneticGeneration +
                    1).zfill(3) +
                '_agentsRewardsMax.p'),
            'wb')
        dump(reward_agentsMax, filehandler)
        filehandler.close()
        filehandler = open(
            join(
                self.wrkspc,
                'temp',
                self.envSettings['MODELNAME'] +
                '_gen' +
                str(
                    self.geneticGeneration +
                    1).zfill(3) +
                '_agentsRewardsMean.p'),
            'wb')
        dump(reward_agentsMean, filehandler)
        filehandler.close()

        return reward_agentsMean

    def randomAgentsGenetic(self, num_agents):
        """Create defined number of agents for genetic optimisation and save
        them to disk individually.
        """

        self.geneticGeneration = 0
        for agentIdx in range(num_agents):
            agent = self.createNNModel(seed=agentIdx)
            agent.save(join(self.wrkspc, 'temp', self.envSettings['MODELNAME'] + '_gen' + str(
                self.geneticGeneration + 1).zfill(3) + '_agent' + str(agentIdx + 1).zfill(4) + '.model'))
            del agent

        garbageCollect()

    def returnChildrenGenetic(self, sorted_parent_indexes):
        """Mutate best parents, keep elite child and save them to disk
        individually.
        """

        filehandler = open(
            join(
                self.wrkspc,
                'temp',
                self.envSettings['MODELNAME'] +
                '_gen' +
                str(
                    self.geneticGeneration +
                    1).zfill(3) +
                '_agentsSortedParentIndexes.p'),
            'rb')
        sorted_parent_indexes = load(filehandler)
        filehandler.close()

        bestAgent = load_model(join(
                self.wrkspc,
                'temp',
                self.envSettings['MODELNAME'] + '_gen' + str(
                    self.geneticGeneration + 1).zfill(3) + '_agent' + str(
                    sorted_parent_indexes[0] + 1).zfill(4) + '.model'))

        candidate_parent_indices = sorted_parent_indexes[:
                                                         self.hyParams['NAGENTELITES']]
        # first take selected parents from sorted_parent_indexes and generate
        # N-1 children
        for i in range(self.hyParams['NAGENTS'] - 1):
            # what is this doing?!
            # why not generate from selected elite parents? --> changed to to
            # that --> does it work?
            selected_agent_index = candidate_parent_indices[randint(
                len(candidate_parent_indices))]

            # loading this agent and saving without collecting
            agent = load_model(join(
                    self.wrkspc,
                    'temp',
                    self.envSettings['MODELNAME'] + '_gen' + str(
                        self.geneticGeneration + 1).zfill(3) + '_agent' + str(
                        selected_agent_index + 1).zfill(4) + '.model'))

            childrenAgent = self.mutateGenetic(agent)

            if self.geneticGeneration + 1 < self.hyParams['NGENERATIONS']:
                childrenAgent.save(join(self.wrkspc, 'temp', self.envSettings['MODELNAME'] + '_gen' + str(
                    self.geneticGeneration + 2).zfill(3) + '_agent' + str(i + 1).zfill(4) + '.model'))

        # adding single-best elite child
        elite_child = self.addEliteGenetic(sorted_parent_indexes)

        if self.geneticGeneration + 1 < self.hyParams['NGENERATIONS']:
            elite_child.save(join(
                    self.wrkspc,
                    'temp',
                    self.envSettings['MODELNAME'] + '_gen' + str(
                        self.geneticGeneration + 2).zfill(3) + '_agent' + str(
                        self.hyParams['NAGENTS']).zfill(4) + '.model'))

        bestAgent.save(join(
                self.wrkspc,
                'temp',
                self.envSettings['MODELNAME'] +
                '_gen' +
                str(
                    self.geneticGeneration +
                    1).zfill(3) +
                '_agentBest.model'))

        del bestAgent, filehandler

    def mutateGenetic(self, agent):
        """Mutate single agent model.

        Mutation power is hyperparameter and 
        https://arxiv.org/pdf/1712.06567.pdf has example values.
        """

        weights = agent.get_weights()
        paramIdx = 0
        for parameters in weights:
            if self.mutateDecision(self.hyParams['MUTATIONPROBABILITY']):
                weights[paramIdx] = add(parameters,
                                        self.hyParams['MUTATIONPOWER'] * randn())
            paramIdx += 1
        agent.set_weights(weights)

        return agent

    def mutateDecision(self, probability):
        """Return boolean defining whether to mutate or not."""
        return random() < probability


    def addEliteGenetic(self, sorted_parent_indexes, NAGENTELITES=10):
        """Select and store best agent to disk."""

        # is rerunning necessary here if scores were imported?
        # this is only helpful if running a longer cross-validation to check
        # for changes in ranking

        top_score, top_elite_index, i = None, None, 0
        for score in self.rewards:
            if (top_score is None) or (score > top_score):
                top_score = score
                top_elite_index = i
            i += 1

        # this goes wrong, if rewards list has different length than candidateEliteIndices
        # with long run, but not if NAGENTELITES is larger than number of
        # agents

        print('Elite selected with index ', top_elite_index, ' and score',
            top_score)

        agentBest = load_model(join(
                self.wrkspc,
                'temp',
                self.envSettings['MODELNAME'] + '_gen' + str(
                    self.geneticGeneration + 1).zfill(3) + '_agent' + str(
                    top_elite_index + 1).zfill(4) + '.model'))

        return agentBest

    def getqsGivenAgentModel(self, agentModel, state):
        """ Query given model for Q values given observations of state
        """
        # predict_on_batch robust in parallel operation?
        return agentModel.predict_on_batch(
            array(state).reshape(-1, (*shape(state))))[0]

    def getAction(self, mode='random', keyPressed=None, agent=None,
            modelNameLoad=None, state=None):
        """Retrieve a choice of action.

        Either the action is determined from the player pressing a button or
        chosen randomly. If the player does not press a key within the given
        timeframe for a game, the action remains unchanged.
        """

        if mode == 'manual':
            if keyPressed in self.actionSpace:
                self.action = keyPressed
            else:
                self.action = 'keep'
        if mode == 'random':
            actionIdx = randint(0, high=len(self.actionSpace))
            self.action = self.actionSpace[actionIdx]
        if mode == 'modelNameLoad':
            # loading model and parameters
            agentModel = load_model(
                join(self.wrkspc, 'models', modelNameLoad + '.model'))
            # agentModel.load_weights(join(self.wrkspc, 'models', modelNameLoad + 'Weights.h5'))
            # predicting with given model
            actionIdx = argmax(
                self.getqsGivenAgentModel(
                    agentModel, state))
            self.action = self.actionSpace[actionIdx]
        if mode == 'model':
            # predicting with given model
            actionIdx = argmax(
                self.getqsGivenAgentModel(
                    agent, state))
            self.action = self.actionSpace[actionIdx]

        return self.action

    def yieldChunks(self, lst, n):
        """Yield successive n-sized chunks from a given list.
        Taken from: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def pickleLoad(self, path):
        """Load pickled object from file."""
        filehandler = open(path, 'rb')
        objectLoaded = load(filehandler)
        filehandler.close()
        return objectLoaded

    def pickleDump(self, path, objectToDump):
        """Store object to file using pickle."""
        filehandler = open(path, 'wb')
        dump(objectToDump, filehandler)
        filehandler.close()

    def GPUAllowMemoryGrowth(self):
        """Allow GPU memory to grow to enable parallelism on a GPU."""
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        sess = TFSession(config=config)
        K.set_session(sess)

class FloPyEnv():
    """Environment to perform forward simulation using MODFLOW and MODPATH.

    On first call initializes a model with a randomly-placed operating well,
    initializes the corresponding steady-state flow solution as a starting state
    and initializes a random starting action and a random particle on the
    Western side.

    On a calling a step, it loads the current state, tracks the particle's
    trajectory through the model domain and returns the environment's new state,
    the new particle location as an observation and a flag if the particle has
    reached the operating well or not as a state.
    """

    def __init__(
            self,
            ENVTYPE='1',
            PATHMF2005=None,
            PATHMP6=None,
            MODELNAME='FloPyArcade',
            _seed=None,
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            NAGENTSTEPS=None):
        """Constructor."""

        self.ENVTYPE = ENVTYPE
        self.PATHMF2005, self.PATHMP6 = PATHMF2005, PATHMP6
        self.MODELNAME = MODELNAME
        self.SAVEPLOT = flagSavePlot
        self.MANUALCONTROL = flagManualControl
        self.MANUALCONTROLTIME = 0.1
        self.RENDER = flagRender
        self.NAGENTSTEPS = NAGENTSTEPS
        self.info = ''
        self.comments = ''
        self.done = False

        self.wrkspc = dirname(abspath(__file__))
        if 'library.zip' in self.wrkspc:
            # changing workspace in case of call from executable
            self.wrkspc = dirname(dirname(self.wrkspc))

        # setting up the model paths and ensuring they exists
        create_folder_structure(self.wrkspc)
        self.modelpth = join(self.wrkspc, 'models', MODELNAME)
        if not exists(self.modelpth):
            makedirs(self.modelpth)

        # general environment settings,
        # like model domain and grid definition
        # uses SI units for length and time
        # currently fails with arbitray model extents?
        # why is a periodLength of 2.0 necessary to simulate 1 day?
        self.minX, self.minY = 0., 0.
        self.extentX, self.extentY = 100., 100.
        self.zBot, self.zTop = 0, 50.
        self.nLay, self.nRow, self.nCol = 1, 100, 100
        self.headSpecWest, self.headSpecEast = 10.0, 6.0
        self.minQ = -3000.0
        self.maxQ = -500.0
        self.wellSpawnBufferXWest, self.wellSpawnBufferXEast = 50.0, 20.0
        self.wellSpawnBufferY = 20.0
        self.periods, self.periodLength, self.periodSteps = 1, 2.0, 5
        self.periodSteadiness = True
        self.maxSteps = 300
        self.sampleHeadsEvery = 10

        self.dRow = self.extentX / self.nCol
        self.dCol = self.extentY / self.nRow
        self.dVer = (self.zTop - self.zBot) / self.nLay
        self.botM = linspace(self.zTop, self.zBot, self.nLay + 1)

        self.wellRadius = sqrt((2 * self.dCol)**2 + (2 * self.dRow)**2)

        if self.ENVTYPE == '1':
            self.minH = 6.0
            self.maxH = 10.0
            self.actionSpace = ['up', 'keep', 'down']
            self.actionRange = 0.5
            self.deviationPenaltyFactor = 10.0
        elif self.ENVTYPE == '2':
            self.minH = 6.0
            self.maxH = 12.0
            self.actionSpace = ['up', 'keep', 'down']
            self.actionRange = 5.0
            self.deviationPenaltyFactor = 4.0
        elif self.ENVTYPE == '3':
            self.minH = 6.0
            self.maxH = 10.0
            self.actionSpace = ['up', 'keep', 'down', 'left', 'right']
            self.actionRange = 10.0
            self.deviationPenaltyFactor = 10.0

        self.actionSpaceSize = len(self.actionSpace)

        self.rewardMax = 1000
        self.distanceMax = 97.9

        self._SEED = _seed
        if self._SEED is not None:
            seed(self._SEED)

        self.timeStep = 0
        self.keyPressed = None
        self.initializeSimulators(PATHMF2005, PATHMP6)
        if self.ENVTYPE == '1' or self.ENVTYPE == '2':
            self.initializeAction()
        self.initializeParticle()
        self.initializeModel()
        self.initializeWell(self.minQ, self.maxQ)
        if self.ENVTYPE == '3':
            self.initializeAction()

        self.reward, self.rewardCurrent = 0., 0.

        # initializing trajectories container for potential plotting
        self.trajectories = {}
        for i in ['x', 'y', 'z']:
            self.trajectories[i] = []

        # running MODFLOW to determine steady-state solution as a initial state
        self.runMODFLOW()

        self.state = {}
        self.state['heads'] = self.heads
        if self.ENVTYPE == '1':
            self.state['actionValueNorth'] = self.actionValueNorth
            self.state['actionValueSouth'] = self.actionValueSouth
        elif self.ENVTYPE == '2':
            self.state['actionValue'] = self.actionValue
        elif self.ENVTYPE == '3':
            self.state['actionValueX'] = self.actionValueX
            self.state['actionValueY'] = self.actionValueY

        self.observations = {}
        self.observations['particleCoords'] = self.particleCoords
        self.observations['headsFullField'] = self.heads[0::self.sampleHeadsEvery,
                                                0::self.sampleHeadsEvery,
                                                0::self.sampleHeadsEvery]
        lParticle, cParticle, rParticle = self.cellInfoFromCoordinates(
            [self.particleCoords[0], self.particleCoords[1], self.particleCoords[2]])
        lWell, cWell, rWell = self.cellInfoFromCoordinates(
            [self.wellX, self.wellY, self.wellZ])
        if self.ENVTYPE == '1':
            self.observations['heads'] = [self.actionValueNorth,
                                          self.actionValueSouth]
        elif self.ENVTYPE == '2':
            # this can cause issues with unit testing, as model expects different input 
            self.observations['heads'] = [self.actionValue]
        elif self.ENVTYPE == '3':
            self.observations['heads'] = [self.headSpecNorth,
                                          self.headSpecSouth]
        self.observations['heads'] += [self.heads[lParticle-1, rParticle-1, cParticle-1]]
        self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.particleCoords)
        self.observations['heads'] += [self.heads[lWell-1, rWell-1, cWell-1]]
        self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.wellCoords)
        self.observations['wellQ'] = self.wellQ
        self.observations['wellCoords'] = self.wellCoords

        self.observationsNormalized = {}
        self.observationsNormalized['particleCoords'] = divide(
            copy(self.particleCoords), self.minX + self.extentX)
        self.observationsNormalized['headsFullField'] = divide(self.observations['headsFullField'],
            self.maxH)
        self.observationsNormalized['heads'] = divide(self.observations['heads'],
            self.maxH)
        self.observationsNormalized['wellQ'] = self.wellQ / self.minQ
        self.observationsNormalized['wellCoords'] = divide(
            self.wellCoords, self.minX + self.extentX)

        self.observationsVector = self.observationsDictToVector(
            self.observations)
        self.observationsVectorNormalized = self.observationsDictToVector(
            self.observationsNormalized)

    def step(self, observations, action, rewardCurrent):
        """Perform a single step of forwards simulation."""

        # rendering initial timestep
        if self.timeStep == 0:
            if self.RENDER or self.MANUALCONTROL or self.SAVEPLOT:
                self.render()

        self.timeStep += 1
        self.keyPressed = None
        self.periodSteadiness = False

        if self.ENVTYPE == '1':
            self.getActionValue(action)
        elif self.ENVTYPE == '2':
            self.getActionValue(action)
        elif self.ENVTYPE == '3':
            self.getActionValue(action)

        observations = self.observationsVectorToDict(observations)
        self.particleCoordsBefore = observations['particleCoords']

        # it might be obsolete to feed this back,
        # as it can be stored with the object
        self.rewardCurrent = rewardCurrent

        if self.timeStep > 1:
            # correcting for different reading order
            # why is this necessary?
            self.particleCoords[0] = self.extentX - self.particleCoords[0]

        if self._SEED is not None:
            seed(self._SEED)

        self.initializeState(self.state)
        self.updateModel()
        self.updateWell()

        self.runMODFLOW()
        self.runMODPATH()
        self.evaluateParticleTracking()

        # calculating game reward
        self.reward = self.calculateGameReward(self.trajectories) 

        self.state = {}
        self.state['heads'] = self.heads
        if self.ENVTYPE == '1':
            self.state['actionValueNorth'] = self.actionValueNorth
            self.state['actionValueSouth'] = self.actionValueSouth
        elif self.ENVTYPE == '2':
            self.state['actionValue'] = self.actionValue
        elif self.ENVTYPE == '3':
            self.state['actionValueX'] = self.actionValueX
            self.state['actionValueY'] = self.actionValueY

        self.observations = {}
        self.observationsNormalized = {}
        self.observations['particleCoords'] = self.particleCoords
        self.observations['headsFullField'] = self.heads[0::self.sampleHeadsEvery,
                                                0::self.sampleHeadsEvery,
                                                0::self.sampleHeadsEvery]
        lParticle, cParticle, rParticle = self.cellInfoFromCoordinates(
            [self.particleCoords[0], self.particleCoords[1], self.particleCoords[2]])
        lWell, cWell, rWell = self.cellInfoFromCoordinates(
            [self.wellX, self.wellY, self.wellZ])
        if self.ENVTYPE == '1':
            self.observations['heads'] = [self.actionValueNorth,
                                          self.actionValueSouth]
        elif self.ENVTYPE == '2':
            # this can cause issues with unit testing, as model expects different input 
            self.observations['heads'] = [self.actionValue]
        elif self.ENVTYPE == '3':
            self.observations['heads'] = [self.headSpecNorth,
                                          self.headSpecSouth]
        self.observations['heads'] += [self.heads[lParticle-1, rParticle-1, cParticle-1]]
        self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.particleCoords)
        self.observations['heads'] += [self.heads[lWell-1, rWell-1, cWell-1]]
        self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.wellCoords)
        self.observations['wellQ'] = self.wellQ
        self.observations['wellCoords'] = self.wellCoords
        self.observationsNormalized['particleCoords'] = divide(
            copy(self.particleCoordsAfter), self.minX + self.extentX)
        self.observationsNormalized['headsFullField'] = divide(self.observations['headsFullField'],
            self.maxH)
        self.observationsNormalized['heads'] = divide(self.observations['heads'],
            self.maxH)
        self.observationsNormalized['wellQ'] = self.wellQ / self.minQ
        self.observationsNormalized['wellCoords'] = divide(
            self.wellCoords, self.minX + self.extentX)

        self.observationsVector = self.observationsDictToVector(
            self.observations)
        self.observationsVectorNormalized = self.observationsDictToVector(
            self.observationsNormalized)

        if self.observations['particleCoords'][0] >= self.extentX - self.dCol:
            self.success = True
        else:
            self.success = False

        # checking if particle is within horizontal distance of well
        dx = self.particleCoords[0] - self.wellCoords[0]
        # why would the correction for Y coordinate be necessary
        dy = self.extentY - self.particleCoords[1] - self.wellCoords[1]
        self.distanceWellParticle = sqrt(dx**2 + dy**2)
        if self.distanceWellParticle <= self.wellRadius:
            self.done = True
            self.reward = (self.rewardCurrent) * (-1.0)

        # checking if particle has reached eastern boundary
        if self.particleCoordsAfter[0] >= self.minX + self.extentX - self.dCol:
            self.done = True

        # checking if particle has returned to western boundary
        if self.particleCoordsAfter[0] <= self.minX + self.dCol:
            self.done = True
            self.reward = (self.rewardCurrent) * (-1.0)

        if self.ENVTYPE == '1' or self.ENVTYPE == '3':
            # checking if particle has reached northern boundary
            if self.particleCoordsAfter[1] >= self.minY + self.extentY - self.dRow:
            # if self.particleCoordsAfter[1] >= self.minY + \
            #         self.extentY - self.dRow:
                self.done = True
                self.reward = (self.rewardCurrent) * (-1.0)

        # checking if particle has reached southern boundary
        if self.particleCoordsAfter[1] <= self.minY + self.dRow:
            self.done = True
            self.reward = (self.rewardCurrent) * (-1.0)

        # aborting game if a threshold of steps have been taken
        if self.timeStep == self.maxSteps:
            if self.done != True:
                self.done = True
                self.reward = (self.rewardCurrent) * (-1.0)

        self.rewardCurrent += self.reward

        if self.RENDER or self.MANUALCONTROL or self.SAVEPLOT:
            self.render()

        if self.done:
            # necessary to remove these file handles to release file locks
            del self.mf, self.cbb, self.hdobj

            for f in listdir(self.modelpth):
                # removing files in folder
                remove(join(self.modelpth, f))
            if exists(self.modelpth):
                # removing folder with model files after run
                rmdir(self.modelpth)

        return self.observations, self.reward, self.done, self.info

    def initializeSimulators(self, PATHMF2005=None, PATHMP6=None):
        """Initialize simulators depending on operating system.

        Executables have to be specified or located in simulators subfolder.
        """

        # setting name of MODFLOW and MODPATH executables
        if system() == 'Windows':
            if PATHMF2005 is None:
                self.exe_name = join(self.wrkspc, 'simulators',
                                     'MF2005.1_12', 'bin', 'mf2005'
                                     ) + '.exe'
            elif PATHMF2005 is not None:
                self.exe_name = PATHMF2005
            if PATHMP6 is None:
                self.exe_mp = join(self.wrkspc, 'simulators',
                                   'modpath.6_0', 'bin', 'mp6'
                                   ) + '.exe'
            elif PATHMP6 is not None:
                self.exe_mp += PATHMP6
        elif system() == 'Linux':
            if PATHMF2005 is None:
                self.exe_name = join(self.wrkspc, 'simulators', 'mf2005')
            elif PATHMF2005 is not None:
                self.exe_name = PATHMF2005
            if PATHMP6 is None:
                self.exe_mp = join(self.wrkspc, 'simulators', 'mp6')
            elif PATHMP6 is not None:
                self.exe_mp = PATHMP6
        else:
            print('Operating system is unknown.')

        self.versionMODFLOW = 'mf2005'
        self.versionMODPATH = 'mp6'

    def initializeAction(self):
        """Initialize actions randomly."""
        if self.ENVTYPE == '1':
            self.actionValueNorth = uniform(self.minH, self.maxH)
            self.actionValueSouth = uniform(self.minH, self.maxH)
        elif self.ENVTYPE == '2':
            self.actionValue = uniform(self.minH, self.maxH)
        elif self.ENVTYPE == '3':
            self.action = 'keep'
            self.actionValueX = self.wellX
            self.actionValueY = self.wellY

    def initializeParticle(self):
        """Initialize spawn of particle randomly.

         The particle will be placed on the Western border just east of the
         Western stream with with buffer to boundaries.
         """

        self.particleSpawnBufferY = 20.0
        self.particleX = self.extentX - 1.1 * self.dCol
        ymin = 0.0 + self.particleSpawnBufferY
        ymax = self.extentY - self.particleSpawnBufferY
        self.particleY = uniform(ymin, ymax)
        self.particleZ = self.zTop
        self.particleCoords = [self.particleX, self.particleY, self.particleZ]

    def initializeModel(self):
        """Initialize groundwater flow model."""

        if self.ENVTYPE == '3':
            self.headSpecNorth = uniform(self.minH, self.maxH)
            self.headSpecSouth = uniform(self.minH, self.maxH)
        self.constructingModel()

    def initializeWell(self, minQ, maxQ):
        """Initialize well randomly in the aquifer domain within margins."""

        xmin = 0.0 + self.wellSpawnBufferXWest
        xmax = self.extentX - self.wellSpawnBufferXEast
        ymin = 0.0 + self.wellSpawnBufferY
        ymax = self.extentY - self.wellSpawnBufferY
        self.wellX = uniform(xmin, xmax)
        self.wellY = uniform(ymin, ymax)
        self.wellZ = self.zTop
        self.wellCoords = [self.wellX, self.wellY, self.wellZ]

        # determining layer, row and column corresponding to well location
        l, c, r = self.cellInfoFromCoordinates([self.wellX,
                                                self.wellY,
                                                self.wellZ]
                                               )
        self.wellCellLayer, self.wellCellColumn, self.wellCellRow = l, c, r

        self.wellQ = uniform(minQ, maxQ)
        # adding WEL package to the MODFLOW model
        lrcq = {0: [[l - 1, r - 1, c - 1, self.wellQ]]}
        ModflowWel(self.mf, stress_period_data=lrcq)

    def initializeState(self, state):
        """Initialize aquifer hydraulic head with state from previous step."""

        self.heads = state['heads']

    def updateModel(self):
        """Update model domain for transient simulation."""

        self.constructingModel()

    def updateWell(self):
        """Update model to continue using well."""

        if self.ENVTYPE == '3':
            # updating well location from action taken
            self.wellX = self.actionValueX
            self.wellY = self.actionValueY
            self.wellZ = self.wellZ

            self.wellCoords = [self.wellX, self.wellY, self.wellZ]

            l, c, r = self.cellInfoFromCoordinates([self.wellX,
                                                    self.wellY,
                                                    self.wellZ]
                                                   )
            self.wellCellLayer = l
            self.wellCellColumn = c
            self.wellCellRow = r

        # adding WEL package to the MODFLOW model
        lrcq = {0: [[self.wellCellLayer - 1,
                     self.wellCellRow - 1,
                     self.wellCellColumn - 1,
                     self.wellQ]]}
        self.wel = ModflowWel(self.mf, stress_period_data=lrcq)

    def constructingModel(self):
        """Construct the groundwater flow model used for the arcade game.

        Flopy is used as a MODFLOW wrapper for input file construction.

        A specified head boundary condition is situated on the western, eastern
        and southern boundary. The southern boundary condition can be modified
        during the game. Generally, the western and eastern boundaries promote
        groundwater to flow towards the west. To simplify, all model parameters
        and aquifer thickness is homogeneous throughout.
        """

        # assigning model name and creating model object
        self.mf = Modflow(self.MODELNAME, exe_name=self.exe_name,
                          verbose=False
                          )

        # changing workspace to model path
        # changed line 1065 in mbase.py to suppress console output
        self.mf.change_model_ws(new_pth=self.modelpth)

        # creating the discretization object
        if self.periodSteadiness:
            self.dis = ModflowDis(self.mf, self.nLay,
                                  self.nRow, self.nCol,
                                  delr=self.dRow, delc=self.dCol,
                                  top=self.zTop,
                                  botm=self.botM[1:],
                                  steady=self.periodSteadiness
                                  )
        elif self.periodSteadiness == False:
            self.dis = ModflowDis(self.mf, self.nLay,
                                  self.nRow, self.nCol,
                                  delr=self.dRow, delc=self.dCol,
                                  top=self.zTop,
                                  botm=self.botM[1:],
                                  steady=self.periodSteadiness,
                                  nper=self.periods,
                                  nstp=self.periodSteps,
                                  perlen=self.periodLength
                                  )

        # defining variables for the BAS package
        self.ibound = ones((self.nLay, self.nRow, self.nCol), dtype=int32)
        if self.ENVTYPE == '1' or self.ENVTYPE == '3':
            self.ibound[:, 5:-5, 0] = -1
            self.ibound[:, 5:-5, -1] = -1
            self.ibound[:, -1, 5:-5] = -1
            self.ibound[:, 0, 5:-5] = -1
        elif self.ENVTYPE == '2':
            self.ibound[:, :-5, 0] = -1
            self.ibound[:, :-5, -1] = -1
            self.ibound[:, -1, 5:-5] = -1

        if self.periodSteadiness:
            self.strt = ones((self.nLay, self.nRow, self.nCol),
                             dtype=float32
                             )
        elif self.periodSteadiness == False:
            self.strt = self.heads

        if self.ENVTYPE == '1':
            self.strt[:, 5:-5, 0] = self.headSpecWest
            self.strt[:, 5:-5, -1] = self.headSpecEast
            self.strt[:, -1, 5:-5] = self.actionValueSouth
            self.strt[:, 0, 5:-5] = self.actionValueNorth
        elif self.ENVTYPE == '2':
            self.strt[:, :-5, 0] = self.headSpecWest
            self.strt[:, :-5, -1] = self.headSpecEast
            self.strt[:, -1, 5:-5] = self.actionValue
        elif self.ENVTYPE == '3':
            self.strt[:, 5:-5, 0] = self.headSpecWest
            self.strt[:, 5:-5, -1] = self.headSpecEast
            self.strt[:, -1, 5:-5] = self.headSpecSouth
            self.strt[:, 0, 5:-5] = self.headSpecNorth

        ModflowBas(self.mf, ibound=self.ibound, strt=self.strt)

        # adding LPF package to the MODFLOW model
        ModflowLpf(self.mf, hk=10., vka=10., ipakcb=53)

        # why is this relevant for particle tracking?
        stress_period_data = {}
        for kper in range(self.periods):
            for kstp in range([self.periodSteps][kper]):
                stress_period_data[(kper, kstp)] = ['save head',
                                                    'save drawdown',
                                                    'save budget',
                                                    'print head',
                                                    'print budget'
                                                    ]

        # adding OC package to the MODFLOW model for output control
        ModflowOc(self.mf, stress_period_data=stress_period_data,
            compact=True)

        # adding PCG package to the MODFLOW model
        ModflowPcg(self.mf)

    def runMODFLOW(self):
        """Execute forward groundwater flow simulation using MODFLOW."""

        # writing MODFLOW input files
        self.mf.write_input()
        # self.check = self.mf.check(verbose=False)

        # running the MODFLOW model
        self.successMODFLOW, self.buff = self.mf.run_model(silent=True)
        if not self.successMODFLOW:
            raise Exception('MODFLOW did not terminate normally.')

        # loading simulation heads and times
        self.fnameHeads = join(self.modelpth, self.MODELNAME + '.hds')
        with HeadFile(self.fnameHeads) as hf:
            self.hdobj = hf
            # shouldn't we pick the heads at a specific runtime?
            self.times = self.hdobj.get_times()
            self.heads = self.hdobj.get_data(totim=self.times[-1])

        # loading discharge data
        self.fnameBudget = join(self.modelpth, self.MODELNAME + '.cbc')
        with CellBudgetFile(self.fnameBudget) as cbf:
            self.cbb = cbf
            self.frf = self.cbb.get_data(text='FLOW RIGHT FACE')[0]
            self.fff = self.cbb.get_data(text='FLOW FRONT FACE')[0]

    def runMODPATH(self):
        """Execute forward particle tracking simulation using MODPATH."""

        # creating MODPATH simulation object
        self.mp = Modpath(self.MODELNAME, exe_name=self.exe_mp,
                          modflowmodel=self.mf,
                          model_ws=self.modelpth
                          )

        self.mpbas = ModpathBas(self.mp,
                                hnoflo=self.mf.bas6.hnoflo,
                                hdry=self.mf.lpf.hdry,
                                ibound=self.mf.bas6.ibound.array,
                                prsity=0.2,
                                prsityCB=0.2
                                )

        self.sim = self.mp.create_mpsim(trackdir='forward', simtype='pathline',
                                        packages='RCH')

        # writing MODPATH input files
        self.mp.write_input()

        # manipulating input file to contain custom particle location
        out = []
        keepFlag = True
        fIn = open(join(self.modelpth, self.MODELNAME + '.mpsim'),
                   'r', encoding='utf-8')
        inLines = fIn.readlines()
        for line in inLines:
            if 'rch' in line:
                keepFlag = False
                out.append(self.MODELNAME + '.mploc\n')
                # particle generation option 2
                # budget output option 3
                # TimePointCount (number of TimePoints)
                # is this ever respected?
                out.append(str(2) + '\n')
                # why does removing this work?
                del out[7]
                # TimePoints
                out.append('0.000000   1.000000\n')
            if keepFlag:
                out.append(line)
        fIn.close()

        # writing particle tracking settings to file
        fOut = open(join(self.modelpth, self.MODELNAME + '.mpsim'),
                    'w')
        for line in range(len(out)):
            if 'mplst' in out[line]:
                _ = u'2   1   2   1   1   2   2   3   1   1   1   1\n'
                out[line + 1] = _
            fOut.write(out[line])
        fOut.close()

        # determining layer, row and column corresponding to particle location
        l, c, r = self.cellInfoFromCoordinates(self.particleCoords)

        # determining fractions of current cell to represent particle location
        # in MODPATH input file, with fracCol coordinate correction being
        # critical.
        fracCol = 1.0 - ((self.particleCoords[0] / self.dCol)
                         - ((c - 1) * self.dCol)
                         )
        fracRow = (self.particleCoords[1] / self.dRow) - ((r - 1) * self.dRow)
        fracVer = (self.particleCoords[2] / self.dVer) - ((l - 1) * self.dVer)

        # writing current particle location to file
        fOut = open(join(self.modelpth, self.MODELNAME + '.mploc'),
                    'w')
        fOut.write('1\n')
        fOut.write(u'1\n')
        fOut.write(u'particle\n')
        fOut.write(u'1\n')
        fOut.write(u'1 1 1' +
                   u' ' +
                   str(l) +
                   u' ' +
                   str(self.nRow -
                       r +
                       1) +
                   u' ' +
                   str(self.nCol -
                       c +
                       1) +
                   u' ' +
                   str('%.6f' %
                       (fracCol)) +
                   u' ' +
                   str('%.6f' %
                       fracRow) +
                   u' ' +
                   str('%.6f' %
                       fracVer) +
                   u' 0.000000 ' +
                   u'particle\n')
        # GroupName
        fOut.write('particle\n')
        # LocationCount, ReleaseStartTime, ReleaseOption
        fOut.write('1 0.000000 1\n')
        fOut.close()

        # running the MODPATH model
        self.mp.run_model(silent=True)

    def evaluateParticleTracking(self):
        """Evaluate particle tracking results from MODPATH.

        Determines new particle coordinates after advective transport during the
        game.
        """

        # loading the pathline data
        self.pthfile = join(self.modelpth, self.mp.sim.pathline_file)
        self.pthobj = PathlineFile(self.pthfile)
        self.p0 = self.pthobj.get_data(partid=0)

        # filtering results to select appropriate timestep
        # why is it running for longer?
        self.particleTrajX = extract(self.p0['time'] <= self.periodLength,
                                     self.p0['x']
                                     )
        self.particleTrajY = extract(self.p0['time'] <= self.periodLength,
                                     self.p0['y']
                                     )
        self.particleTrajZ = extract(self.p0['time'] <= self.periodLength,
                                     self.p0['z']
                                     )

        self.trajectories['x'].append(self.particleTrajX)
        self.trajectories['y'].append(self.particleTrajY)
        self.trajectories['z'].append(self.particleTrajZ)

        self.particleCoordsAfter = [self.particleTrajX[-1],
                                    self.particleTrajY[-1],
                                    self.particleTrajZ[-1]
                                    ]

        # changing current particle coordinate to new
        self.particleCoords = copy(self.particleCoordsAfter)

    def calculateGameReward(self, trajectories):
        """Calculate game reward.

        Reward is a function of deviation from the straightmost path to the
        eastern boundary. The penalty for deviation is measured by the ratio
        between the length of the straightmost path along the x axis and the
        length of the actually traveled path.
        """

        x = trajectories['x'][-1]
        y = trajectories['y'][-1]
        lengthShortest = x[-1] - x[0]
        lengthActual = self.calculatePathLength(x, y)

        # this ratio defines the fraction of the highest possible reward
        if lengthActual != 0.:
            pathLengthRatio = lengthShortest / lengthActual
        elif lengthActual == 0.:
            pathLengthRatio = 1.0
            # giving no reward if travelled neither forwards nor backwards
            self.gameReward = 0.

        distanceFraction = lengthShortest / self.distanceMax
        self.rewardMaxGame = (distanceFraction * self.rewardMax)
        self.gameReward = self.rewardMaxGame * \
            (pathLengthRatio**self.deviationPenaltyFactor)

        # negative reward for traveling backwards
        # potential problem: rewards may be reward for going backward an then going
        # less deviating way forward?
        if lengthShortest < 0.:
            self.gameReward *= -1.0 * self.gameReward

        return self.gameReward

    def reset(self, _seed=None, MODELNAME=None):
        """Reset environment with same settings but potentially new seed."""
        self.__init__(
            self.ENVTYPE,
            self.PATHMF2005,
            self.PATHMP6,
            self.MODELNAME if MODELNAME is None else MODELNAME,
            _seed=_seed,
            flagSavePlot=self.SAVEPLOT,
            flagManualControl=self.MANUALCONTROL,
            flagRender=self.RENDER)

        close()

    def render(self):
        """Plot the simulation state at the current timestep.

        Displaying and/or saving the visualisation. The active display can take
        user input from the keyboard to control the environment.
        """
        if self.timeStep == 0:
            self.renderInitializeCanvas()
            self.plotfilesSaved = []
            self.extent = (self.dRow / 2., self.extentX - self.dRow / 2.,
                           self.extentY - self.dCol / 2., self.dCol / 2.
                           )

        self.modelmap = PlotMapView(model=self.mf, layer=0)
        self.grid = self.modelmap.plot_grid(zorder=1, lw=0.1)
        self.headsplot = self.modelmap.plot_array(self.heads,
                                                  masked_values=[999.],
                                                  alpha=0.5, zorder=2,
                                                  cmap=get_cmap('terrain')
                                                  )
        self.quadmesh = self.modelmap.plot_ibound(zorder=3)
        self.quiver = self.modelmap.plot_discharge(self.frf, self.fff,
                                                   head=self.heads,
                                                   alpha=0.1, zorder=4
                                                   )

        self.renderWellSafetyZone(zorder=3)
        # self.renderContourLines(n=30, zorder=4)
        self.renderIdealParticleTrajectory(zorder=5)
        self.renderTextOnCanvasPumpingRate(zorder=10)
        self.renderTextOnCanvasGameOutcome(zorder=10)
        self.renderParticle(zorder=6)
        self.renderParticleTrajectory(zorder=6)
        self.renderTextOnCanvasTimeAndScore(zorder=10)

        self.renderRemoveAxesTicks()
        self.renderSetAxesLimits()
        self.renderSetAxesLabels()

        if self.MANUALCONTROL:
            self.renderUserInterAction()
        elif not self.MANUALCONTROL:
            if self.RENDER:
                show(block=False)
                pause(self.MANUALCONTROLTIME)
        if self.SAVEPLOT:
            self.renderSavePlot()
            if self.done or self.timeStep==self.NAGENTSTEPS:
                self.renderAnimationFromFiles()

        self.renderClearAxes()

    def renderInitializeCanvas(self):
        """Initialize plot canvas with figure and axes."""
        self.fig = figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(1, 1, 1, aspect='equal')
        self.ax3 = self.ax.twinx()
        self.ax2 = self.ax3.twiny()

    def renderIdealParticleTrajectory(self, zorder=5):
        """Plot ideal particle trajectory associated with maximum reward."""
        self.ax2.plot([self.minX, self.minX + self.extentX],
                      [self.particleY, self.particleY],
                      lw=1.5, c='white', linestyle='--', zorder=zorder,
                      alpha=0.5
                      )

    def renderContourLines(self, n=30, zorder=4):
        """Plot n contour lines of the head field."""
        self.levels = linspace(min(self.heads), max(self.heads), n)
        self.contours = self.modelmap.contour_array(self.heads,
            levels=self.levels, alpha=0.5, zorder=zorder)

    def renderWellSafetyZone(self, zorder=3):
        """Plot well safety zone."""
        wellBufferCircle = Circle((self.wellX, self.extentY - self.wellY),
                                  self.wellRadius,
                                  edgecolor='r', facecolor=None, fill=False,
                                  zorder=zorder, alpha=1.0, lw=2.0,
                                  label='protection zone'
                                  )
        self.ax2.add_artist(wellBufferCircle)

    def renderTextOnCanvasPumpingRate(self, zorder=10):
        """Plot pumping rate on figure."""
        self.ax2.text(self.wellX + 3., self.extentY - self.wellY,
            str(int(self.wellQ)) + '\nm3/d',
            fontsize=12, color='black', zorder=zorder
            )

    def renderTextOnCanvasGameOutcome(self, zorder=10):
        """Plot final game outcome on figure."""
        gameResult = ''
        if self.done:
            if self.success:
                gameResult = 'You won.'
            elif self.success == False:
                gameResult = 'You lost.'
        self.ax2.text(35, 80, gameResult,
                      fontsize=30, color='red', zorder=zorder
                      )

    def renderTextOnCanvasTimeAndScore(self, zorder=10):
        """Plot final game outcome on figure."""
        timeString = '%.0f' % (float(self.timeStep) *
                               (self.periodLength - 1.0))
        self.ax2.text(5, 92,
                      'FloPy Arcade game'
                      + ', timestep '
                      + str(int(self.timeStep))
                      + '\nscore: '
                      + str(int(self.rewardCurrent))
                      + '     '
                      + timeString
                      + ' d elapsed',
                      fontsize=12,
                      zorder=zorder
                      )

    def renderParticle(self, zorder=6):
        """Plot particle at current current state."""
        if self.timeStep == 0:
            self.ax2.scatter(self.minX,
                 self.particleCoords[1],
                 lw=4,
                 c='red',
                 zorder=zorder)
        elif self.timeStep > 0:
            self.ax2.scatter(self.trajectories['x'][-1][-1],
                             self.trajectories['y'][-1][-1],
                             lw=2,
                             c='red',
                             zorder=zorder)

    def renderParticleTrajectory(self, zorder=6):
        """Plot particle trajectory until current state."""
        if self.timeStep > 0:

            # generating fading colors
            countCoords, colorsLens = 0, []
            for i in range(len(self.trajectories['x'])):
                countCoords += len(self.trajectories['x'][i])
                colorsLens.append(len(self.trajectories['x'][i]))
            colorsFadeAlphas = linspace(0.1, 1.0, countCoords)
            colorsRGBA = zeros((countCoords, 4))
            colorsRGBA[:, 0] = 1.0
            colorsRGBA[:, 3] = colorsFadeAlphas

            idxCount = 0
            for i in range(len(self.trajectories['x'])):
                self.ax2.plot(self.trajectories['x'][i],
                              self.trajectories['y'][i],
                              lw=2,
                              c=colorsRGBA[idxCount + colorsLens[i] - 1,
                                           :],
                              zorder=zorder)
                idxCount += 1

    def renderRemoveAxesTicks(self):
        """Remove axes ticks from figure."""
        self.ax.set_xticks([]), self.ax.set_yticks([])
        self.ax2.set_xticks([]), self.ax2.set_yticks([])
        if self.ENVTYPE == '1' or self.ENVTYPE == '3':
            self.ax3.set_xticks([]), self.ax3.set_yticks([])

    def renderSetAxesLimits(self):
        """Set limits of axes from given extents of environment domain."""
        self.ax.set_xlim(left=self.minX, right=self.minX + self.extentX)
        self.ax.set_ylim(bottom=self.minY, top=self.minY + self.extentY)
        self.ax2.set_xlim(left=self.minX, right=self.minX + self.extentX)
        self.ax2.set_ylim(bottom=self.minY, top=self.minY + self.extentY)
        self.ax3.set_xlim(left=self.minX, right=self.minX + self.extentX)
        self.ax3.set_ylim(bottom=self.minY, top=self.minY + self.extentY)

    def renderSetAxesLabels(self):
        """Set labels to axes."""
        self.ax.set_ylabel('Start\nwater level:   ' + str('%.2f' %
                                                          self.headSpecWest) + ' m', fontsize=12)
        self.ax3.set_ylabel('water level:   ' + str('%.2f' %
                                                    self.headSpecEast) + ' m\nDestination', fontsize=12)
        if self.ENVTYPE == '1':
            self.ax.set_xlabel('water level:   ' + str('%.2f' %
                                                       self.actionValueSouth) 
                                                       + ' m', fontsize=12)
            self.ax2.set_xlabel('water level:   ' +
                                str('%.2f' %
                                    self.actionValueNorth) +
                                ' m', fontsize=12)
        elif self.ENVTYPE == '2':
            self.ax.set_xlabel('water level:   ' + str('%.2f' %
                                                       self.actionValue) 
                                                       + ' m', fontsize=12)
        elif self.ENVTYPE == '3':
            self.ax.set_xlabel('water level:   ' + 
                               str('%.2f' %
                                   self.headSpecSouth) +
                               ' m',
                               fontsize=12)
            self.ax2.set_xlabel('water level:   ' +
                                str('%.2f' %
                                    self.headSpecNorth) +
                                ' m',
                                fontsize=12)

    def renderUserInterAction(self):
        """Enable user control of the environment."""
        if self.timeStep == 0:
            # determining if called from IPython notebook
            if 'ipykernel' in modules:
                self.flagFromIPythonNotebook = True
            else:
                self.flagFromIPythonNotebook = False

        if self.flagFromIPythonNotebook:
            # changing plot updates of IPython notebooks
            # currently unsolved: need to capture key stroke here as well
            from IPython import display
            display.clear_output(wait=True)
            display.display(self.fig)
        elif not self.flagFromIPythonNotebook:
            self.fig.canvas.mpl_connect(
                'key_press_event', self.captureKeyPress)
            show(block=False)
            waitforbuttonpress(timeout=self.MANUALCONTROLTIME)

    def renderSavePlot(self):
        """Save plot of the currently rendered timestep."""
        if self.timeStep == 0:
            # setting up the path to save results plots in
            self.plotspth = join(self.wrkspc, 'runs')
            # ensuring plotspth directory exists
            if not exists(self.plotspth):
                makedirs(self.plotspth)

        plotfile = join(self.wrkspc, 'runs',
                              self.MODELNAME
                              + str(self.timeStep).zfill(len(str(abs(self.NAGENTSTEPS)))+1)
                              + '.png'
                              )
        self.fig.savefig(plotfile, dpi=70)
        self.plotfilesSaved.append(plotfile)

    def renderClearAxes(self):
        """Clear all axis after timestep."""
        self.ax.cla()
        self.ax.clear()
        self.ax2.cla()
        self.ax2.clear()
        if self.ENVTYPE == '1' or self.ENVTYPE == '3':
            self.ax3.cla()
            self.ax3.clear()

    def render3d(self):
        """Render environment in 3 dimensions."""
        from mpl_toolkits import mplot3d
        import numpy as np
        import matplotlib.pyplot as plt
        test = self.dis.get_node_coordinates()
        # self.fig = plt.figure()
        ax3d = plt.axes(projection='3d')
        xx, yy = np.meshgrid(test[0], test[1], sparse=True)
        ax3d.plot_surface(xx, yy, np.reshape(np.ndarray.flatten(self.heads), (self.nRow, self.nCol)), cmap='viridis', edgecolor='none')
        ax3d.scatter(self.trajectories['x'][-1][-1], self.trajectories['y'][-1][-1],
                 lw=2, c='red', zorder=6
                 )
        show(block=False)
        # waitforbuttonpress(timeout=self.MANUALCONTROLTIME)
        sleep(10)
        plt.close()

    def renderAnimationFromFiles(self):
        """Create animation of fulll game run.
        Code taken from and credit to:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
        """
        with get_writer(join(self.wrkspc, 'runs',
            self.MODELNAME + '.gif'), mode='I') as writer:
            for filename in self.plotfilesSaved:
                image = imread(filename)
                writer.append_data(image)
                remove(filename)

    def cellInfoFromCoordinates(self, coords):
        """Determine layer, row and column corresponding to model location."""

        x, y, z = coords[0], coords[1], coords[2]

        layer = int(ceil((z + self.zBot) / self.dVer))
        column = int(ceil((x + self.minX) / self.dCol))
        row = int(ceil((y + self.minY) / self.dRow))

        return layer, column, row

    def surroundingHeadsFromCoordinates(self, coords):
        """Determine hydraulic head of surrounding cells. Returns head of the
        same cell in the case of surrounding edges of the environment domain.
        """

        l, c, r = self.cellInfoFromCoordinates([coords[0], coords[1],
            coords[2]])

        headsSurrounding = []
        for rIdx in range(3):
            for cIdx in range(3):
                if (rIdx == 1) and (cIdx == 1):
                    pass
                else:
                    try:
                        headsSurrounding.append(self.heads[l-1, r-rIdx, c-cIdx])
                    except:
                        # if surrounding head does not exist at domain boundary
                        headsSurrounding.append(self.heads[l-1, r-1, c-1])

        return headsSurrounding

    def calculatePathLength(self, x, y):
        """Calculate length of advectively traveled path."""

        n = len(x)
        lv = []
        for i in range(n):
            if i > 0:
                lv.append(sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2))
        pathLength = sum(lv)

        return pathLength

    def captureKeyPress(self, event):
        """Capture key pressed through manual user interaction."""

        self.keyPressed = event.key

    def getActionValue(self, action):
        """Retrieve a list of performable actions."""

        if self.ENVTYPE == '1':
            if action == 'up':
                self.actionValueNorth = self.actionValueNorth + self.actionRange
                self.actionValueSouth = self.actionValueSouth + self.actionRange
            elif action == 'down':
                self.actionValueNorth = self.actionValueNorth - self.actionRange
                self.actionValueSouth = self.actionValueSouth - self.actionRange

        elif self.ENVTYPE == '2':
            if action == 'up':
                self.actionValue = self.actionValue + 0.1 * self.actionRange
            elif action == 'down':
                self.actionValue = self.actionValue - 0.1 * self.actionRange

        elif self.ENVTYPE == '3':
            if action == 'up':
                if self.wellY > self.dRow + self.actionRange:
                    self.actionValueY = self.wellY - self.actionRange
            elif action == 'left':
                if self.wellX > self.dCol + self.actionRange:
                    self.actionValueX = self.wellX - self.actionRange
            elif action == 'right':
                if self.wellX < self.extentX - self.dCol - self.actionRange:
                    self.actionValueX = self.wellX + self.actionRange
            elif action == 'down':
                if self.wellY < self.extentY - self.dRow - self.actionRange:
                    self.actionValueY = self.wellY + self.actionRange

    def observationsDictToVector(self, observationsDict):
        """Convert dictionary of observations to list."""
        observationsVector = []
        for obs in observationsDict['particleCoords']:
            observationsVector.append(obs)
        # for obs in observationsDict['headsFullField'].flatten().flatten():
        #     observationsVector.append(obs)
        for obs in observationsDict['heads']:
            observationsVector.append(obs)
        observationsVector.append(observationsDict['wellQ'])
        for obs in observationsDict['wellCoords']:
            observationsVector.append(obs)
        return observationsVector

    def observationsVectorToDict(self, observationsVector):
        """Convert list of observations to dictionary."""
        observationsDict = {}
        observationsDict['particleCoords'] = observationsVector[:3]
        observationsDict['heads'] = observationsVector[3:-5]
        observationsDict['wellQ'] = observationsVector[-4]
        observationsDict['wellCoords'] = observationsVector[:-3]
        return observationsDict


class FloPyArcade():
    """Instance of a FLoPy arcade game.

    Initializes a game agent and environment. Then allows to play the game.
    """

    def __init__(self, agent=None, modelNameLoad=None, modelName='FloPyArcade',
        NAGENTSTEPS=200, PATHMF2005=None, PATHMP6=None,
        flagSavePlot=False, flagManualControl=False, flagRender=False):
        """Constructor."""

        self.PATHMF2005 = PATHMF2005
        self.PATHMP6 = PATHMP6
        self.NAGENTSTEPS = NAGENTSTEPS
        self.SAVEPLOT = flagSavePlot
        self.MANUALCONTROL = flagManualControl
        self.RENDER = flagRender
        self.MODELNAME = modelName if modelName is not None else modelNameLoad
        self.agent = agent
        self.MODELNAMELOAD = modelNameLoad
        self.done = False

    def play(self, env=None, ENVTYPE='1', seed=None):
        """Play an instance of the Flopy arcade game."""

        t0 = time()

        # creating the environment
        if env is None:
            env = FloPyEnv(ENVTYPE, self.PATHMF2005, self.PATHMP6, _seed=seed,
                MODELNAME=self.MODELNAME if not None else 'FloPyArcade',
                flagSavePlot=self.SAVEPLOT,
                flagManualControl=self.MANUALCONTROL,
                flagRender=self.RENDER,
                NAGENTSTEPS=self.NAGENTSTEPS)
        observations, self.done = env.observationsVectorNormalized, env.done
        self.actionRange, self.actionSpace = env.actionRange, env.actionSpace

        agent = FloPyAgent(actionSpace=self.actionSpace)

        # game loop
        self.rewardTotal = 0.
        for game in range(self.NAGENTSTEPS):

            if not self.done:
                # without user control input: generating random agent action
                if self.MANUALCONTROL:
                    action = agent.getAction('manual', env.keyPressed)
                elif self.MANUALCONTROL == False:
                    if self.MODELNAMELOAD is None and self.agent is None:
                        action = agent.getAction('random')
                    elif self.MODELNAMELOAD is not None:
                        action = agent.getAction(
                            'modelNameLoad',
                            modelNameLoad=self.MODELNAMELOAD,
                            state=env.observationsVectorNormalized
                            )
                    elif self.agent is not None:
                        action = agent.getAction(
                            'model',
                            agent=self.agent,
                            state=env.observationsVectorNormalized
                            )

                observations, reward, self.done, _ = env.step(
                    env.observationsVectorNormalized, action, self.rewardTotal)

                self.rewardTotal += reward

            elif self.done:

                if self.MANUALCONTROL:
                    # freezing screen shortly when game is done
                    sleep(5)

                self.success = env.success
                if env.success:
                    successString = 'won'
                elif env.success == False:
                    successString = 'lost'
                    # total loss of reward if entering well protection zone
                    self.rewardTotal = 0.0

                print('The game was ' +
                      successString +
                      ' after ' +
                      str(game) +
                      ' timesteps with a reward of ' +
                      str(int(self.rewardTotal)) +
                      ' points.')
                close('all')
                break

        self.gamesPlayed = game
        self.runtime = (time() - t0) / 60.