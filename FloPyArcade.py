#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


# imports for environments
from matplotlib import use as matplotlibBackend
matplotlibBackend('Agg')
from flopy.modflow import Modflow, ModflowBas, ModflowDis, ModflowLpf
from flopy.modflow import ModflowOc, ModflowPcg, ModflowWel
from flopy.modpath import Modpath6 as Modpath
from flopy.modpath import Modpath6Bas as ModpathBas
from flopy.plot import PlotMapView
from flopy.utils import CellBudgetFile, HeadFile, PathlineFile
from imageio import get_writer, imread
from itertools import product
from joblib import dump as joblibDump
from joblib import load as joblibLoad
from math import ceil, floor
from matplotlib.cm import get_cmap
from matplotlib.pyplot import Circle, close, figure, pause, show
from matplotlib.pyplot import get_current_fig_manager
from matplotlib.pyplot import margins, NullLocator
from matplotlib.pyplot import waitforbuttonpress
from numpy import abs, add, arange, argmax, argsort, array, ceil, copy, divide
from numpy import extract, float32, fromstring, int32, linspace, max, maximum, min, minimum
from numpy import mean, ones, shape, sqrt, subtract, sum, uint8, zeros
from numpy.random import randint, random, randn, uniform
from numpy.random import seed as numpySeed
from os import environ, listdir, makedirs, remove, rmdir
from os.path import abspath, dirname, exists, join
from platform import system
from sys import modules
if 'ipykernel' in modules:
    from IPython import display
from time import sleep, time


# suppressing TensorFlow output on import, except fatal errors
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
from logging import getLogger, FATAL
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
getLogger('tensorflow').setLevel(FATAL)

# additional imports for agents
from collections import deque, defaultdict
from datetime import datetime
from gc import collect as garbageCollect
from itertools import count
from pathos import helpers as pathosHelpers
from pathos.pools import _ProcessPool as Pool
from pathos.pools import _ThreadPool as ThreadPool
from pickle import dump, load
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Activation, BatchNormalization, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import clone_model, load_model, model_from_json
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from random import sample as randomSample, seed as randomSeed
from tensorflow.compat.v1 import ConfigProto, set_random_seed
from tensorflow.compat.v1 import Session as TFSession
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import load_model as TFload_model
from tqdm import tqdm
from uuid import uuid4

# avoiding freeze issues on Linux when loading Tensorflow model
# https://github.com/keras-team/keras/issues/9964
# https://stackoverflow.com/questions/40615795/pathos-enforce-spawning-on-linux
pathosHelpers.mp.context._force_start_method('spawn')


class FloPyAgent():
    """Agent to navigate a spawned particle advectively through one of the
    aquifer environments, collecting reward along the way.
    """

    def __init__(self, observationsVector=None, actionSpace=['keep'],
                 hyParams=None, envSettings=None, mode='random',
                 maxTasksPerWorker=20, maxTasksPerWorkerMutate=10,
                 maxTasksPerWorkerNoveltySearch=100000, zFill=6):
        """Constructor"""

        self.wrkspc = dirname(abspath(__file__))
        if 'library.zip' in self.wrkspc:
            # changing workspace in case of call from compiled executable
            self.wrkspc = dirname(dirname(self.wrkspc))

        # initializing arguments
        self.observationsVector = observationsVector
        self.actionSpace = actionSpace
        self.actionSpaceSize = len(self.actionSpace)
        self.hyParams, self.envSettings = hyParams, envSettings
        self.agentMode = mode
        self.maxTasksPerWorker = maxTasksPerWorker
        self.maxTasksPerWorkerMutate = maxTasksPerWorkerMutate
        self.maxTasksPerWorkerNoveltySearch = maxTasksPerWorkerNoveltySearch
        self.zFill = zFill

        # setting seeds
        if self.envSettings is not None:
            self.SEED = self.envSettings['SEEDAGENT']
            numpySeed(self.SEED)
            randomSeed(self.SEED)
            set_random_seed(self.SEED)

        # creating required folders if inexistent
        self.modelpth = join(self.wrkspc, 'models')
        if not exists(self.modelpth):
            makedirs(self.modelpth)

        if self.agentMode == 'DQN':
            # initializing DQN agent
            self.initializeDQNAgent()

        if self.agentMode == 'genetic':
            # creating required folders if inexistent
            self.tempModelpth = join(self.wrkspc, 'temp', 
                self.envSettings['MODELNAME'])
            if not exists(self.tempModelpth):
                makedirs(self.tempModelpth)
            if self.envSettings['BESTAGENTANIMATION']:
                runModelpth = join(self.wrkspc, 'runs',
                    self.envSettings['MODELNAME'])
                if not exists(runModelpth):
                    makedirs(runModelpth)
            if self.hyParams is not None:
                if self.hyParams['NOVELTYSEARCH']:
                    self.tempNoveltypth = join(self.tempModelpth, 'novelties')
                    if not exists(self.tempNoveltypth):
                        makedirs(self.tempNoveltypth)
                    else:
                        if self.envSettings is not None:
                            if not self.envSettings['RESUME']:
                                if exists(self.tempNoveltypth):
                                    for f in listdir(self.tempNoveltypth):
                                        remove(join(self.tempNoveltypth, f))

            # initializing genetic agents and saving hyperparameters and
            # environment settings or loading them if resuming
            if not self.envSettings['RESUME']:
                self.geneticGeneration = 0
                self.initializeGeneticAgents()
                self.pickleDump(join(self.tempModelpth,
                    self.envSettings['MODELNAME'] + '_hyParams.p'),
                    self.hyParams)
                self.pickleDump(join(self.tempModelpth,
                    self.envSettings['MODELNAME'] + '_envSettings.p'),
                    self.envSettings)
            elif self.envSettings['RESUME']:
                self.hyParams = self.pickleLoad(join(self.tempModelpth,
                    self.envSettings['MODELNAME'] + '_hyParams.p'))

    def initializeDQNAgent(self):
        """Initialize agent to perform Deep Double Q-Learning.
        Bug fix for predict function:
        https://github.com/keras-team/keras/issues/6462
        """

        # actionType = self.envSettings

        actionType = FloPyEnv(initWithSolution=False).getActionType(self.envSettings['ENVTYPE'])
        
        # initializing main predictive and target model
        self.mainModel = self.createNNModel(actionType)
        # self.mainModel._make_predict_function()
        self.targetModel = self.createNNModel(actionType)
        # self.targetModel._make_predict_function()
        self.targetModel.set_weights(self.mainModel.get_weights())

        # initializing array with last training data of specified length
        self.replayMemory = deque(maxlen=self.hyParams['REPLAYMEMORYSIZE'])
        self.epsilon = self.hyParams['EPSILONINITIAL']

        # initializing counter for updates on target network
        self.targetUpdateCount = 0

    def initializeGeneticAgents(self):
        """Initialize genetic ensemble of agents."""

        chunksTotal = self.yieldChunks(arange(self.hyParams['NAGENTS']),
            self.envSettings['NAGENTSPARALLEL']*self.maxTasksPerWorkerMutate)
        for chunk in chunksTotal:
            _ = self.multiprocessChunks(self.randomAgentGenetic, chunk)

    # to avoid thread erros
    # https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
    # matplotlibBackend('Agg')

    def runDQN(self, env):
        """
        Run main pipeline for Deep Q-Learning optimisation.
        # Inspiration and larger parts of code modified after sentdex
        # https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
        """

        self.actionType = env.actionType

        # generating seeds to generate reproducible cross-validation data
        # note: avoids variability from averaged new games
        numpySeed(self.envSettings['SEEDAGENT'])
        self.seedsCV = randint(self.envSettings['SEEDAGENT'],
            size=self.hyParams['NGAMESCROSSVALIDATED']
            )

        gameRewards = []
        # iterating over games being played
        for iGame in tqdm(range(1, self.hyParams['NGAMES']+1), ascii=True,
            unit='games'):
            env.reset() # no need for seed?

            # simulating, updating replay memory and training main network
            self.takeActionsUpdateAndTrainDQN(env)
            if env.success:
                self.gameReward = self.gameReward
            elif env.success == False:
                # overwriting simulation memory to zero if no success
                # to test: is it better to give reward an not reset to 0?
                self.gameReward = 0.0
                self.updateReplayMemoryZeroReward(self.gameStep) # is this better on or off?
            gameRewards.append(self.gameReward)

            # cross validation, after every given number of games
            if not iGame % self.hyParams['CROSSVALIDATEEVERY'] or iGame == 1:
                self.crossvalidateDQN(env)
    
                MODELNAME = self.envSettings['MODELNAME']
                DQNfstring = f'{MODELNAME}{iGame:_>7.0f}ep'\
                    f'{self.max_rewardCV:_>7.1f}max'\
                    f'{self.average_rewardCV:_>7.1f}avg'\
                    f'{self.min_rewardCV:_>7.1f}min'\
                    f'{datetime.now().strftime("%Y%m%d%H%M%S")}datetime.h5'
                if self.average_rewardCV >= self.envSettings['REWARDMINTOSAVE']:
                    # saving model if larger than a specified reward threshold
                    self.mainModel.save(join(self.wrkspc, 'models', DQNfstring))

            # decaying epsilon
            if self.epsilon > self.hyParams['EPSILONMIN']:
                self.epsilon *= self.hyParams['EPSILONDECAY']
                self.epsilon = max([self.hyParams['EPSILONMIN'], self.epsilon])

    def runGenetic(self, env, noveltySearch=False):
        """Run main pipeline for genetic agent optimisation.
        # Inspiration and larger parts of code modified after and inspired by:
        # https://github.com/paraschopra/deepneuroevolution
        # https://arxiv.org/abs/1712.06567
        """

        self.actionType = env.actionType

        if self.hyParams['NOVELTYSEARCH'] and not self.envSettings['KEEPMODELHISTORY']:
            raise ValueError('Settings and hyperparameters require changes: ' + \
                             'If model history is not kept, novelty search cannot ' + \
                             'be performed as it updates novelty archive regularly ' + \
                             'and might require loading agents of previous generations.')

        if self.hyParams['NAGENTS'] <= self.hyParams['NNOVELTYELITES']:
            raise ValueError('Settings and hyperparameters require changes: ' + \
                             'The number of novelty elites considered during novelty search ' + \
                             'should be lower than the number of agents considered.' + \
                             'to evolve.')

        # setting environment and number of games
        self.env, n = env, self.hyParams['NGAMESAVERAGED']
        if noveltySearch:
            self.noveltySearch, self.noveltyArchive = noveltySearch, {}
            self.noveltyItemCount = 0
            self.agentsUnique, self.agentsUniqueIDs = [], []
            self.agentsDuplicate = []
            if env.actionType == 'discrete':
                self.actionsUniqueIDMapping = defaultdict(count().__next__)
        cores = self.envSettings['NAGENTSPARALLEL']
        # generating unique process ID from system time
        self.pid = str(uuid4())

        agentCounts = [iAgent for iAgent in range(self.hyParams['NAGENTS'])]
        self.rereturnChildrenGenetic = False
        for self.geneticGeneration in range(self.hyParams['NGENERATIONS']):
            self.flagSkipGeneration = False
            self.generatePathPrefixes()

            if self.envSettings['RESUME']:
                if self.noveltySearch:
                    if self.geneticGeneration > 0:
                        self.noveltyArchive = self.pickleLoad(join(
                            self.tempPrevModelPrefix + '_noveltyArchive.p'))
                        self.noveltyItemCount = len(self.noveltyArchive.keys())
                sortedParentIdxs, continueFlag, breakFlag = self.resumeGenetic()
                if continueFlag: continue
                if breakFlag: break

                if self.noveltySearch:
                    # regenerating list of unique and duplicate agents
                    # in case of resume
                    for iAgent in range(self.noveltyItemCount):
                        agentStr = 'agent' + str(iAgent+1)
                        # self.noveltyArchive[agentStr] = {}
                        tempAgentPrefix = self.noveltyArchive[agentStr]['modelFile'].replace('.h5', '')
                        pth = join(tempAgentPrefix + '_results.p')
                        actions = self.pickleLoad(pth)['actions']
                        if env.actionType == 'discrete':
                            actionsAll = [action for actions_ in actions for action in actions_]
                            actionsUniqueID = self.actionsUniqueIDMapping[tuple(actionsAll)]
                            self.noveltyArchive[agentStr]['actionsUniqueID'] = actionsUniqueID
                            if actionsUniqueID not in self.agentsUniqueIDs:
                                # checking if unique ID from actions already exists
                                self.agentsUnique.append(iAgent)
                                self.agentsUniqueIDs.append(actionsUniqueID)
                            else:
                                self.agentsDuplicate.append(iAgent)

            # simulating agents in environment, returning average of n runs
            self.rewards = self.runAgentsRepeatedlyGenetic(agentCounts, n, env)
            # sorting by rewards in reverse, starting with indices of top reward
            # https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
            sortedParentIdxs = argsort(
                self.rewards)[::-1][:self.hyParams['NAGENTELITES']]
            self.bestAgentReward = self.rewards[sortedParentIdxs[0]]
            self.pickleDump(join(self.tempModelPrefix +
                '_agentsSortedParentIndexes.p'), sortedParentIdxs)

            if self.noveltySearch:
                print('Performing novelty search')
                # iterating through agents and storing with novelty in archive
                # calculating average nearest-neighbor novelty score
                for iAgent in range(self.hyParams['NAGENTS']):
                    noveltiesAgent, actionsAll = [], []
                    itemID = self.noveltyItemCount
                    k = self.noveltyItemCount
                    agentStr = 'agent' + str(k+1)
                    self.noveltyArchive[agentStr] = {}
                    # self.noveltyArchive[agentStr]['novelties'] = {}
                    tempAgentPrefix = join(self.tempModelPrefix + '_agent'
                        + str(iAgent + 1).zfill(self.zFill))
                    modelFile = tempAgentPrefix + '.h5'
                    pth = join(tempAgentPrefix + '_results.p')
                    actions = self.pickleLoad(pth)['actions']
                    actionsAll = [action for actions_ in actions for action in actions_]
                    # https://stackoverflow.com/questions/38291372/assign-unique-id-to-list-of-lists-in-python-where-duplicates-get-the-same-id

                    if env.actionType == 'discrete':
                        actionsUniqueID = self.actionsUniqueIDMapping[tuple(actionsAll)]
                        self.noveltyArchive[agentStr]['actionsUniqueID'] = actionsUniqueID
                    self.noveltyArchive[agentStr]['itemID'] = itemID
                    self.noveltyArchive[agentStr]['modelFile'] = modelFile
                    # self.noveltyArchive[agentStr]['actions'] = actions

                    if env.actionType == 'discrete':
                        if not self.noveltyItemCount > self.hyParams['NNOVELTYNEIGHBORS']:
                                # removing duplicate novelty calculation only in case neighbour limit is not reached
                                # as otherwise the same novelty might not apply
                                if actionsUniqueID not in self.agentsUniqueIDs:
                                    # checking if unique ID from actions already exists
                                    self.agentsUnique.append(k)
                                    self.agentsUniqueIDs.append(actionsUniqueID)
                                else:
                                    self.agentsDuplicate.append(k)
                        else:
                            self.agentsUnique, self.agentsUniqueIDs, self.agentsDuplicate = [], [], []
                            # otherwise computed as if unique to avoid assigning novelty
                            # from duplicates with different nearest neighbours
                            for iNov in range(self.noveltyItemCount+1):
                                self.agentsUnique.append(iNov)
                                self.noveltyArchive['agent' + str(iNov+1)]['actionsUniqueID'] = iNov
                                # self.agentsUniqueIDs.append(iNov)
                    elif env.actionType == 'continuous':
                            self.agentsUnique, self.agentsUniqueIDs, self.agentsDuplicate = [], [], []
                            for iNov in range(self.noveltyItemCount+1):
                                self.agentsUnique.append(iNov)
                                self.noveltyArchive['agent' + str(iNov+1)]['actionsUniqueID'] = iNov
                    self.noveltyItemCount += 1

                if not self.noveltyItemCount > self.hyParams['NNOVELTYNEIGHBORS']:
                    print('Novelty search:', len(self.agentsUnique), 'unique agents', len(self.agentsDuplicate), 'duplicate agents')
                else:
                    print('Novelty search')
                # updating novelty of unique agents
                # Note: This can become a massive bottleneck with increasing
                # number of stored agent information and generations
                # despite parallelization
                noveltiesUniqueAgents, t0 = [], time()
                args = [iAgent for iAgent in self.agentsUnique]

                chunksTotal = self.yieldChunks(args,
                    cores*self.maxTasksPerWorkerNoveltySearch)
                for chunk in chunksTotal:
                    noveltiesPerAgent = self.multiprocessChunks(
                        self.calculateNoveltyPerAgent, chunk)
                    noveltiesUniqueAgents += noveltiesPerAgent

                    # self.noveltyArchive[agentStr]['modelFile'] = modelFile

                # calculating novelty of unique agents
                for iUniqueAgent in self.agentsUnique:
                    agentStr = 'agent' + str(iUniqueAgent+1)
                    actionsUniqueID = self.noveltyArchive[agentStr]['actionsUniqueID']
                    novelty = noveltiesUniqueAgents[actionsUniqueID]
                    self.noveltyArchive[agentStr]['novelty'] = novelty
                # updating novelty of duplicate agents from existing value
                for iDuplicateAgent in self.agentsDuplicate:
                    # finding ID of agent representing duplicate agent's actions
                    agentStr = 'agent' + str(iDuplicateAgent+1)
                    actionsUniqueID = self.noveltyArchive[agentStr]['actionsUniqueID']
                    novelty = noveltiesUniqueAgents[actionsUniqueID]
                    self.noveltyArchive[agentStr]['novelty'] = novelty

                self.pickleDump(join(self.tempModelPrefix +
                    '_noveltyArchive.p'), self.noveltyArchive)
                self.novelties, self.noveltyFilenames = [], []
                for k in range(self.noveltyItemCount):
                    agentStr = 'agent' + str(k+1)
                    self.novelties.append(
                        self.noveltyArchive[agentStr]['novelty'])
                    self.noveltyFilenames.append(
                        self.noveltyArchive[agentStr]['modelFile'])
                # print('len(self.noveltyFilenames)', len(self.noveltyFilenames))
                print('Finished novelty search, took', int(time()-t0), 's')

            # returning best-performing agents
            self.returnChildrenGenetic(sortedParentIdxs)
            if self.geneticGeneration+1 >= self.hyParams['ADDNOVELTYEVERY']:
                print('lowest novelty', min(self.novelties))
                print('average novelty', mean(self.novelties))
                print('highest novelty', max(self.novelties))

            MODELNAME = self.envSettings['MODELNAME']
            MODELNAMEGENCOUNT = (MODELNAME + '_gen' +
                str(self.geneticGeneration + 1).zfill(self.zFill) + '_avg' +
                str('%.1f' % (max(self.rewards))))
            # saving best agent of the current generation
            self.saveBestAgent(MODELNAME)

            if self.envSettings['BESTAGENTANIMATION']:
                if not self.flagSkipGeneration:
                    self.saveBestAgentAnimation(env, self.bestAgentFileName,
                        MODELNAMEGENCOUNT, MODELNAME)
            if not self.envSettings['KEEPMODELHISTORY']:
                # removing stored agent models of finished generation
                # as the storage requirements can be substantial
                for agentIdx in range(self.hyParams['NAGENTS']):
                    remove(join(self.tempModelPrefix + '_agent' +
                        str(agentIdx + 1).zfill(self.zFill) + '.h5'))

    def createNNModel(self, actionType, seed=None):
        """Create fully-connected feed-forward multi-layer neural network."""
        if seed is None:
            seed = self.SEED
        model = Sequential()
        initializer = glorot_uniform(seed=seed)
        nHiddenNodes = copy(self.hyParams['NHIDDENNODES'])
        # resetting numpy seeds to generate reproducible architecture
        numpySeed(seed)

        # applying architecture (variable number of nodes per hidden layer)
        if self.agentMode == 'genetic' and self.hyParams['ARCHITECTUREVARY']:
            for layerIdx in range(len(nHiddenNodes)):
                nHiddenNodes[layerIdx] = randint(2, self.hyParams['NHIDDENNODES'][layerIdx]+1)
        for layerIdx in range(len(nHiddenNodes)):
            inputShape = shape(self.observationsVector) if layerIdx == 0 else []
            model.add(Dense(units=nHiddenNodes[layerIdx],
                input_shape=inputShape,
                kernel_initializer=glorot_uniform(seed=seed),
                bias_initializer='zeros',
                use_bias=True))
            if self.hyParams['BATCHNORMALIZATION']:
                model.add(BatchNormalization())
            model.add(Activation(self.hyParams['HIDDENACTIVATIONS'][layerIdx]))
            if 'DROPOUTS' in self.hyParams:
                if self.hyParams['DROPOUTS'][layerIdx] != 0.0:
                    model.add(Dropout(self.hyParams['DROPOUTS'][layerIdx]))

        # adding output layer
        if actionType == 'discrete':
            model.add(Dense(self.actionSpaceSize, activation='linear',
                kernel_initializer=initializer), bias_initializer='zeros')
        elif actionType == 'continuous':
            # sigmoid used here as actions are predicted as fraction of actionRange
            model.add(Dense(self.actionSpaceSize, activation='sigmoid',
                kernel_initializer=initializer, bias_initializer='zeros'))

        # compiling to avoid warning while saving agents in genetic search
        # specifics are irrelevant, as genetic models are not optimized
        # along gradients
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001),
                      metrics=['mean_squared_error'])

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
            self.targetUpdateCount += 1

        # updating target network with weights of main network,
        # if counter reaches set value
        if self.targetUpdateCount > self.hyParams['UPDATEPREDICTIVEMODELEVERY']:
            self.targetModel.set_weights(self.mainModel.get_weights())
            self.targetUpdateCount = 0

    def takeActionsUpdateAndTrainDQN(self, env):
        """
        Take an action and update as well as train the main network every
        step during a game. Acts as the main operating body of the runDQN
        function.
        """

        # retrieving initial state
        current_state = env.observationsVectorNormalized

        # resetting counters prior to restarting game
        # env.stepInitial()
        self.gameReward, self.gameStep, done = 0, 1, False
        for game in range(self.hyParams['NAGENTSTEPS']):
            # epsilon defines the fraction of random to queried actions
            if random() > self.epsilon:
                # retrieving action from Q table
                if env.actionType == 'discrete':
                    actionIdx = argmax(self.getqsGivenAgentModel(
                        self.mainModel, env.observationsVectorNormalized))
                    action = self.actionSpace[actionIdx]
                elif env.actionType == 'continuous':
                    action = self.getqsGivenAgentModel(
                        self.mainModel, env.observationsVectorNormalized)
            else:
                # retrieving random action
                if env.actionType == 'discrete':
                    actionIdx = randint(0, self.actionSpaceSize)
                    action = self.actionSpace[actionIdx]
                elif env.actionType == 'continuous':
                    action = list(uniform(low=0.0, high=1.0, size=self.actionSpaceSize))

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

    def crossvalidateDQN(self, env):
        """Simulate a given number of games and cross-validate current DQN
        success.
        """

        # loop to cross-validate on unique set of models
        self.gameRewardsCV = []
        for iGame in range(self.hyParams['NGAMESCROSSVALIDATED']):
            # resetting variables and environment
            self.gameReward, step, done = 0.0, 0, False
            seedCV = self.seedsCV[iGame]
            env.reset(seedCV, initWithSolution=env.initWithSolution)

            current_state = env.observationsVectorNormalized
            # iterating until game ends
            for _ in range(self.hyParams['NAGENTSTEPS']):
                # querying for Q values
                if env.actionType == 'discrete':
                    actionIdx = argmax(self.getqsGivenAgentModel(
                        self.mainModel, env.observationsVectorNormalized))
                    action = self.actionSpace[actionIdx]
                elif env.actionType == 'continuous':
                    action = self.getqsGivenAgentModel(
                        self.mainModel, env.observationsVectorNormalized)
                # simulating and counting total reward
                new_state, reward, done, info = env.step(
                    env.observationsVectorNormalized, action,
                    self.gameReward)
                self.gameReward += reward
                if self.envSettings['RENDER']:
                    if not iGame % self.envSettings['RENDEREVERY']:
                        if not done: env.render()
                current_state = new_state
                step += 1
                if done: break

            if not env.success:
                self.gameReward = 0.0
            self.gameRewardsCV.append(self.gameReward)

        self.average_rewardCV = mean(self.gameRewardsCV)
        self.min_rewardCV = min(self.gameRewardsCV)
        self.max_rewardCV = max(self.gameRewardsCV)

    def runAgentsGenetic(self, agentCounts, env):
        """Run genetic agent optimisation, if opted for with multiprocessing.
        """

        # running in parallel if specified
        # debug: add if available number of CPU cores is exceeded
        cores = self.envSettings['NAGENTSPARALLEL']

        if __name__ == 'FloPyArcade':
            if self.envSettings['SURROGATESIMULATOR'] is not None:
                # removing environment in case of surrogate model
                # as TensorFlow model cannot be pickled
                if hasattr(self, 'env'):
                    del self.env

            reward_agents, runtimes = [], []
            runtimeGenEstimate, runtimeGensEstimate = None, None
            generationsRemaining = (self.hyParams['NGENERATIONS'] -
                (self.geneticGeneration + 1))
            chunksTotal = self.yieldChunks(agentCounts,
                cores*self.maxTasksPerWorker)
            nChunks = ceil((max(agentCounts)+1)/(cores*self.maxTasksPerWorker))
            nChunks *= (self.hyParams['NGAMESAVERAGED']+1-self.currentGame)
            nChunksRemaining = copy(nChunks)
            # batch processing to avoid memory explosion
            # https://stackoverflow.com/questions/18414020/memory-usage-keep-growing-with-pythons-multiprocessing-pool/20439272
            for chunk in chunksTotal:
                t0 = time()
                if len(runtimes) == 0:
                    runtimeGenEstimate, runtimeGensEstimate = '?', '?'
                elif len(runtimes) != 0:
                    runtimeGenEstimate = mean(runtimes) * nChunksRemaining
                    runtimeGensEstimate = mean(runtimes) * nChunksRemaining
                    runtimeGensEstimate += ((mean(runtimes) * nChunks) *
                        generationsRemaining)
                    runtimeGenEstimate = '%.3f' % (runtimeGenEstimate/(60*60))
                    runtimeGensEstimate = '%.3f' % (runtimeGensEstimate/(60*60))

                print('Currently: ' + str(min(chunk)) + '/' + 
                      str(max(agentCounts)+1) + ' agents, ' +
                      str(self.currentGame) + '/' +
                      str(self.hyParams['NGAMESAVERAGED']) + ' games, ' +
                      str(self.geneticGeneration + 1) + '/' +
                      str(self.hyParams['NGENERATIONS']) + ' generations\n' +
                      runtimeGenEstimate + ' h for generation, ' +
                      runtimeGensEstimate + ' h for all generations')
                print('----------')
                reward_chunks = self.multiprocessChunks(
                    self.runAgentsGeneticSingleRun, chunk)
                reward_agents += reward_chunks
                runtimes.append(time() - t0)
                nChunksRemaining -= 1

        return reward_agents

    def runAgentsGeneticSingleRun(self, agentCount):
        """Run single game within genetic agent optimisation."""

        # print('debug running', self.tempModelPrefix + '_agent'
        #     + str(agentCount + 1).zfill(self.zFill))
        tempAgentPrefix = join(self.tempModelPrefix + '_agent'
            + str(agentCount + 1).zfill(self.zFill))
        t0load_model = time()
        # loading specific agent and weights with given ID
        agent = load_model(join(tempAgentPrefix + '.h5'), compile=False)
        # print('debug duration load_model compiled', time() - t0load_model)

        MODELNAMETEMP = ('Temp' + self.pid +
            '_' + str(agentCount + 1))
        SEEDTEMP = self.envSettings['SEEDENV'] + self.currentGame-1
        if self.envSettings['SURROGATESIMULATOR'] is None:
            env = self.env
            # resetting to unique temporary folder to enable parallelism
            # Note: This will resimulate the initial environment state
            env.reset(MODELNAME=MODELNAMETEMP, _seed=SEEDTEMP)
        elif self.envSettings['SURROGATESIMULATOR'] is not None:
            # this must be initialized here as surrogate TensorFlow models
            # cannot be pickled for use in parallel operation
            env = FloPyEnvSurrogate(self.envSettings['SURROGATESIMULATOR'],
                self.envSettings['ENVTYPE'],
                MODELNAME=MODELNAMETEMP, _seed=SEEDTEMP,
                NAGENTSTEPS=self.hyParams['NAGENTSTEPS'])
    
        results, keys = {}, ['trajectories', 'actions', 'rewards', 'wellCoords']
        if self.currentGame == 1:
            trajectories = [[] for _ in range(self.hyParams['NGAMESAVERAGED'])]
            actions = [[] for _ in range(self.hyParams['NGAMESAVERAGED'])]
            rewards = [[] for _ in range(self.hyParams['NGAMESAVERAGED'])]
            wellCoords = [[] for _ in range(self.hyParams['NGAMESAVERAGED'])]
        elif self.currentGame > 1:
            pth = join(tempAgentPrefix + '_results.p')
            results = self.pickleLoad(pth)
            trajectories, actions = results[keys[0]], results[keys[1]]
            rewards, wellCoords = results[keys[2]], results[keys[3]]

        r, t0game = 0, time()
        for step in range(self.hyParams['NAGENTSTEPS']):
            if env.actionType == 'discrete':
                actionIdx = argmax(self.getqsGivenAgentModel(agent,
                    env.observationsVectorNormalized))
                action = self.actionSpace[actionIdx]
            if env.actionType == 'continuous':
                action = self.getqsGivenAgentModel(
                    agent, env.observationsVectorNormalized)
            
            t0step = time()
            # note: need to feed normalized observations
            new_observation, reward, done, info = env.step(
                env.observationsVectorNormalized, action, r)
            # print('debug duration step', time() - t0step)
            actions[self.currentGame-1].append(action)
            rewards[self.currentGame-1].append(reward)
            wellCoords[self.currentGame-1].append(env.wellCoords)
            r += reward
            if self.envSettings['RENDER']:
                env.render()

            if done or (step == self.hyParams['NAGENTSTEPS']-1): # or if reached end
                # print('debug duration game', time() - t0game)
                if env.success == False:
                    r = 0
                # saving specific simulation results pertaining to agent
                trajectories[self.currentGame-1].append(env.trajectories)
                objects = [trajectories, actions, rewards, wellCoords]
                for i, objectCurrent in enumerate(objects):
                    results[keys[i]] = objectCurrent
                pth = join(tempAgentPrefix + '_results.p')
                self.pickleDump(pth, results)
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
            print('Currently: ' + str(game + 1) + '/' + str(n) + ' games, ' +
                  str(self.geneticGeneration + 1) + '/' +
                  str(self.hyParams['NGENERATIONS']) + ' generations')
            rewardsAgentsCurrent = self.runAgentsGenetic(agentCounts, env)
            reward_agentsMin = minimum(reward_agentsMin, rewardsAgentsCurrent)
            reward_agentsMax = maximum(reward_agentsMax, rewardsAgentsCurrent)
            reward_agentsMean = add(reward_agentsMean, rewardsAgentsCurrent)
        reward_agentsMean = divide(reward_agentsMean, n)

        prefix = self.tempModelPrefix
        self.pickleDump(prefix + '_agentsRewardsMin.p', reward_agentsMin)
        self.pickleDump(prefix + '_agentsRewardsMax.p', reward_agentsMax)
        self.pickleDump(prefix + '_agentsRewardsMean.p', reward_agentsMean)

        return reward_agentsMean

    def randomAgentGenetic(self, agentIdx, generation=1):
        """Creates an agent for genetic optimisation and saves
        it to disk individually.
        """

        actionType = FloPyEnv(initWithSolution=False).getActionType(self.envSettings['ENVTYPE'])
        agent = self.createNNModel(actionType, seed=self.envSettings['SEEDAGENT']+agentIdx)
        agent.save(join(self.tempModelpth, self.envSettings['MODELNAME'] +
            '_gen' + str(generation).zfill(self.zFill) + '_agent' +
            str(agentIdx + 1).zfill(self.zFill) + '.h5'))

    def returnChildrenGenetic(self, sortedParentIdxs):
        """Mutate best parents, keep elite child and save them to disk
        individually.
        """

        if self.rereturnChildrenGenetic:
            generation = self.geneticGeneration
            tempModelPrefixBefore = self.tempModelPrefix
            tempNextModelPrefixBefore = self.tempNextModelPrefix
            self.tempModelPrefix = self.tempPrevModelPrefix
            self.tempNextModelPrefix = tempModelPrefixBefore
            self.rewards = self.pickleLoad(join(self.tempModelPrefix +
                '_agentsRewardsMean.p'))
        elif not self.rereturnChildrenGenetic:
            # self.tempModelPrefix = self.tempModelPrefix
            generation = self.geneticGeneration + 1

        tempNextModelPrefix = join(self.tempModelpth,
            self.envSettings['MODELNAME'] + '_gen' +
            str(generation+1).zfill(self.zFill))

        if self.noveltySearch:
            recalculateNovelties = False
            try:
                self.novelties
            except Exception as e:
                recalculateNovelties = True
            if len(self.novelties) == 0:
                recalculateNovelties = True

            if recalculateNovelties:
                self.novelties, self.noveltyFilenames = [], []
                for k in range(self.noveltyItemCount):
                    agentStr = 'agent' + str(k+1)
                    self.novelties.append(
                        self.noveltyArchive[agentStr]['novelty'])
                    self.noveltyFilenames.append(
                        self.noveltyArchive[agentStr]['modelFile'])
            self.candidateNoveltyParentIdxs = argsort(
                self.novelties)[::-1][:self.hyParams['NNOVELTYELITES']]

        bestAgent = load_model(join(self.tempModelPrefix + '_agent' +
            str(sortedParentIdxs[0] + 1).zfill(self.zFill) + '.h5'),
            compile=False)

        if not self.rereturnChildrenGenetic:
            bestAgent.save(join(self.tempModelPrefix + '_agentBest.h5'))
        if generation < self.hyParams['NGENERATIONS']:
            bestAgent.save(join(tempNextModelPrefix + '_agent' +
                str(self.hyParams['NAGENTS']).zfill(self.zFill) + '.h5'))
            nAgentElites = self.hyParams['NAGENTELITES']
            nNoveltyAgents = self.hyParams['NNOVELTYELITES']
            self.candidateParentIdxs = sortedParentIdxs[:nAgentElites]
            chunksTotal = self.yieldChunks(arange(self.hyParams['NAGENTS']-1),
                self.envSettings['NAGENTSPARALLEL']*self.maxTasksPerWorkerMutate)
            for chunk in chunksTotal:
                _ = self.multiprocessChunks(self.returnChildrenGeneticSingleRun,
                    chunk)

        if self.rereturnChildrenGenetic:
            # resetting temporarily changed prefixes
            self.tempModelPrefix = tempModelPrefixBefore
            self.tempNextModelPrefix = tempNextModelPrefixBefore

    def returnChildrenGeneticSingleRun(self, childIdx):
        """
        """

        len_ = len(self.candidateParentIdxs)
        selected_agent_index = self.candidateParentIdxs[randint(len_)]
        agentPth = join(self.tempModelPrefix + '_agent' +
            str(selected_agent_index + 1).zfill(self.zFill) + '.h5')

        if self.noveltySearch:
            if ((self.geneticGeneration+1) % self.hyParams['ADDNOVELTYEVERY']) == 0:
                remainingElites = self.hyParams['NAGENTS'] - (childIdx+1)
                if self.rereturnChildrenGenetic:
                    generation = self.geneticGeneration
                else:
                    generation = self.geneticGeneration + 1
                if (childIdx+1 ==
                    remainingElites - self.hyParams['NNOVELTYELITES']):

                    print('Performing novelty evolution after generation',
                        generation)
                if remainingElites <= self.hyParams['NNOVELTYELITES']:
                    # selecting a novelty parent randomly, might skip most novel
                    # len_ = len(self.candidateNoveltyParentIdxs)
                    # selected_agent_index = self.candidateNoveltyParentIdxs[randint(len_)]
                    # selecting each novelty parent individually
                    selected_agent_index = self.candidateNoveltyParentIdxs[int(
                        remainingElites)-1]
                    agentPth = self.noveltyFilenames[selected_agent_index]

        # loading given parent agent, current with retries in case of race
        # condition: https://bugs.python.org/issue36773
        success = False
        while not success:
            try:
                agent = load_model(agentPth,
                    compile=False)
                success = True
            except Exception as e:
                success = False
                print('Retrying loading parent agent, possibly due to lock.')
                print(e)
                sleep(1)
        # altering parent agent to create child agent
        childrenAgent = self.mutateGenetic(agent)
        childrenAgent.save(join(self.tempNextModelPrefix +
            '_agent' + str(childIdx + 1).zfill(self.zFill) + '.h5'))

        # print('debug loading', agentPth)
        # print('debug saving', join(self.tempNextModelPrefix +
        #     '_agent' + str(childIdx + 1).zfill(self.zFill) + '.h5'))

    def mutateGenetic(self, agent):
        """Mutate single agent model.
        Mutation power is a hyperparameter. Find example values at:
        https://arxiv.org/pdf/1712.06567.pdf
        """

        mProb = self.hyParams['MUTATIONPROBABILITY']
        mPower = self.hyParams['MUTATIONPOWER']
        weights, paramIdx = agent.get_weights(), 0
        # mutating weights and biases
        for parameters in weights:
            if self.mutateDecision(mProb):
                weights[paramIdx] = add(parameters, mPower * randn())
            paramIdx += 1
        agent.set_weights(weights)

        return agent

    def mutateDecision(self, probability):
        """Return boolean defining whether to mutate or not."""
        return random() < probability

    def getqsGivenAgentModel(self, agentModel, state):
        """ Query given model for Q values given observations of state
        """
        # predict_on_batch robust in parallel operation?

        return agentModel.predict_on_batch(
            array(state).reshape(-1, (*shape(state))))[0]

    def loadAgentModel(self, modelNameLoad=None, compiled=False):
        """Load an agent model."""

        # Note: change model load and save as json mode is faster
        # this is a temporary quickfix

        # self.modelpth
        modelPrefix = join(self.modelpth, modelNameLoad)
        if exists(modelPrefix + '.json'):
            with open(modelPrefix + '.json') as json_file:
                json_config = json_file.read()
            agentModel = model_from_json(json_config)
            agentModel.load_weights(modelPrefix + 'Weights.h5')
        if not exists(modelPrefix + '.json'):
            try: agentModel = load_model(modelPrefix + '.model', compile=compiled)
            except: agentModel = load_model(modelPrefix + '.h5', compile=compiled)
            json_config = agentModel.to_json()
            with open(modelPrefix + '.json', 'w') as json_file:
                json_file.write(json_config)
            agentModel.save_weights(modelPrefix + 'Weights.h5')

        return agentModel

    def getAction(self, mode='random', keyPressed=None, agent=None,
            modelNameLoad=None, state=None, actionType='discrete'):
        """Determine an action given an agent model.
        Either the action is determined from the player pressing a button or
        chosen randomly. If the player does not press a key within the given
        timeframe for a game, the action remains unchanged.
        Note: mode 'modelNameLoad' can massively throttleneck in loops from
        recurring model loading overhead.
        """

        if actionType == 'discrete':
            if mode == 'manual':
                if keyPressed in self.actionSpace:
                    self.action = keyPressed
                else:
                    self.action = 'keep'
            if mode == 'random':
                actionIdx = randint(0, high=len(self.actionSpace))
                self.action = self.actionSpace[actionIdx]
            if mode == 'modelNameLoad':
                agentModel = self.loadAgentModel(modelNameLoad)
                actionIdx = argmax(self.getqsGivenAgentModel(agentModel, state))
                self.action = self.actionSpace[actionIdx]
            if mode == 'model':
                actionIdx = argmax(self.getqsGivenAgentModel(agent, state))
                self.action = self.actionSpace[actionIdx]

        elif actionType == 'continuous':
            if mode == 'manual':
                print('Regression mode is currently not supported in manual control.')
                quit()
            if mode == 'random':
                self.action = []
                for i in range(self.actionSpaceSize):
                    actionVal = uniform(low=0.0, high=1.0)
                    self.action.append(actionVal)
            if mode == 'modelNameLoad':
                agentModel = self.loadAgentModel(modelNameLoad)
                self.action = self.getqsGivenAgentModel(agentModel, state)
            if mode == 'model':
                self.action = self.getqsGivenAgentModel(agent, state)

        return self.action

    def resumeGenetic(self):
        # checking if bestModel already exists for current generation
        # skipping calculations then to to resume at a later stage
        continueFlag, breakFlag = False, False
        bestAgentpth = join(self.tempModelPrefix + '_agentBest.h5')
        if exists(bestAgentpth):
            indexespth = join(self.tempModelPrefix +
                '_agentsSortedParentIndexes.p')
            noveltyArchivepth = join(self.tempModelPrefix +
                '_noveltyArchive.p')
            self.sortedParentIdxs = self.pickleLoad(indexespth)
            self.noveltyArchive = self.pickleLoad(noveltyArchivepth)
            self.flagSkipGeneration, continueFlag = True, True

            self.novelties, self.noveltyFilenames = [], []
            for k in range(self.noveltyItemCount):
                agentStr = 'agent' + str(k+1)
                self.novelties.append(
                    self.noveltyArchive[agentStr]['novelty'])
                self.noveltyFilenames.append(
                    self.noveltyArchive[agentStr]['modelFile'])

        # regenerating children for generation to resume at
        else:
            if self.envSettings['KEEPMODELHISTORY']:
                with Pool(1) as executor:
                    self.rereturnChildrenGenetic = True
                    self.returnChildrenGenetic(self.sortedParentIdxs)
                    self.rereturnChildrenGenetic = False
            elif not self.envSettings['KEEPMODELHISTORY']:
                print('Resuming impossible with missing model history.')
                breakFlag = True
            # changing resume flag if resuming
            self.envSettings['RESUME'] = False

        return self.sortedParentIdxs, continueFlag, breakFlag

    def calculateNoveltyPerPair(self, args):
        agentStr = 'agent' + str(args[0]+1)
        agentStr2 = 'agent' + str(args[1]+1)

        tempAgentPrefix = self.noveltyArchive[agentStr]['modelFile'].replace('.h5', '')
        tempAgentPrefix2 = self.noveltyArchive[agentStr2]['modelFile'].replace('.h5', '')
        pth = join(tempAgentPrefix + '_results.p')
        pth2 = join(tempAgentPrefix2 + '_results.p')
        actions = self.pickleLoad(pth)['actions']
        actions2 = self.pickleLoad(pth2)['actions']

        # actions = self.noveltyArchive[agentStr]['actions']
        # actions2 = self.noveltyArchive[agentStr2]['actions']

        novelty = 0.
        for g in range(len(actions)):
            novelty += self.actionNoveltyMetric(actions[g],
                actions2[g])

        return novelty

    def calculateNoveltyPerAgent(self, iAgent):
        agentStr = 'agent' + str(iAgent+1)
        # loading novelties for specific agent from disk

        if not exists(join(self.tempNoveltypth, agentStr + '_novelties.p')):
            agentNovelties = {}
        else:
            with open(join(self.tempNoveltypth, agentStr + '_novelties.p'), 'rb') as f:
                agentNovelties = self.pickleLoad(f, compressed='lz4')

        novelties = []
        if self.noveltyItemCount <= self.hyParams['NNOVELTYNEIGHBORS']:
            # neighborLimitReached = False
            for iAgent2 in range(self.noveltyItemCount):
                if iAgent != iAgent2:
                    try:
                        # calculate novelties only if unavailable
                        # as keys mostly exists, try/except check should be performant here
                        novelty = agentNovelties[str(iAgent+1) + '_' + str(iAgent2+1)]
                    except:
                        novelty = self.calculateNoveltyPerPair([iAgent, iAgent2])
                        agentNovelties[str(iAgent+1) + '_' + str(iAgent2+1)] = novelty
                    novelties.append(novelty)

        elif self.noveltyItemCount > self.hyParams['NNOVELTYNEIGHBORS']:
            # neighborLimitReached = True

            # checking if half of NNOVELTYNEIGHBORS are available surrounding the given index
            # if not agents are selected until the index boundary and more from the other end
            nLower = int(floor(self.hyParams['NNOVELTYNEIGHBORS']/2))
            nHigher = int(ceil(self.hyParams['NNOVELTYNEIGHBORS']/2))
            bottomReached = False
            if iAgent - nLower >= 0:
                rangeLower = iAgent - nLower
                rangeHigher = iAgent + nHigher
            else:
                rangeLower, rangeHigher = 0, self.hyParams['NNOVELTYNEIGHBORS']
                bottomReached = True

            if not bottomReached:
                if iAgent + nHigher < self.noveltyItemCount:
                    rangeLower = iAgent - nLower
                    rangeHigher = iAgent + nHigher
                else:
                    rangeLower = self.noveltyItemCount - self.hyParams['NNOVELTYNEIGHBORS']
                    rangeHigher = self.noveltyItemCount

            for iAgent2 in range(rangeLower, rangeHigher):
                if iAgent != iAgent2:
                    try:
                        # calculate novelties only if unavailable
                        # as keys mostly exists, try/except check should be performant here
                        novelty = agentNovelties[str(iAgent+1) + '_' + str(iAgent2+1)]
                    except:
                        novelty = self.calculateNoveltyPerPair([iAgent, iAgent2])
                        agentNovelties[str(iAgent+1) + '_' + str(iAgent2+1)] = novelty
                    novelties.append(novelty)

        self.pickleDump(join(self.tempNoveltypth,
            agentStr + '_novelties.p'), agentNovelties, compress='lz4')

        novelty = mean(novelties)

        return novelty

    def actionNoveltyMetric(self, actions1, actions2):
        # finding largest object, or determining equal length
        actions1 = array(actions1)
        actions2 = array(actions2)

        if self.env.actionType == 'discrete':
            if len(actions1) > len(actions2):
                shorterObj = actions2
                longerObj = actions1
            if len(actions1) <= len(actions2):
                shorterObj = actions1
                longerObj = actions2
            diffsCount = len(shorterObj) - sum(array(shorterObj) == array(longerObj[:len(shorterObj)]))

            # enabling this might promote agents having acted longer but not
            # too different to begin with
            # diffsCount += float(abs(len(longerObj) - len(shorterObj)))

            # dividing by the length of it, to avoid rewarding longer objects
            novelty = diffsCount/len(shorterObj)

        elif self.env.actionType == 'continuous':
            actions1 = array(actions1)
            actions2 = array(actions2)

            # there are values per action in the action space
            ln = min([len(actions1[:,0]), len(actions2[:,0])])
            nActionLists = shape(actions2)[1]

            diffsCount = ln*nActionLists
            diffs = sum(abs(subtract(
                actions1[:ln,:].flatten(),
                actions2[:ln,:].flatten()
                )))
            novelty = diffs/diffsCount

        return novelty

    def saveBestAgent(self, MODELNAME):
        # saving best agent of the current generation
        bestAgent = load_model(join(self.tempModelPrefix + '_agentBest.h5'),
                compile=False)
        self.bestAgentFileName = (f'{MODELNAME}' + '_gen' +
            str(self.geneticGeneration+1).zfill(self.zFill) + '_avg' +
            f'{self.bestAgentReward:_>7.1f}')
        bestAgent.save(join(self.modelpth, self.bestAgentFileName + '.h5'))

    def saveBestAgentAnimation(self, env, bestAgentFileName, MODELNAMEGENCOUNT,
        MODELNAME):
        # playing a game with best agent to visualize progress
        game = FloPyArcade(modelNameLoad=bestAgentFileName,
            modelName=MODELNAMEGENCOUNT,
            animationFolder=MODELNAME,
            NAGENTSTEPS=self.hyParams['NAGENTSTEPS'],
            PATHMF2005=self.envSettings['PATHMF2005'],
            PATHMP6=self.envSettings['PATHMP6'],
            flagSavePlot=True, flagManualControl=False,
            flagRender=False,
            nLay=env.nLay, nRow=env.nRow, nCol=env.nCol)
        game.play(
            ENVTYPE=self.envSettings['ENVTYPE'],
            seed=self.envSettings['SEEDENV'] + self.currentGame-1)

    def generatePathPrefixes(self):
        self.tempModelPrefix = join(self.tempModelpth,
            self.envSettings['MODELNAME'] + '_gen' +
            str(self.geneticGeneration + 1).zfill(self.zFill))
        self.tempNextModelPrefix = join(self.tempModelpth,
            self.envSettings['MODELNAME'] + '_gen' +
            str(self.geneticGeneration + 2).zfill(self.zFill))
        self.tempPrevModelPrefix = join(self.tempModelpth,
            self.envSettings['MODELNAME'] + '_gen' +
            str(self.geneticGeneration).zfill(self.zFill))

    def yieldChunks(self, lst, n):
        """Yield successive n-sized chunks from a given list.
        Taken from: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def multiprocessChunks(self, function, chunk, parallelProcesses=None, wait=False):
        """Process function in parallel given a chunk of arguments."""

        # Pool object from pathos instead of multiprocessing library necessary
        # as tensor.keras models are currently not pickleable
        # https://github.com/tensorflow/tensorflow/issues/32159
        if parallelProcesses == None:
            parallelProcesses = self.envSettings['NAGENTSPARALLEL']
        p = Pool(processes=parallelProcesses)
        pasync = p.map_async(function, chunk)
        # waiting is important to order results correctly when running
        # -- really?
        # in asynchronous mode (correct reward order is validated)
        pasync = pasync.get()
        if wait:
            pasync.wait()
        p.close()
        p.join()
        p.terminate()

        return pasync

    def pickleLoad(self, path, compressed=None):
        """Load pickled object from file."""
        if compressed == None:
            filehandler = open(path, 'rb')
            objectLoaded = load(filehandler)
            filehandler.close()
        elif compressed in ['lz4', 'lzma']:
            objectLoaded = joblibLoad(path)
        else:
            print('Unknown compression algorithm specified.')

        return objectLoaded

    def pickleDump(self, path, objectToDump, compress=None, compressLevel=3):
        """Store object to file using pickle."""
        if compress == None:
            filehandler = open(path, 'wb')
            dump(objectToDump, filehandler)
            filehandler.close()
        elif compress == 'lz4':
            with open(path, 'wb') as f:
                joblibDump(objectToDump, f, compress=('lz4', compressLevel))
        elif compress == 'lzma':
            with open(path, 'wb') as f:
                joblibDump(objectToDump, f, compress=('lzma', compressLevel))
        else:
            print('Unknown compression algorithm specified.')

    def GPUAllowMemoryGrowth(self):
        """Allow GPU memory to grow to enable parallelism on a GPU."""
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        sess = TFSession(config=config)
        K.set_session(sess)

    def suppressTensorFlowWarnings(self):
        # suppressing TensorFlow output on import, except fatal errors
        # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
        from logging import getLogger, FATAL
        environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        getLogger('tensorflow').setLevel(FATAL)

class FloPyEnv():
    """Environment to perform forward simulation using MODFLOW and MODPATH.
    On first call, initializes a model with a randomly-placed operating well,
    initializes the corresponding steady-state flow solution as a starting state
    and initializes a random starting action and a random particle on the
    Western side.
    On calling a step, it loads the current state, tracks the particle's
    trajectory through the model domain and returns the environment's new state,
    the new particle location as an observation and a flag if the particle has
    reached the operating well or not as a state.
    """

    def __init__(self,
                 ENVTYPE='1s-d', PATHMF2005=None, PATHMP6=None,
                 MODELNAME='FloPyArcade', ANIMATIONFOLDER='FloPyArcade',
                 _seed=None, flagSavePlot=False, flagManualControl=False,
                 manualControlTime=0.1, flagRender=False, NAGENTSTEPS=None,
                 nLay=1, nRow=100, nCol=100, OBSPREP='vector',
                 initWithSolution=True):
        """Constructor."""

        self.ENVTYPE = ENVTYPE
        self.PATHMF2005, self.PATHMP6 = PATHMF2005, PATHMP6
        self.MODELNAME = 'FloPyArcade' if (MODELNAME==None) else MODELNAME
        self.ANIMATIONFOLDER = ANIMATIONFOLDER
        self.SAVEPLOT = flagSavePlot
        self.MANUALCONTROL = flagManualControl
        self.MANUALCONTROLTIME = manualControlTime
        self.RENDER = flagRender
        self.NAGENTSTEPS = NAGENTSTEPS
        self.info, self.comments = '', ''
        self.done = False
        self.nLay, self.nRow, self.nCol = nLay, nRow, nCol
        self.OBSPREP = OBSPREP
        self.initWithSolution = initWithSolution

        self.wrkspc = dirname(abspath(__file__))
        if 'library.zip' in self.wrkspc:
            # changing workspace in case of call from executable
            self.wrkspc = dirname(dirname(self.wrkspc))
        # setting up the model path and ensuring it exists
        self.modelpth = join(self.wrkspc, 'models', self.MODELNAME)
        if not exists(self.modelpth):
            makedirs(self.modelpth)
        self.actionType = self.getActionType(self.ENVTYPE)

        self._SEED = _seed
        if self._SEED is not None:
            numpySeed(self._SEED)
        self.defineEnvironment()
        self.timeStep, self.keyPressed = 0, None
        self.reward, self.rewardCurrent = 0., 0.

        self.initializeSimulators(PATHMF2005, PATHMP6)
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '2s-d', '2s-c', '2r-d', '2r-c']:
            self.initializeAction()
        self.initializeParticle()

        # this needs to be transformed, yet not understood why
        self.particleCoords[0] = self.extentX - self.particleCoords[0]

        if self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            self.headSpecNorth = uniform(self.minH, self.maxH)
            self.headSpecSouth = uniform(self.minH, self.maxH)
        self.initializeModel()

        self.wellX, self.wellY, self.wellZ, self.wellCoords, self.wellQ = self.initializeWellRate(self.minQ, self.maxQ)
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            helperWells = {}
            for i in range(self.nHelperWells):
                w = str(i+1)
                helperWells['wellX'+w], helperWells['wellY'+w], helperWells['wellZ'+w], helperWells['wellCoords'+w], helperWells['wellQ'+w] = self.initializeWellRate(self.minQhelper, self.maxQhelper)
            self.helperWells = helperWells

        self.initializeWell()
        if self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            self.initializeAction()
        # initializing trajectories container for potential plotting
        self.trajectories = {}
        for i in ['x', 'y', 'z']:
            self.trajectories[i] = []

        if self.initWithSolution:
            self.stepInitial()

    def stepInitial(self):
        """Initialize with the steady-state solution.
        
        Note: If just initializing environments without intention of solving,
        or intentions of solving later, this can be a massive throttleneck.
        """

        # running MODFLOW to determine steady-state solution as a initial state
        self.runMODFLOW()

        self.state = {}
        self.state['heads'] = copy(self.heads)
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self.state['actionValueNorth'] = self.actionValueNorth
            self.state['actionValueSouth'] = self.actionValueSouth
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            self.state['actionValue'] = self.actionValue
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c']:
            self.state['actionValueX'] = self.actionValueX
            self.state['actionValueY'] = self.actionValueY
        elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.state['actionValueX'+w] = self.helperWells['actionValueX'+w]
                self.state['actionValueY'+w] = self.helperWells['actionValueX'+w]
        elif self.ENVTYPE in ['5s-d', '5s-c', '5r-d', '5r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.state['actionValueQ'+w] = self.helperWells['actionValueQ'+w]
        elif self.ENVTYPE in ['6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.state['actionValueX'+w] = self.helperWells['actionValueX'+w]
                self.state['actionValueY'+w] = self.helperWells['actionValueX'+w]
                self.state['actionValueQ'+w] = self.helperWells['actionValueQ'+w]

        self.observations = {}
        self.observationsNormalized, self.observationsNormalizedHeads = {}, {}
        self.observations['particleCoords'] = self.particleCoords
        if self.OBSPREP != 'convolutional':
            self.observations['headsSampledField'] = self.heads[0::self.sampleHeadsEvery,
                                                    0::self.sampleHeadsEvery,
                                                    0::self.sampleHeadsEvery]
        elif self.OBSPREP == 'convolutional':
            self.observations['heads'] = self.heads
        lParticle, cParticle, rParticle = self.cellInfoFromCoordinates(
            [self.particleCoords[0], self.particleCoords[1], self.particleCoords[2]])
        lWell, cWell, rWell = self.cellInfoFromCoordinates(
            [self.wellX, self.wellY, self.wellZ])

        # note: these heads from actions are not necessary to return as observations for surrogate modeling
        # but for reinforcement learning
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self.observations['heads'] = [self.actionValueNorth,
                                          self.actionValueSouth]
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            # this can cause issues with unit testing, as model expects different input 
            self.observations['heads'] = [self.actionValue]
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            self.observations['heads'] = [self.headSpecNorth,
                                          self.headSpecSouth]
        # note: it sees the surrounding heads of the particle and the well
        # self.observations['heads'] += [self.heads[lParticle-1, rParticle-1, cParticle-1]]
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.particleCoords, distance=0.5*self.wellRadius)
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.particleCoords, distance=1.5*self.wellRadius)
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.particleCoords, distance=2.5*self.wellRadius)
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.wellCoords, distance=1.5*self.wellRadius)
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.wellCoords, distance=2.0*self.wellRadius)
        self.observations['heads'] += list(array(self.observations['headsSampledField']).flatten())
        self.observations['wellQ'] = self.wellQ
        self.observations['wellCoords'] = self.wellCoords
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.observations['wellQ'+w] = self.helperWells['wellQ'+w]
                self.observations['wellCoords'+w] = self.helperWells['wellCoords'+w]
        self.observationsNormalized['particleCoords'] = divide(
            copy(self.particleCoords), self.minX + self.extentX)
        self.observationsNormalized['heads'] = divide(array(self.observations['heads']) - self.minH,
            self.maxH - self.minH)
        # self.observationsNormalized['heads'] = divide(array(self.observations['heads']) - self.minH,
        #     self.maxH - self.minH)
        self.observationsNormalized['wellQ'] = self.wellQ / self.minQ
        self.observationsNormalized['wellCoords'] = divide(
            self.wellCoords, self.minX + self.extentX)
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.observationsNormalized['wellQ'+w] = self.helperWells['wellQ'+w] / self.minQhelper
                self.observationsNormalized['wellCoords'+w] = divide(
                    self.helperWells['wellCoords'+w], self.minX + self.extentX)
        self.observationsNormalizedHeads['heads'] = divide(array(self.heads) - self.minH,
            self.maxH - self.minH)

        if self.OBSPREP != 'convolutional':
            self.observationsVector = self.observationsDictToVector(
                self.observations)
            self.observationsVectorNormalized = self.observationsDictToVector(
                self.observationsNormalized)
            self.observationsVectorNormalizedHeads = self.observationsDictToVector(
                self.observationsNormalizedHeads)
        elif self.OBSPREP != 'convolutional':
            self.observationsVector = self.observationsDictToVector(
                self.observations)
            self.observationsVectorNormalized = self.observationsDictToVector(
                self.observationsNormalized)

        # alternatively normalize well z coordinate to vertical extent
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self.stressesVectorNormalized = [(self.actionValueSouth - self.minH)/(self.maxH - self.minH),
                                             (self.actionValueNorth - self.minH)/(self.maxH - self.minH),
                                             self.wellQ/self.minQ, self.wellX/(self.minX+self.extentX),
                                             self.wellY/(self.minY+self.extentY), self.wellZ/(self.zBot+self.zTop)]
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            self.stressesVectorNormalized = [(self.actionValue - self.minH)/(self.maxH - self.minH),
                                             self.wellQ/self.minQ, self.wellX/(self.minX+self.extentX),
                                             self.wellY/(self.minY+self.extentY), self.wellZ/(self.zBot+self.zTop)]
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c']:
            self.stressesVectorNormalized = [(self.headSpecSouth - self.minH)/(self.maxH - self.minH),
                                             (self.headSpecNorth - self.minH)/(self.maxH - self.minH),
                                             self.wellQ/self.minQ, self.wellX/(self.minX+self.extentX),
                                             self.wellY/(self.minY+self.extentY), self.wellZ/(self.zBot+self.zTop)]
        elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            self.stressesVectorNormalized = [(self.headSpecSouth - self.minH)/(self.maxH - self.minH),
                                             (self.headSpecNorth - self.minH)/(self.maxH - self.minH)]
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.stressesVectorNormalized += [self.helperWells['wellQ'+w]/self.minQ, self.helperWells['wellX'+w]/(self.minX+self.extentX),
                    self.helperWells['wellY'+w]/(self.minY+self.extentY), self.helperWells['wellZ'+w]/(self.zBot+self.zTop)]

        self.timeStepDuration = []

    def step(self, observations, action, rewardCurrent):
        """Perform a single step of forwards simulation."""

        if self.timeStep == 0:
            if not self.initWithSolution:
                self.stepInitial()
            # rendering initial timestep
            if self.RENDER or self.MANUALCONTROL or self.SAVEPLOT:
                self.render()
        self.timeStep += 1
        self.keyPressed = None
        self.periodSteadiness = False
        t0total = time()

        # print('debug action step', action)
        self.setActionValue(action)
        observations = self.observationsVectorToDict(observations)
        self.particleCoordsBefore = observations['particleCoords']

        # it might be obsolete to feed this back,
        # as it can be stored with the object
        self.rewardCurrent = rewardCurrent

        # does this need to be enabled? It disables numpy finding different
        # random numbers throughout the game,
        # for example for random action exploration
        # if self._SEED is not None:
        #     numpySeed(self._SEED)

        self.initializeState(self.state)
        self.updateModel()
        self.updateWellRate()
        self.updateWell()
        self.runMODFLOW()
        self.runMODPATH()
        self.evaluateParticleTracking()

        # calculating game reward
        self.reward = self.calculateGameReward(self.trajectories) 

        self.state = {}
        self.state['heads'] = self.heads
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self.state['actionValueNorth'] = self.actionValueNorth
            self.state['actionValueSouth'] = self.actionValueSouth
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            self.state['actionValue'] = self.actionValue
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c']:
            self.state['actionValueX'] = self.actionValueX
            self.state['actionValueY'] = self.actionValueY
        elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.state['actionValueX'+w] = self.helperWells['actionValueX'+w]
                self.state['actionValueY'+w] = self.helperWells['actionValueY'+w]
        elif self.ENVTYPE in ['5s-d', '5s-c', '5r-d', '5r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.state['actionValueQ'+w] = self.helperWells['actionValueQ'+w]
        elif self.ENVTYPE in ['6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.state['actionValueX'+w] = self.helperWells['actionValueX'+w]
                self.state['actionValueY'+w] = self.helperWells['actionValueY'+w]
                self.state['actionValueQ'+w] = self.helperWells['actionValueQ'+w]

        self.observations = {}
        self.observationsNormalized, self.observationsNormalizedHeads = {}, {}
        self.observations['particleCoords'] = self.particleCoords
        if self.OBSPREP != 'convolutional':
            self.observations['headsSampledField'] = self.heads[0::self.sampleHeadsEvery,
                                                    0::self.sampleHeadsEvery,
                                                    0::self.sampleHeadsEvery]
        elif self.OBSPREP == 'convolutional':
            self.observations['headsSampledField'] = self.heads
        lParticle, cParticle, rParticle = self.cellInfoFromCoordinates(
            [self.particleCoords[0], self.particleCoords[1], self.particleCoords[2]])
        lWell, cWell, rWell = self.cellInfoFromCoordinates(
            [self.wellX, self.wellY, self.wellZ])
        # note: these heads from actions are not necessary to return as observations for surrogate modeling
        # but for reinforcement learning
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self.observations['heads'] = [self.actionValueNorth,
                                          self.actionValueSouth]
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            # this can cause issues with unit testing, as model expects different input 
            self.observations['heads'] = [self.actionValue]
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            self.observations['heads'] = [self.headSpecNorth,
                                          self.headSpecSouth]
        # note: it sees the surrounding heads of the particle and the well
        # self.observations['heads'] += [self.heads[lParticle-1, rParticle-1, cParticle-1]]
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.particleCoords, distance=0.5*self.wellRadius)
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.particleCoords, distance=1.5*self.wellRadius)
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.particleCoords, distance=2.5*self.wellRadius)
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.wellCoords, distance=1.5*self.wellRadius)
        # self.observations['heads'] += self.surroundingHeadsFromCoordinates(self.wellCoords, distance=2.0*self.wellRadius)
        self.observations['heads'] += list(array(self.observations['headsSampledField']).flatten())
        self.observations['wellQ'] = self.wellQ
        self.observations['wellCoords'] = self.wellCoords
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.observations['wellQ'+w] = self.helperWells['wellQ'+w]
                self.observations['wellCoords'+w] = self.helperWells['wellCoords'+w]
        self.observationsNormalized['particleCoords'] = divide(
            copy(self.particleCoordsAfter), self.minX + self.extentX)
        self.observationsNormalized['heads'] = divide(array(self.observations['heads']) - self.minH,
            self.maxH - self.minH)
        # self.observationsNormalized['heads'] = divide(array(self.observations['heads']) - self.minH,
        #     self.maxH - self.minH)
        self.observationsNormalized['wellQ'] = self.wellQ / self.minQ
        self.observationsNormalized['wellCoords'] = divide(
            self.wellCoords, self.minX + self.extentX)
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.observationsNormalized['wellQ'+w] = self.helperWells['wellQ'+w] / self.minQhelper
                self.observationsNormalized['wellCoords'+w] = divide(
                    self.helperWells['wellCoords'+w], self.minX + self.extentX)
        self.observationsNormalizedHeads['heads'] = divide(array(self.heads) - self.minH,
            self.maxH - self.minH)

        if self.OBSPREP != 'convolutional':
            self.observationsVector = self.observationsDictToVector(
                self.observations)
            self.observationsVectorNormalized = self.observationsDictToVector(
                self.observationsNormalized)
            self.observationsVectorNormalizedHeads = self.observationsDictToVector(
                self.observationsNormalizedHeads)
        elif self.OBSPREP == 'convolutional':
            self.observationsVector = self.observationsDictToVector(
                self.observations)
            self.observationsVectorNormalized = self.observationsDictToVector(
                self.observationsNormalized)

        if self.observations['particleCoords'][0] >= self.extentX - self.dCol:
            self.success = True
        else:
            self.success = False

        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self.stressesVectorNormalized = [(self.actionValueSouth - self.minH)/(self.maxH - self.minH),
                                             (self.actionValueNorth - self.minH)/(self.maxH - self.minH),
                                             self.wellQ/self.minQ, self.wellX/(self.minX+self.extentX),
                                             self.wellY/(self.minY+self.extentY), self.wellZ/(self.zBot+self.zTop)]
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            self.stressesVectorNormalized = [(self.actionValue - self.minH)/(self.maxH - self.minH),
                                             self.wellQ/self.minQ, self.wellX/(self.minX+self.extentX),
                                             self.wellY/(self.minY+self.extentY), self.wellZ/(self.zBot+self.zTop)]
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c']:
            self.stressesVectorNormalized = [(self.headSpecSouth - self.minH)/(self.maxH - self.minH),
                                             (self.headSpecNorth - self.minH)/(self.maxH - self.minH),
                                             self.wellQ/self.minQ, self.wellX/(self.minX+self.extentX),
                                             self.wellY/(self.minY+self.extentY), self.wellZ/(self.zBot+self.zTop)]
        elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            self.stressesVectorNormalized = [(self.headSpecSouth - self.minH)/(self.maxH - self.minH),
                                             (self.headSpecNorth - self.minH)/(self.maxH - self.minH)]
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.stressesVectorNormalized += [self.helperWells['wellQ'+w]/self.minQ, self.helperWells['wellX'+w]/(self.minX+self.extentX),
                    self.helperWells['wellY'+w]/(self.minY+self.extentY), self.helperWells['wellZ'+w]/(self.zBot+self.zTop)]

        # checking if particle is within horizontal distance of well
        dx = self.particleCoords[0] - self.wellCoords[0]
        # why would the correction for Y coordinate be necessary
        dy = self.extentY - self.particleCoords[1] - self.wellCoords[1]
        self.distanceWellParticle = sqrt(dx**2 + dy**2)
        if self.distanceWellParticle <= self.wellRadius:
            self.done = True
            self.reward = (self.rewardCurrent) * (-1.0)
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            coords = []
            for i in range(self.nHelperWells):
                w = str(i+1)
                coords.append(self.helperWells['wellCoords'+w])
            for c in coords:
                dx = self.particleCoords[0] - c[0]
                # why would the correction for Y coordinate be necessary
                dy = self.extentY - self.particleCoords[1] - c[1]
                self.distanceWellParticle = sqrt(dx**2 + dy**2)
                if self.distanceWellParticle <= self.helperWellRadius:
                    self.done = True
                    self.reward = (self.rewardCurrent) * (-1.0)

        # checking if particle has reached eastern boundary
        if self.particleCoordsAfter[0] >= self.minX + self.extentX - self.dCol:
            self.done = True

        # checking if particle has returned to western boundary
        if self.particleCoordsAfter[0] <= self.minX + self.dCol:
            self.done = True
            self.reward = (self.rewardCurrent) * (-1.0)

        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
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
        self.timeStepDuration.append(time() - t0total)

        if self.RENDER or self.MANUALCONTROL or self.SAVEPLOT:
            self.render()

        if self.done:
            # print('debug average timeStepDuration', mean(self.timeStepDuration))

            # necessary to remove these file handles to release file locks
            del self.mf, self.cbb, self.hdobj

            for f in listdir(self.modelpth):
                # removing files in folder
                remove(join(self.modelpth, f))
            if exists(self.modelpth):
                # removing folder with model files after run
                rmdir(self.modelpth)

        return self.observations, self.reward, self.done, self.info

    def defineEnvironment(self):
        """Define environmental variables."""

        # general environment settings,
        # like model domain and grid definition
        # uses SI units for length and time
        # currently fails with arbitray model extents?
        # why is a periodLength of 2.0 necessary to simulate 1 day?
        self.minX, self.minY = 0., 0.
        self.extentX, self.extentY = 100., 100.
        self.zBot, self.zTop = 0., 50.
        # previously self.nRow, self.nCol = 100, 100, for comparison check /dev/discretizationHeadDependence.txt
        # self.nLay, self.nRow, self.nCol = 1, 800, 800
        self.headSpecWest, self.headSpecEast = 60.0, 56.0
        self.minQ = -2000.0
        self.maxQ = -500.0
        self.wellSpawnBufferXWest, self.wellSpawnBufferXEast = 50.0, 20.0
        self.wellSpawnBufferY = 20.0
        self.periods, self.periodLength, self.periodSteps = 1, 1.0, 11
        self.periodSteadiness = True
        self.maxSteps = self.NAGENTSTEPS
        self.sampleHeadsEvery = 5

        self.dRow = self.extentX / self.nCol
        self.dCol = self.extentY / self.nRow
        self.dVer = (self.zTop - self.zBot) / self.nLay
        self.botM = linspace(self.zTop, self.zBot, self.nLay + 1)

        self.wellRadius = sqrt((2 * 1.)**2 + (2 * 1.)**2)
        # print('debug wellRadius', self.wellRadius)

        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self.minH = 56.0
            self.maxH = 60.0
            self.nHelperWells = 0
            self.deviationPenaltyFactor = 10.0
            self.actionRange = 0.5
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            self.minH = 56.0
            self.maxH = 62.0
            self.nHelperWells = 0
            self.deviationPenaltyFactor = 4.0
            self.actionRange = 0.5
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c']:
            self.minH = 56.0
            self.maxH = 60.0
            self.nHelperWells = 0
            self.deviationPenaltyFactor = 10.0
            self.actionRange = 10.0
        elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            self.helperWellRadius = self.wellRadius/4
            self.minH = 56.0
            self.maxH = 60.0
            self.nHelperWells = 20 # 7
            self.minQhelper = -600.0
            self.maxQhelper = 600.0
            self.deviationPenaltyFactor = 10.0

        if self.ENVTYPE in ['1r-d', '1r-c', '2r-d', '2r-c', '3r-d', '3r-c', '4r-d', '4r-c', '5r-d', '5r-c', '6r-d', '6r-c']:
            self.maxHChange = 0.2
            self.maxQChange = 50.0
            self.maxCoordChange = 2.5

        if self.ENVTYPE in ['1s-d', '1r-d', '2s-d', '2r-d']:
            self.actionSpace = ['up', 'keep', 'down']
        elif self.ENVTYPE in ['1s-c', '1r-c', '2s-c', '2r-c']:
            self.actionSpace = ['up', 'down']
        elif self.ENVTYPE in ['3s-d', '3r-d']:
           self.actionSpace = ['up', 'keep', 'down', 'left', 'right']
        elif self.ENVTYPE in ['3s-c', '3r-c']:
           self.actionSpace = ['up', 'down', 'left', 'right']
        # inspired by https://stackoverflow.com/questions/42591283/all-possible-combinations-of-a-set-as-a-list-of-strings
        # this gets too large with many wells
        elif self.ENVTYPE in ['4s-d', '4r-d', '5s-d', '5r-d', '6s-d', '6r-d']:
            if self.ENVTYPE in ['4s-d', '4r-d']:
                self.actionSpaceIndividual = ['up', 'keep', 'down', 'left', 'right']
                self.actionRange = 2.5
            elif self.ENVTYPE in ['5s-d', '5r-d']:
                self.actionSpaceIndividual = ['moreQ', 'keepQ', 'lessQ']
                self.actionRangeQ = 100.0
            elif self.ENVTYPE in ['6s-d', '6r-d']:
                self.actionSpaceIndividual = ['up', 'keep', 'down', 'left', 'right', 'moreQ', 'keepQ', 'lessQ']
                self.actionRange = 2.5
                self.actionRangeQ = 100.0
            self.actionSpace = list(''.join(map(str, comb)) for comb in product(self.actionSpaceIndividual, repeat=self.nHelperWells))

        elif self.ENVTYPE in ['4s-c', '4r-c', '5s-c', '5r-c', '6s-c', '6r-c']:
            if self.ENVTYPE in ['4s-c', '4r-c']:
                self.actionSpaceIndividual = ['up', 'down', 'left', 'right']
                self.actionRange = 2.5
            elif self.ENVTYPE in ['5s-c', '5r-c']:
                self.actionSpaceIndividual = ['moreQ', 'lessQ']
                self.actionRangeQ = 100.0
            elif self.ENVTYPE in ['6s-c', '6r-c']:
                self.actionSpaceIndividual = ['up', 'down', 'left', 'right', 'moreQ', 'lessQ']
                self.actionRange = 2.5
                self.actionRangeQ = 50.0

            self.actionSpace = []
            for i in range(self.nHelperWells):
                self.actionSpace += [j+str(i+1) for j in self.actionSpaceIndividual]

        self.actionSpaceSize = len(self.actionSpace)

        self.rewardMax = 1000
        self.distanceMax = 97.9

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
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self.actionValueNorth = uniform(self.minH, self.maxH)
            self.actionValueSouth = uniform(self.minH, self.maxH)
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            self.actionValue = uniform(self.minH, self.maxH)
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c']:
            self.actionValueX = self.wellX
            self.actionValueY = self.wellY
        elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.helperWells['actionValueX'+w] = self.helperWells['wellX'+w]
                self.helperWells['actionValueY'+w] = self.helperWells['wellY'+w]
        elif self.ENVTYPE in ['5s-d', '5s-c', '5r-d', '5r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.helperWells['actionValueQ'+w] = self.helperWells['wellQ'+w]
        elif self.ENVTYPE in ['6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.helperWells['actionValueX'+w] = self.helperWells['wellX'+w]
                self.helperWells['actionValueY'+w] = self.helperWells['wellY'+w]
                self.helperWells['actionValueQ'+w] = self.helperWells['wellQ'+w]

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

        self.constructModel()

    def initializeWellRate(self, minQ, maxQ):
        """Initialize well randomly in the aquifer domain within margins."""

        xmin = 0.0 + self.wellSpawnBufferXWest
        xmax = self.extentX - self.wellSpawnBufferXEast
        ymin = 0.0 + self.wellSpawnBufferY
        ymax = self.extentY - self.wellSpawnBufferY
        wellX = uniform(xmin, xmax)
        wellY = uniform(ymin, ymax)
        wellZ = self.zTop
        wellCoords = [wellX, wellY, wellZ]
        wellQ = uniform(minQ, maxQ)

        return wellX, wellY, wellZ, wellCoords, wellQ

    def initializeWell(self):
        """Implement initialized well as model feature."""
        l, c, r = self.cellInfoFromCoordinates([self.wellX,
                                                self.wellY,
                                                self.wellZ]
                                               )
        self.wellCellLayer, self.wellCellColumn, self.wellCellRow = l, c, r
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.helperWells['l'+w], self.helperWells['c'+w], self.helperWells['r'+w] = self.cellInfoFromCoordinates(
                    [self.helperWells['wellX'+w], self.helperWells['wellY'+w], self.helperWells['wellZ'+w]])

        # print('debug well cells', self.wellCellLayer, self.wellCellColumn, self.wellCellRow)
        # adding WEL package to the MODFLOW model
        lrcq = {0: [[l-1, r-1, c-1, self.wellQ]]}
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            lrcq_ = [[l-1, r-1, c-1, self.wellQ]]
            for i in range(self.nHelperWells):
                w = str(i+1)
                lrcq_.append([self.helperWells['l'+w]-1, self.helperWells['r'+w]-1, self.helperWells['c'+w]-1, self.helperWells['wellQ'+w]])
            lrcq = {0: lrcq_}
        self.wel = ModflowWel(self.mf, stress_period_data=lrcq)

    def initializeState(self, state):
        """Initialize aquifer hydraulic head with state from previous step."""

        self.headsPrev = copy(self.state['heads'])

    def getActionType(self, ENVTYPE):
        """Retrieve action type from ENVTYPE variable."""
        
        if '-d' in ENVTYPE:
            actionType = 'discrete'
        elif '-c' in ENVTYPE:
            actionType = 'continuous'
        else:
            print('Environment name is unknown.')
            quit()

        return actionType

    def updateModel(self):
        """Update model domain for transient simulation."""

        self.constructModel()

    def updateWellRate(self):
        """Update model to continue using well."""

        if self.ENVTYPE in ['1r-d', '1r-c', '2r-d', '2r-c', '3r-d', '3r-c', '4r-d', '4r-c', '5r-d', '5r-c', '6r-d', '6r-c']:
            # generating random well rate fluctuations
            dQ = uniform(low=-1.0, high=1.0)*self.maxQChange
            updatedQ = self.wellQ + dQ

            # applying fluctuation if not surpassing pumping rate constraints
            if updatedQ < self.maxQ and updatedQ > self.minQ:
                self.wellQ = updatedQ
            elif updatedQ > self.maxQ:
                self.wellQ = self.maxQ
            elif updatedQ < self.minQ:
                self.wellQ = self.minQ

    def updateWell(self):

        if self.ENVTYPE in ['1r-d', '1r-c', '2r-d', '2r-c', '4r-d', '4r-c', '5r-d', '5r-c', '6r-d', '6r-c']:
            # generating random well location fluctuations
            dwellX = uniform(low=-1.0, high=1.0)*self.maxCoordChange
            dwellY = uniform(low=-1.0, high=1.0)*self.maxCoordChange
            updatedwellX = self.wellX + dwellX
            updatedwellY = self.wellY + dwellY

            # applying fluctuation if not moving to model boundary cell
            if updatedwellX > self.dCol:
                if updatedwellX < self.extentX - self.dCol:
                    self.wellX = updatedwellX
            if updatedwellY > self.dRow:
                if updatedwellY < self.extentY - self.dRow:
                    self.wellY = updatedwellY

        if self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c']:
            # updating well location from action taken
            self.wellX = self.actionValueX
            self.wellY = self.actionValueY
            self.wellZ = self.wellZ
        self.wellCoords = [self.wellX, self.wellY, self.wellZ]

        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                # updating well location from action taken
                self.helperWells['wellX'+w] = self.helperWells['actionValueX'+w]
                self.helperWells['wellY'+w] = self.helperWells['actionValueY'+w]
                self.helperWells['wellZ'+w] = self.helperWells['wellZ'+w]
                self.helperWells['wellCoords'+w] = [self.helperWells['wellX'+w], self.helperWells['wellY'+w], self.helperWells['wellZ'+w]]

        if self.ENVTYPE in ['5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                # updating well location from action taken
                self.helperWells['wellQ'+w] = self.helperWells['actionValueQ'+w]

        # adding WEL package to the MODFLOW model
        l, c, r = self.cellInfoFromCoordinates([self.wellX,
            self.wellY, self.wellZ])
        self.wellCellLayer = l
        self.wellCellColumn = c
        self.wellCellRow = r
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.helperWells['l'+w], self.helperWells['c'+w], self.helperWells['r'+w] = self.cellInfoFromCoordinates(
                    [self.helperWells['wellX'+w], self.helperWells['wellY'+w], self.helperWells['wellZ'+w]])

        lrcq = {0: [[l-1, r-1, c-1, self.wellQ]]}
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            lrcq_ = [[l-1, r-1, c-1, self.wellQ]]
            for i in range(self.nHelperWells):
                w = str(i+1)
                lrcq_.append([self.helperWells['l'+w]-1, self.helperWells['r'+w]-1, self.helperWells['c'+w]-1, self.helperWells['wellQ'+w]])
            lrcq = {0: lrcq_}
        self.wel = ModflowWel(self.mf, stress_period_data=lrcq)

    def constructModel(self):
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
                                  steady=self.periodSteadiness,
                                  itmuni=4, # time units: days
                                  lenuni=2 # time units: meters
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
                                  # +1 is needed here, as 2 seems to equal 1 day, and so on
                                  # somehow longer head simulation is necessary to do particle tracking in the same timeframe
                                  perlen=[2*self.periodLength],
                                  itmuni=4, # time units: days
                                  lenuni=2 # time units: meters
                                  )

        self.ibound = ones((self.nLay, self.nRow, self.nCol), dtype=int32)
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            self.ibound[:, 1:-1, 0] = -1
            self.ibound[:, 1:-1, -1] = -1
            self.ibound[:, 0, :] = -1
            self.ibound[:, -1, :] = -1
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            self.ibound[:, :-1, 0] = -1
            self.ibound[:, :-1, -1] = -1
            self.ibound[:, -1, :] = -1

        if self.periodSteadiness:
            self.strt = ones((self.nLay, self.nRow, self.nCol),
                             dtype=float32
                             )
        elif self.periodSteadiness == False:
            self.strt = self.headsPrev

        if self.timeStep > 0:
            if self.ENVTYPE in ['3r-d', '3r-c', '4r-d', '4r-c', '5r-d', '5r-c', '6r-d', '6r-c']:
                # generating random boundary condition fluctuations
                dHSouth = uniform(low=-1.0, high=1.0)*self.maxHChange
                dHNorth = uniform(low=-1.0, high=1.0)*self.maxHChange
                updatedSpecSouth = self.headSpecSouth + dHSouth
                updatedSpecNorth = self.headSpecNorth + dHNorth
                # applying fluctuation if not surpassing head constraints
                if updatedSpecSouth > self.minH and updatedSpecSouth < self.maxH:
                    self.headSpecSouth = updatedSpecSouth
                elif updatedSpecSouth < self.minH:
                    self.headSpecSouth = self.minH
                elif updatedSpecSouth > self.maxH:
                    self.headSpecSouth = self.maxH
                if updatedSpecNorth > self.minH and updatedSpecNorth < self.maxH:
                    self.headSpecNorth = updatedSpecNorth
                elif updatedSpecNorth < self.minH:
                    self.headSpecNorth = self.minH
                elif updatedSpecNorth > self.maxH:
                    self.headSpecNorth = self.maxH

        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            self.strt[:, 1:-1, 0] = self.headSpecWest
            self.strt[:, 1:-1, -1] = self.headSpecEast
            self.strt[:, 0, :] = self.actionValueSouth
            self.strt[:, -1, :] = self.actionValueNorth
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            self.strt[:, :-1, 0] = self.headSpecWest
            self.strt[:, :-1, -1] = self.headSpecEast
            self.strt[:, -1, :] = self.actionValue
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            self.strt[:, 1:-1, 0] = self.headSpecWest
            self.strt[:, 1:-1, -1] = self.headSpecEast
            self.strt[:, 0, :] = self.headSpecSouth
            self.strt[:, -1, :] = self.headSpecNorth


        ModflowBas(self.mf, ibound=self.ibound, strt=self.strt)

        # adding LPF package to the MODFLOW model
        ModflowLpf(self.mf, hk=10., vka=10., ss=1e-05, sy=0.15, ipakcb=53)

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
        # debugging model setup if enabled
        # self.check = self.mf.check(verbose=True)

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
            self.realTime = self.times[-1]
            # print('debug self.times', self.times)
            # print('debug self.realTime', self.realTime)
            self.heads = self.hdobj.get_data(totim=self.times[-1])

        # loading discharge data
        self.fnameBudget = join(self.modelpth, self.MODELNAME + '.cbc')
        with CellBudgetFile(self.fnameBudget) as cbf:
            self.cbb = cbf
            self.frf = self.cbb.get_data(text='FLOW RIGHT FACE')[0]
            self.fff = self.cbb.get_data(text='FLOW FRONT FACE')[0]

    def runMODPATH(self):
        """Execute forward particle tracking simulation using MODPATH."""

        # this needs to be transformed, yet not understood why
        self.particleCoords[0] = self.extentX - self.particleCoords[0]

        # creating MODPATH simulation objects
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
        # refer to documentation https://pubs.usgs.gov/tm/6a41/pdf/TM_6A_41.pdf
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
                # out.append(str(2) + '\n')
                out.append(str(2) + '\n')
                # why does removing this work?
                del out[7]
                # TimePoints
                # out.append('0.000000   ' + '{:.6f}'.format(self.realTime) + '\n')
                # 5.000001E-01
                # out.append('0.000000   5.000001E-01\n')
                out.append('0.000000   1.000000\n')
            if keepFlag:
                out.append(line)
        fIn.close()

        # print('debug self.realTime', self.realTime)

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
        # in MODPATH input file
        # with fracCol coordinate correction being critical.
        # taking fraction float and remove floored integer from it
        fracCol = 1.0 - ((self.particleCoords[0] / self.dCol) - float(int(self.particleCoords[0] / self.dCol)))
        fracRow = (self.particleCoords[1] / self.dRow) - float(int((self.particleCoords[1] / self.dRow)))
        fracVer = self.particleCoords[2] / self.dVer

        # writing current particle location to file
        fOut = open(join(self.modelpth, self.MODELNAME + '.mploc'),
                    'w')
        fOut.write('1\n')
        fOut.write(u'1\n')
        fOut.write(u'particle\n')
        fOut.write(u'1\n')
        fOut.write(u'1 1 1' + u' ' + str(self.nLay - l + 1) +
                   u' ' + str(self.nRow - r + 1) +
                   u' ' + str(self.nCol - c + 1) +
                   u' ' + str('%.6f' % (fracCol)) +
                   u' ' + str('%.6f' % (fracRow)) +
                   u' ' + str('%.6f' % (fracVer)) +
                   u' 0.000000 ' + u'particle\n')

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
        # print('debug self.p0[time]', self.p0['time'])

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
        lengthActual = self.calculatePathLength(x, y)
        lengthShortest = x[-1] - x[0]
        distanceFraction = lengthShortest / self.distanceMax
        self.rewardMaxGame = (distanceFraction * self.rewardMax)

        # pathLengthRatio defines the fraction of the highest possible reward
        if lengthActual != 0.:
            pathLengthRatio = lengthShortest / lengthActual
            self.gameReward = self.rewardMaxGame * \
                (pathLengthRatio**self.deviationPenaltyFactor)
        elif lengthActual == 0.:
            # returning no reward if travelled neither forwards nor backwards
            pathLengthRatio = 1.0
            self.gameReward = 0.
        # negative reward for traveling backwards
        # potential problem: maybe reward for going backward and then going
        # less deviating way forward?
        if lengthShortest < 0.:
            self.gameReward *= -1.0 * self.gameReward

        return self.gameReward

    def reset(self, _seed=None, MODELNAME=None, initWithSolution=None):
        """Reset environment with same settings but potentially new seed."""
        
        if initWithSolution == None:
            initWithSolution=self.initWithSolution
        
        self.__init__(self.ENVTYPE, self.PATHMF2005, self.PATHMP6,
            self.MODELNAME if MODELNAME is None else MODELNAME,
            _seed=_seed, flagSavePlot=self.SAVEPLOT,
            flagManualControl=self.MANUALCONTROL, flagRender=self.RENDER,
            nLay=self.nLay, nRow=self.nRow, nCol=self.nCol,
            initWithSolution=initWithSolution)
        close()

    def render(self, returnFigure=False, dpi=120):
        """Plot the simulation state at the current timestep.
        Displaying and/or saving the visualisation. The active display can take
        user input from the keyboard to control the environment.
        """
        if self.timeStep == 0:
            self.renderInitializeCanvas()
            self.plotfilesSaved = []
            self.extent = (self.dRow / 2.0, self.extentX - self.dRow / 2.0,
             self.extentY - self.dCol / 2.0, self.dCol / 2.0)

        self.modelmap = PlotMapView(model=(self.mf), layer=0)
        self.headsplot = self.modelmap.plot_array((self.heads), masked_values=[
         999.0],
          alpha=0.5,
          zorder=2,
          cmap=(get_cmap('terrain')))
        self.quadmesh = self.modelmap.plot_ibound(zorder=3)
        self.renderWellSafetyZone(zorder=3)
        self.renderContourLines(n=30, zorder=4)
        self.renderIdealParticleTrajectory(zorder=5)
        self.renderTextOnCanvasPumpingRate(zorder=10)
        self.renderTextOnCanvasGameOutcome(zorder=10)
        self.renderParticle(zorder=6)
        self.renderParticleTrajectory(zorder=6)
        self.renderTextOnCanvasTimeAndScore(zorder=10)
        self.renderRemoveAxesTicks()
        self.renderSetAxesLimits()
        self.renderAddAxesTextLabels()
        for ax in [self.ax, self.ax2, self.ax3]:
            ax.axis('off')

        if returnFigure:
            s = self.fig.get_size_inches()
            self.fig.gca().set_axis_off()
            margins(0,0)
            self.fig.gca().xaxis.set_major_locator(NullLocator())
            self.fig.gca().yaxis.set_major_locator(NullLocator())
            self.fig.tight_layout(pad=0.)
            self.fig.set_size_inches(7, 7)
            self.fig.set_dpi(dpi)
            self.fig.canvas.draw()
            data = self.fig.canvas.tostring_rgb()
            rows, cols = self.fig.canvas.get_width_height()
            imarray = copy(fromstring(data, dtype=uint8).reshape(cols, rows, 3))
            self.fig.set_size_inches(s)
        if not returnFigure:
            if self.MANUALCONTROL:
                self.renderUserInterAction()
            else:
                if not self.MANUALCONTROL:
                    if self.RENDER:
                        show(block=False)
                        pause(self.MANUALCONTROLTIME)
            if self.SAVEPLOT:
                self.renderSavePlot(dpi=dpi)
                if self.done or self.timeStep == self.NAGENTSTEPS:
                    self.renderAnimationFromFiles()

        self.renderClearAxes()
        del self.headsplot

        if returnFigure:
            return imarray

    def renderInitializeCanvas(self):
        """Initialize plot canvas with figure and axes."""
        self.fig = figure(figsize=(7, 7))

        if 'ipykernel' in modules:
            self.flagFromIPythonNotebook = True
        else:
            self.flagFromIPythonNotebook = False

        if not self.flagFromIPythonNotebook:
            self.figManager = get_current_fig_manager()
            maximized = False
            try:
                self.figManager.window.state('zoomed')
                maximized = True
            except:
                pass
            if not maximized:
                try:
                    self.figManager.full_screen_toggle()
                except:
                    pass

        self.ax = self.fig.add_subplot(1, 1, 1, aspect='equal')
        self.ax3 = self.ax.twinx()
        self.ax2 = self.ax3.twiny()

        self.fig.gca().set_axis_off()
        margins(0,0)
        self.fig.gca().xaxis.set_major_locator(NullLocator())
        self.fig.gca().yaxis.set_major_locator(NullLocator())
        self.fig.tight_layout(pad=0.)

    def renderIdealParticleTrajectory(self, zorder=5):
        """Plot ideal particle trajectory associated with maximum reward."""
        self.ax2.plot([self.minX, self.minX + self.extentX], [
         self.particleY, self.particleY],
          lw=1.5,
          c='white',
          linestyle='--',
          zorder=zorder,
          alpha=0.5)

    def renderContourLines(self, n=30, zorder=4):
        """Plot n contour lines of the head field."""
        self.levels = linspace(min(self.heads), max(self.heads), n)
        self.contours = self.modelmap.contour_array((self.heads), levels=(self.levels),
          alpha=0.5,
          zorder=zorder)

    def renderWellSafetyZone(self, zorder=3):
        """Plot well safety zone."""
        wellBufferCircle = Circle((self.wellCoords[0], self.extentY - self.wellCoords[1]), (self.wellRadius),
          edgecolor='r',
          facecolor=None,
          fill=False,
          zorder=zorder,
          alpha=1.0,
          lw=2.0,
          label='protection zone')
        self.ax2.add_artist(wellBufferCircle)
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            wellCoords = []
            for i in range(self.nHelperWells):
                w = str(i+1)
                wellCoords.append(self.helperWells['wellCoords'+w])
            for i, c in enumerate(wellCoords):
                if self.helperWells['wellQ'+str(i+1)] >= 0.:
                    wellSafetyZoneColor = 'blue'
                else:
                    wellSafetyZoneColor = 'red'
                wellBufferCircle = Circle((c[0], self.extentY - c[1]), (self.helperWellRadius),
                  edgecolor=wellSafetyZoneColor,
                  facecolor=None,
                  fill=False,
                  zorder=zorder,
                  alpha=0.5,
                  lw=2.0,
                  label='protection zone')
                self.ax2.add_artist(wellBufferCircle)

    def renderTextOnCanvasPumpingRate(self, zorder=10):
        """Plot pumping rate on figure."""
        self.ax2.text((self.wellX + 3.0), (self.extentY - self.wellY), (str(int(self.wellQ)) + '\nm3/d'),
          fontsize=12,
          color='black',
          zorder=zorder)

    def renderTextOnCanvasGameOutcome(self, zorder=10):
        """Plot final game outcome on figure."""
        gameResult = ''
        if self.done:
            if self.success:
                gameResult = 'Success.'
            elif self.success == False:
                gameResult = 'Failure.'
        self.ax2.text(35, 80, gameResult, fontsize=30,
          color='red',
          zorder=zorder)

    def renderTextOnCanvasTimeAndScore(self, zorder=10):
        """Plot final game outcome on figure."""
        timeString = '%.0f' % (float(self.timeStep) * self.periodLength)
        self.ax2.text(5, 92, ('score: ' + str(int(self.rewardCurrent)) + '\ntime: ' + timeString + ' d'),
          fontsize=12,
          zorder=zorder)

    def renderParticle(self, zorder=6):
        """Plot particle at current current state."""
        if self.timeStep == 0:
            self.ax2.scatter((self.minX), (self.particleCoords[1]),
              lw=4,
              c='red',
              zorder=zorder)
        elif self.timeStep > 0:
            self.ax2.scatter((self.trajectories['x'][(-1)][(-1)]), (self.trajectories['y'][(-1)][(-1)]),
              lw=2,
              c='red',
              zorder=zorder)

    def renderParticleTrajectory(self, zorder=6):
        """Plot particle trajectory until current state."""
        if self.timeStep > 0:
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
                self.ax2.plot((self.trajectories['x'][i]), (self.trajectories['y'][i]),
                  lw=2,
                  c=(colorsRGBA[idxCount + colorsLens[i] - 1, :]),
                  zorder=zorder)
                idxCount += 1

    def renderRemoveAxesTicks(self):
        """Remove axes ticks from figure."""
        (self.ax.set_xticks([]), self.ax.set_yticks([]))
        (self.ax2.set_xticks([]), self.ax2.set_yticks([]))
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            (
             self.ax3.set_xticks([]), self.ax3.set_yticks([]))

    def renderSetAxesLimits(self):
        """Set limits of axes from given extents of environment domain."""
        self.ax.set_xlim(left=(self.minX), right=(self.minX + self.extentX))
        self.ax.set_ylim(bottom=(self.minY), top=(self.minY + self.extentY))
        self.ax2.set_xlim(left=(self.minX), right=(self.minX + self.extentX))
        self.ax2.set_ylim(bottom=(self.minY), top=(self.minY + self.extentY))
        self.ax3.set_xlim(left=(self.minX), right=(self.minX + self.extentX))
        self.ax3.set_ylim(bottom=(self.minY), top=(self.minY + self.extentY))

    def renderAddAxesTextLabels(self):
        """Add labeling text to axes."""
        left, width = self.minX, self.minX + self.extentX
        bottom, height = self.minY, self.minY + self.extentY
        top = bottom + height
        textRight = 'Destination:   ' + str('%.2f' % self.headSpecEast) + ' m'
        textLeft = 'Start:   ' + str('%.2f' % self.headSpecWest) + ' m'
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            textTop = str('%.2f' % self.actionValueNorth) + ' m'
            textBottom = str('%.2f' % self.actionValueSouth) + ' m'
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            textTop = ''
            textBottom = str('%.2f' % self.actionValue) + ' m'
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            textTop = str('%.2f' % self.headSpecNorth) + ' m'
            textBottom = str('%.2f' % self.headSpecSouth) + ' m'
        self.ax2.text((self.minX + 2 * self.dCol), (0.5 * (bottom + top)), textLeft, horizontalalignment='left',
          verticalalignment='center',
          rotation='vertical',
          zorder=10,
          fontsize=12)
        self.ax2.text((self.extentX - 2 * self.dCol), (0.5 * (bottom + top)), textRight, horizontalalignment='right',
          verticalalignment='center',
          rotation='vertical',
          zorder=10,
          fontsize=12)
        self.ax2.text((0.5 * (left + width)), (self.extentY - 2 * self.dRow), textTop, horizontalalignment='center',
          verticalalignment='top',
          rotation='horizontal',
          zorder=10,
          fontsize=12)
        self.ax2.text((0.5 * (left + width)), (self.minY + 2 * self.dRow), textBottom, horizontalalignment='center',
          verticalalignment='bottom',
          rotation='horizontal',
          zorder=10,
          fontsize=12)

    def renderUserInterAction(self):
        """Enable user control of the environment."""
        if self.timeStep == 0:
            if 'ipykernel' in modules:
                self.flagFromIPythonNotebook = True
            else:
                self.flagFromIPythonNotebook = False
        if self.flagFromIPythonNotebook:
            self.fig.canvas.mpl_connect('key_press_event', self.captureKeyPress)
            show(block=False)
            waitforbuttonpress(timeout=(self.MANUALCONTROLTIME))
        elif not self.flagFromIPythonNotebook:
            self.fig.canvas.mpl_connect('key_press_event', self.captureKeyPress)
            show(block=False)
            waitforbuttonpress(timeout=(self.MANUALCONTROLTIME))

    def renderSavePlot(self, dpi=120):
        """Save plot of the currently rendered timestep."""
        if self.timeStep == 0:
            self.plotsfolderpth = join(self.wrkspc, 'runs')
            self.plotspth = join(self.wrkspc, 'runs', self.ANIMATIONFOLDER)
            if not exists(self.plotsfolderpth):
                makedirs(self.plotsfolderpth)
            if not exists(self.plotspth):
                makedirs(self.plotspth)
        plotfile = join(self.plotspth, self.MODELNAME + '_' + str(self.timeStep).zfill(len(str(abs(self.NAGENTSTEPS))) + 1) + '.png')

        s = self.fig.get_size_inches()

        self.fig.gca().set_axis_off()
        margins(0,0)
        self.fig.gca().xaxis.set_major_locator(NullLocator())
        self.fig.gca().yaxis.set_major_locator(NullLocator())
        self.fig.tight_layout(pad=0.)

        self.fig.set_size_inches(7, 7)
        self.fig.savefig(plotfile, dpi=120, bbox_inches = 'tight', pad_inches = 0)
        self.fig.set_size_inches(s)
        self.plotfilesSaved.append(plotfile)

    def renderClearAxes(self):
        """Clear all axis after timestep."""
        try:
            self.ax.cla()
            self.ax.clear()
        except:
            pass

        try:
            self.ax2.cla()
            self.ax2.clear()
        except:
            pass

        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            try:
                self.ax3.cla()
                self.ax3.clear()
            except:
                pass

    def renderAnimationFromFiles(self):
        """Create animation of fulll game run.
        Code taken from and credit to:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
        """
        with get_writer((join(self.wrkspc, 'runs', self.ANIMATIONFOLDER, self.MODELNAME + '.gif')),
          mode='I') as (writer):
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
        if layer == 0:
            layer = 1
        if column == 0:
            column = 1
        if row == 0:
            row = 1
        if layer > self.nLay:
            layer = self.nLay
        if column > self.nCol:
            column = self.nCol
        if row > self.nRow:
            row = self.nRow
        return (layer, column, row)

    def surroundingHeadsFromCoordinates(self, coords, distance):
        """Determine hydraulic head of surrounding cells. Returns head of the
        same cell in the case of surrounding edges of the environment domain.
        """
        headsSurrounding = []
        for rIdx in range(3):
            for cIdx in range(3):
                if rIdx == 1:
                    if cIdx == 1:
                        continue
                    else:
                        rDistance = distance * (rIdx - 1)
                        cDistance = distance * (cIdx - 1)
                        coordX, coordY, coordZ = coords[0] + cDistance, coords[1] + rDistance, coords[2]
                        adjustedCoords = False
                        if coords[0] + cDistance > self.extentX:
                            coordX = self.extentX
                            adjustedCoords = True
                        if coords[0] + cDistance < self.minX:
                            coordX = self.minX
                            adjustedCoords = True
                        if coords[1] + rDistance > self.extentY:
                            coordY = self.extentY
                            adjustedCoords = True
                        if coords[1] + rDistance < self.minY:
                            coordY = self.minY
                            adjustedCoords = True
                        if adjustedCoords:
                            l, c, r = self.cellInfoFromCoordinates([coordX, coordY, coordZ])
                            headsSurrounding.append(self.heads[(l - 1, r - 1, c - 1)])
                    if not adjustedCoords:
                        try:
                            l, c, r = self.cellInfoFromCoordinates([coords[0] + cDistance, coords[1] + rDistance, coords[2]])
                            headsSurrounding.append(self.heads[(l - 1, r - 1, c - 1)])
                        except Exception as e:
                            print(e)
                            print('Something went wrong. Maybe the queried coordinates reside outside the model domain?')

        return headsSurrounding

    def calculatePathLength(self, x, y):
        """Calculate length of advectively traveled path."""
        n = len(x)
        lv = []

        # for i in range(n):
        #     if i > 0:
        #         lv.append(sqrt((x[i] - x[(i - 1)]) ** 2 + (y[i] - y[(i - 1)]) ** 2))
        for i in range(n-1):
            lv.append(sqrt((x[i+1] - x[(i+1 - 1)]) ** 2 + (y[i+1] - y[(i+1 - 1)]) ** 2))

        pathLength = sum(lv)
        return pathLength

    def captureKeyPress(self, event):
        """Capture key pressed through manual user interaction."""
        self.keyPressed = event.key

    def setActionValue(self, action):
        """Set values for performed actions."""
        if self.ENVTYPE in ['1s-d', '1r-d']:
            if action == 'up':
                self.actionValueNorth += self.actionRange
                self.actionValueSouth += self.actionRange
            elif action == 'down':
                self.actionValueNorth -= self.actionRange
                self.actionValueSouth -= self.actionRange
        elif self.ENVTYPE in ['1s-c', '1r-c']:
            self.actionValueNorth += action[0]*self.actionRange
            self.actionValueSouth += action[0]*self.actionRange
            self.actionValueNorth -= action[1]*self.actionRange
            self.actionValueSouth -= action[1]*self.actionRange

        elif self.ENVTYPE in ['2s-d', '2r-d']:
            if action == 'up':
                self.actionValue += self.actionRange
            elif action == 'down':
                self.actionValue -= self.actionRange
        elif self.ENVTYPE in ['2s-c', '2r-c']:
            self.actionValue += action[0]*self.actionRange
            self.actionValue -= action[1]*self.actionRange

        elif self.ENVTYPE in ['3s-d', '3r-d']:
            if action == 'up':
                if self.wellY > self.dRow + self.actionRange:
                    self.actionValueY = self.wellY - self.actionRange
            elif action == 'down':
                if self.wellY < self.extentY - self.dRow - self.actionRange:
                    self.actionValueY = self.wellY + self.actionRange
            elif action == 'left':
                if self.wellX > self.dCol + self.actionRange:
                    self.actionValueX = self.wellX - self.actionRange
            elif action == 'right':
                if self.wellX < self.extentX - self.dCol - self.actionRange:
                    self.actionValueX = self.wellX + self.actionRange
        elif self.ENVTYPE in ['3s-c', '3r-c']:
            dyUp = action[0]*self.actionRange
            dyDown = action[1]*self.actionRange
            dy = dyUp - dyDown
            dxLeft = action[2]*self.actionRange
            dxRight = action[3]*self.actionRange
            dx = dxLeft - dxRight
            # sequence of action values currently might lead to counteracting actions not being allowed
            if self.wellY + dy > self.dRow:
                if self.wellY < self.extentY - self.dRow - dy:
                    self.actionValueY = self.wellY + dy
            if self.wellX + dx > self.dCol:
                if self.wellX + dx < self.extentX - self.dCol:
                    self.actionValueX = self.wellX + dx

        elif self.ENVTYPE in ['4s-d', '4r-d', '5s-d', '5r-d', '6s-d', '6r-d']:
            actionList = []
            while len(action) != 0:
                for action_ in self.actionSpaceIndividual:
                    if action[:len(action_)] == action_:
                        actionList.append(action_)
                        action = action[len(action_):]
            for i in range(self.nHelperWells):
                w = str(i+1)
                if actionList[i] == 'up':
                    if self.helperWells['wellY'+w] > self.dRow + self.actionRange:
                        self.helperWells['actionValueY'+w] = self.helperWells['wellY'+w] - self.actionRange
                elif actionList[i] == 'down':
                    if self.helperWells['wellY'+w] < self.extentY - self.dRow - self.actionRange:
                        self.helperWells['actionValueY'+w] = self.helperWells['wellY'+w] + self.actionRange
                elif actionList[i] == 'left':
                    if self.helperWells['wellX'+w] > self.dCol + self.actionRange:
                        self.helperWells['actionValueX'+w] = self.helperWells['wellX'+w] - self.actionRange
                elif actionList[i] == 'right':
                    if self.helperWells['wellX'+w] < self.extentX - self.dCol - self.actionRange:
                        self.helperWells['actionValueX'+w] = self.helperWells['wellX'+w] + self.actionRange
                elif actionList[i] == 'moreQ':
                    if self.helperWells['wellQ'+w] - self.actionRangeQ > self.minQhelper:
                        self.helperWells['actionValueQ'+w] = self.helperWells['wellQ'+w] - self.actionRangeQ
                elif actionList[i] == 'lessQ':
                    if self.helperWells['wellQ'+w] + self.actionRangeQ < self.maxQhelper:
                        self.helperWells['actionValueQ'+w] = self.helperWells['wellQ'+w] + self.actionRangeQ
        elif self.ENVTYPE in ['4s-c', '4r-c', '5s-c', '5r-c', '6s-c', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                offset = int(i*len(self.actionSpaceIndividual))
                if self.ENVTYPE in ['4s-c', '4r-c', '6s-c', '6r-c']:
                    dyUp = action[offset+0]*self.actionRange
                    dyDown = action[offset+1]*self.actionRange
                    dy = dyUp - dyDown
                    dxLeft = action[offset+2]*self.actionRange
                    dxRight = action[offset+3]*self.actionRange
                    dx = dxLeft - dxRight
                    if self.helperWells['wellY'+w] + dy > self.dRow:
                        if self.helperWells['wellY'+w] < self.extentY - self.dRow - dy:
                            self.helperWells['actionValueY'+w] = self.helperWells['wellY'+w] + dy
                    if self.helperWells['wellX'+w] + dx > self.dCol:
                        if self.helperWells['wellX'+w] + dx < self.extentX - self.dCol:
                            self.helperWells['actionValueX'+w] = self.helperWells['wellX'+w] + dx
                if self.ENVTYPE in ['5s-c', '5r-c', '6s-c', '6r-c']:
                    if self.ENVTYPE in ['5s-c', '5r-c']:
                        dQMore = action[offset+0]*self.actionRangeQ
                        dQLess = action[offset+1]*self.actionRangeQ
                    elif self.ENVTYPE in ['6s-c', '6r-c']:
                        dQMore = action[offset+4]*self.actionRangeQ
                        dQLess = action[offset+5]*self.actionRangeQ
                    dQ = dQLess - dQMore
                    if self.helperWells['wellQ'+w] + dQ > self.minQhelper:
                        if self.helperWells['wellQ'+w] + dQ < self.maxQhelper:
                            self.helperWells['actionValueQ'+w] = self.helperWells['wellQ'+w] + dQ
    
    def observationsDictToVector(self, observationsDict):
        """Convert dictionary of observations to list."""
        observationsVector = []
        if self.OBSPREP != 'convolutional':
            if 'particleCoords' in observationsDict.keys():
                for obs in observationsDict['particleCoords']:
                    observationsVector.append(obs)
            # full field not longer part of reported state
            # for obs in observationsDict['headsSampledField'].flatten().flatten():
            #     observationsVector.append(obs)
            if 'heads' in observationsDict.keys():
                for obs in observationsDict['heads']:
                    observationsVector.append(obs)
            if 'wellQ' in observationsDict.keys():
                observationsVector.append(observationsDict['wellQ'])
            if 'wellCoords' in observationsDict.keys():
                for obs in observationsDict['wellCoords']:
                    observationsVector.append(obs)
            for i in range(self.nHelperWells):
                iStr = str(i)
                if 'wellQ'+iStr in observationsDict.keys():
                    observationsVector.append(observationsDict['wellQ'+iStr])
                if 'wellCoords'+iStr in observationsDict.keys():
                    for obs in observationsDict['wellCoords'+iStr]:
                        observationsVector.append(obs)
        elif self.OBSPREP == 'convolutional':
            pass
            # span across layer for all others

        return observationsVector

    def observationsVectorToDict(self, observationsVector):
        """Convert list of observations to dictionary."""

        # needs knowledge of number of helper wells

        observationsDict = {}
        offset = 4*self.nHelperWells
        observationsDict['particleCoords'] = observationsVector[:3]
        observationsDict['heads'] = observationsVector[3:-(4-offset)]
        observationsDict['wellQ'] = observationsVector[-(4-offset)]
        observationsDict['wellCoords'] = observationsVector[-(3-offset):]
        for i in range(self.nHelperWells):
            iStr = str(i)
            offset = offset-4
            observationsDict['wellQ'+iStr] = observationsVector[-(4-offset)]
            observationsDict['wellCoords'+iStr] = observationsVector[-(3-offset):]

        return observationsDict

    def unnormalize(self, data):
        from numpy import multiply

        keys = data.keys()
        if 'particleCoords' in keys:
            data['particleCoords'] = multiply(data['particleCoords'],
                self.minX + self.extentX)
        if 'heads' in keys:
            data['heads'] = multiply(data['heads'],
                self.maxH)
        if 'wellQ' in keys:
            data['wellQ'] = multiply(data['wellQ'], self.minQ)
        if 'wellCoords' in keys:
            data['wellCoords'] = multiply(data['wellCoords'], self.minX + self.extentX)
        if 'rewards' in keys:
            data['rewards'] = multiply(data['rewards'], self.rewardMax)
        return data

class FloPyArcade():
    """Instance of a FLoPy arcade game.

    Initializes a game agent and environment. Then allows to play the game.
    """

    def __init__(self, agent=None, modelNameLoad=None, modelName='FloPyArcade',
        animationFolder=None, NAGENTSTEPS=200, PATHMF2005=None, PATHMP6=None,
        surrogateSimulator=None, flagSavePlot=False,
        flagManualControl=False, actions=None, flagRender=False,
        keepTimeSeries=False, nLay=1, nRow=100, nCol=100):
        """Constructor."""

        self.PATHMF2005 = PATHMF2005
        self.PATHMP6 = PATHMP6
        self.SURROGATESIMULATOR = surrogateSimulator
        self.NAGENTSTEPS = NAGENTSTEPS
        self.SAVEPLOT = flagSavePlot
        self.MANUALCONTROL = flagManualControl
        self.RENDER = flagRender
        self.MODELNAME = modelName if modelName is not None else modelNameLoad
        self.ANIMATIONFOLDER = animationFolder if modelName is not None else modelNameLoad
        self.agent = agent
        self.MODELNAMELOAD = modelNameLoad
        self.done = False
        self.keepTimeSeries = keepTimeSeries
        self.actions = actions
        self.nLay, self.nRow, self.nCol = nLay, nRow, nCol

        self.ENVTYPES = ['1s-d', '1s-c', '1r-d', '1r-c',
                         '2s-d', '2s-c', '2r-d', '2r-c',
                         '3s-d', '3s-c', '3r-d', '3r-c',
                         '4s-c', '4r-c', # '4s-d', '4r-d'
                         '5s-c', '5r-c', # '5s-d', '5r-d'
                         '6s-c', '6r-c' # '6s-d', '6r-d'
                         ]

        self.wrkspc = FloPyEnv(initWithSolution=False).wrkspc

    def play(self, env=None, ENVTYPE='1s-d', seed=None, returnReward=False, verbose=False):
        """Play an instance of the Flopy arcade game."""

        t0 = time()

        # creating the environment
        if env is None:
            if self.SURROGATESIMULATOR is None:
                self.env = FloPyEnv(ENVTYPE, self.PATHMF2005, self.PATHMP6,
                    _seed=seed,
                    MODELNAME=self.MODELNAME if not None else 'FloPyArcade',
                    ANIMATIONFOLDER=self.ANIMATIONFOLDER if not None else 'FloPyArcade',
                    flagSavePlot=self.SAVEPLOT,
                    flagManualControl=self.MANUALCONTROL,
                    flagRender=self.RENDER,
                    NAGENTSTEPS=self.NAGENTSTEPS,
                    nLay=self.nLay,
                    nRow=self.nRow,
                    nCol=self.nCol)
            elif self.SURROGATESIMULATOR is not None:
                self.env = FloPyEnvSurrogate(self.SURROGATESIMULATOR, ENVTYPE,
                    MODELNAME=self.MODELNAME if not None else 'FloPyArcade',
                    _seed=seed,
                    NAGENTSTEPS=self.NAGENTSTEPS)

        # self.env.stepInitial()
        observations, self.done = self.env.observationsVectorNormalized, self.env.done
        if self.keepTimeSeries:
            # collecting time series of game metrices
            statesNormalized, stressesNormalized = [], []
            rewards, doneFlags, successFlags = [], [], []
            heads, actions, wellCoords, trajectories = [], [], [], []
            statesNormalized.append(observations)
            rewards.append(0.)
            doneFlags.append(self.done)
            successFlags.append(-1)
            heads.append(self.env.heads)
            wellCoords.append(self.env.wellCoords)

        # self.actionRange, self.actionSpace = self.env.actionRange, self.env.actionSpace
        self.actionSpace = self.env.actionSpace
        agent = FloPyAgent(actionSpace=self.actionSpace)

        # game loop
        self.success = False
        self.rewardTotal = 0.

        for self.timeSteps in range(self.NAGENTSTEPS):
            if not self.done:
                # without user control input: generating random agent action
                t0getAction = time()

                if self.MANUALCONTROL:
                    action = agent.getAction('manual', self.env.keyPressed, actionType=self.env.actionType)
                elif self.MANUALCONTROL == False:
                    if self.MODELNAMELOAD is None and self.agent is None:
                        action = agent.getAction('random', actionType=self.env.actionType)
                    elif self.MODELNAMELOAD is not None:
                        action = agent.getAction(
                            'modelNameLoad',
                            modelNameLoad=self.MODELNAMELOAD,
                            state=self.env.observationsVectorNormalized,
                            actionType=self.env.actionType
                            )
                    elif self.agent is not None:
                        action = agent.getAction(
                            'model',
                            agent=self.agent,
                            state=self.env.observationsVectorNormalized,
                            actionType=self.env.actionType
                            )

                if self.actions is not None and self.actions != []:
                    action = self.actions[self.timeSteps]
                # print('actions', action)



                '''
                # this section can be used for visualizing neural network at work

                from matplotlib.pyplot import hist, savefig, figure, subplots
                from numpy import unique, concatenate
                import tensorflow as tf

                agentModel = FloPyAgent().loadAgentModel(self.MODELNAMELOAD)
                weights = agentModel.get_weights()
                input_ = self.env.observationsVectorNormalized
                layers = agentModel.layers
                # print(layers[0].weights)
                # print(agentModel.layers)

                # shape 409, 300 and 300,
                denseIdxs = [0, 2, 4, 6, 8, 10]
                activationIdxs = [1, 3, 5, 7, 9]
                weightsComp, biasComp, activationsComp = [], [], []
                for idx in denseIdxs:
                    layer = layers[idx]
                    weights_ = layer.weights[0]
                    bias = layer.weights[1]
                    # print('shape(weights_)', shape(weights_))
                    # for j in range(shape(weights_)[0]):
                    #     weightsComp += list(weights_[j])
                    weightsComp += list(array(weights_).flatten())
                    # print('shape(bias)', shape(bias))
                    biasComp += list(array(bias))
                    # print('mean bias', mean(array(bias)))
                # print('biasComp', biasComp)

                # activationModels = 
                # for iLayer in activationIdxs:

                aux_model1 = tf.keras.Model(inputs=agentModel.inputs,
                                           outputs=[agentModel.layers[1].output])
                aux_model2 = tf.keras.Model(inputs=agentModel.inputs,
                                           outputs=[agentModel.layers[3].output])
                aux_model3 = tf.keras.Model(inputs=agentModel.inputs,
                                           outputs=[agentModel.layers[5].output])
                aux_model4 = tf.keras.Model(inputs=agentModel.inputs,
                                           outputs=[agentModel.layers[7].output])
                aux_model5 = tf.keras.Model(inputs=agentModel.inputs,
                                           outputs=[agentModel.layers[9].output])

                intermediate_layer_output1 = aux_model1.predict_on_batch(
                    array(input_).reshape(-1, (*shape(input_))))[0]
                intermediate_layer_output2 = aux_model2.predict_on_batch(
                    array(input_).reshape(-1, (*shape(input_))))[0]
                intermediate_layer_output3 = aux_model3.predict_on_batch(
                    array(input_).reshape(-1, (*shape(input_))))[0]
                intermediate_layer_output4 = aux_model4.predict_on_batch(
                    array(input_).reshape(-1, (*shape(input_))))[0]
                intermediate_layer_output5 = aux_model5.predict_on_batch(
                    array(input_).reshape(-1, (*shape(input_))))[0]

                activations = (list(intermediate_layer_output1.flatten())
                    + list(intermediate_layer_output2.flatten())
                    + list(intermediate_layer_output3.flatten())
                    + list(intermediate_layer_output4.flatten())
                    + list(intermediate_layer_output5.flatten()))

                print(self.env.timeStep, mean(action), min(action), max(action), shape(input_), mean(input_))
                print(self.env.timeStep, mean(activations), min(activations), max(activations))
                print('----')

                figWeights, axWeights = subplots()
                axWeights.hist(array(weightsComp), bins=100)
                figWeights.tight_layout()
                axWeights.set_xlim(-1., 1.)
                figWeights.savefig('C:\\weights' + str(self.env.timeStep) + '.png')
                close(figWeights)

                figBias, axBias = subplots()
                print('mean bias', min(biasComp), mean(biasComp), max(biasComp))
                axBias.hist(array(biasComp), bins=100)
                figBias.tight_layout()
                axBias.set_xlim(-1., 1.)
                figBias.savefig('C:\\bias' + str(self.env.timeStep) + '.png')
                close(figBias)

                figActions, axActions = subplots()
                figActions.tight_layout()
                axActions.plot(action)
                axActions.set_ylim(0, 1)
                figActions.savefig('C:\\actions' + str(self.env.timeStep) + '.png')
                close(figActions)

                figActivations, axActivations = subplots()
                figActivations.tight_layout()
                axActivations.plot(activations)
                figActivations.savefig('C:\\activations' + str(self.env.timeStep) + '.png')
                close(figActivations)

                figInput, axInput = subplots()
                axInput.hist(input_, bins=arange(-1., 2.05, 0.05))
                axInput.set_xlim(-1., 2.)
                figInput.savefig('C:\\inputs' + str(self.env.timeStep) + '.png')
                close(figInput)
                '''


                t0step = time()
                observations, reward, self.done, _ = self.env.step(
                    self.env.observationsVectorNormalized, action, self.rewardTotal)

                if self.keepTimeSeries:
                    # collecting time series of game metrices
                    statesNormalized.append(self.env.observationsVectorNormalized)
                    stressesNormalized.append(self.env.stressesVectorNormalized)
                    rewards.append(reward)

                    heads.append(self.env.heads)
                    doneFlags.append(self.done)
                    wellCoords.append(self.env.wellCoords)
                    actions.append(action)
                    if not self.done:
                        successFlags.append(-1)
                    if self.done:
                        if self.env.success:
                            successFlags.append(1)
                        elif not self.env.success:
                            successFlags.append(0)
                self.rewardTotal += reward

            if self.done or self.timeSteps == self.NAGENTSTEPS-1:

                if self.MANUALCONTROL:
                    # freezing screen shortly when game is done
                    sleep(5)

                if self.keepTimeSeries:
                    self.timeSeries = {}
                    self.timeSeries['statesNormalized'] = statesNormalized
                    self.timeSeries['stressesNormalized'] = stressesNormalized
                    self.timeSeries['rewards'] = rewards
                    self.timeSeries['doneFlags'] = doneFlags
                    self.timeSeries['successFlags'] = successFlags

                    self.timeSeries['heads'] = heads
                    self.timeSeries['wellCoords'] = wellCoords
                    self.timeSeries['actions'] = actions
                    self.timeSeries['trajectories'] = self.env.trajectories

                self.success = self.env.success
                if self.env.success:
                    successString = 'won'
                elif self.env.success == False:
                    successString = 'lost'
                    # total loss of reward if entering well protection zone
                    self.rewardTotal = 0.0

                if self.SURROGATESIMULATOR is not None:
                    stringSurrogate = 'surrogate '
                    # print('surrogate')
                else:
                    stringSurrogate = ''
                    # print('not surrogate')
                if verbose:
                    print('The ' + stringSurrogate + 'game was ' +
                          successString +
                          ' after ' +
                          str(self.timeSteps) +
                          ' timesteps with a reward of ' +
                          str(int(self.rewardTotal)) +
                          ' points.')
                close('all')
                break

        self.runtime = (time() - t0) / 60.

        if returnReward:
            return self.rewardTotal
