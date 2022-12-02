#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade
# provides: simulated groundwater flow environments to test reinforcement learning algorithms
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


# imports for environments
from copy import copy, deepcopy
from matplotlib import use as matplotlibBackend
matplotlibBackend('Agg')
from flopy.modflow import Modflow, ModflowBas, ModflowDis, ModflowLpf
from flopy.modflow import ModflowOc, ModflowPcg, ModflowWel
from flopy.mf6 import MFSimulation, ModflowTdis, ModflowIms, ModflowGwf
from flopy.mf6 import ModflowGwfdis, ModflowGwfic, ModflowGwfchd, ModflowGwfnpf
from flopy.mf6 import ModflowGwfoc, ModflowGwfsto, ModflowGwfrcha
from flopy.mf6.modflow.mfgwfwel import ModflowGwfwel
from flopy.modpath import Modpath6 as Modpath
from flopy.modpath import Modpath6Bas as ModpathBas
from flopy.plot import PlotMapView
# from flopy.utils import CellBudgetFile, HeadFile, MfGrdFile, PathlineFile
# from flopy.utils import CellBudgetFile
from flopy.utils import HeadFile, PathlineFile
from flopy.mf6.utils.binarygrid_util import MfGrdFile
from glob import glob
import warnings
warnings.filterwarnings('ignore', 'SelectableGroups dict interface')
from gym import Env as gymEnv
from gym import spaces
from imageio import get_writer, imread
from itertools import chain, product
from joblib import dump as joblibDump
from joblib import load as joblibLoad
from math import ceil, floor
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import Circle, close, figure, pause, show
from matplotlib.pyplot import get_current_fig_manager
from matplotlib.pyplot import margins, NullLocator
from matplotlib.pyplot import waitforbuttonpress
from matplotlib import text as mtext
from matplotlib.transforms import Bbox
from numpy import add, arange, argmax, argsort, array, ceil, chararray, copy, concatenate, divide
from numpy import expand_dims, extract, float32, frombuffer, int32, linspace, max, maximum, min, minimum
from numpy import mean, multiply, ones, prod, reshape, shape, sqrt, subtract, uint8, unique, zeros
from numpy import sum as numpySum
from numpy import abs as numpyAbs
from numpy.random import randint, random, randn, uniform
from numpy.random import seed as numpySeed
from os import chdir, chmod, environ, listdir, makedirs, pathsep, remove, rmdir
from os.path import abspath, dirname, exists, join, sep
from platform import system
from shutil import rmtree
from sys import modules
if 'ipykernel' in modules:
    from IPython import display
from time import sleep, time
from xmipy import XmiWrapper


# suppressing TensorFlow output on import, except fatal errors
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
from logging import getLogger, FATAL
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
getLogger('tensorflow').setLevel(FATAL)

# currently ignoring nested array error
# from numpy import VisibleDeprecationWarning
# warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
warnings.filterwarnings("ignore")

# additional imports for agents
from collections import deque, defaultdict
from datetime import datetime
from gc import collect as garbageCollect
from itertools import count, repeat
from pathos.multiprocessing import cpu_count
from pathos import helpers as pathosHelpers
# from multiprocessing import Pool as Pool
# from pathos.pools import _ProcessPool as Pool
# from pathos.pools import ProcessPool as Pool
# from pathos.pools import ParallelPool as Pool
from pathos.multiprocessing import Pool as Pool
from pathos.pools import _ThreadPool as ThreadPool
from pickle import dump, load
from tensorflow.keras.initializers import glorot_uniform, glorot_normal, random_uniform, random_normal
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import clone_model, load_model, model_from_json
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from random import sample as randomSample, seed as randomSeed
from random import shuffle
from tensorflow.compat.v1 import ConfigProto, set_random_seed
from tensorflow.compat.v1 import Session as TFSession
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import load_model as TFload_model
from tqdm import tqdm
from uuid import uuid4

from tempfile import TemporaryDirectory

# avoiding freeze issues on Linux when loading Tensorflow model
# https://github.com/keras-team/keras/issues/9964
# https://stackoverflow.com/questions/40615795/pathos-enforce-spawning-on-linux
pathosHelpers.mp.context._force_start_method('spawn')


# to avoid thread erros
# https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
# matplotlibBackend('Agg')


class FloPyAgent():
    """Agent to navigate a spawned particle advectively through one of the
    aquifer environments, collecting reward along the way.
    """

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

    def __init__(self, observationsVector=None, actionSpace=['keep'],
                 hyParams=None, envSettings=None, mode='random',
                 maxTasksPerWorker=1000, maxTasksPerWorkerMutate=1000,
                 maxTasksPerWorkerNoveltySearch=10000, zFill=6):
        """Constructor"""

        self.wrkspc = dirname(abspath(__file__))
        if 'library.zip' in self.wrkspc:
            # changing workspace in case of call from compiled executable
            self.wrkspc = dirname(dirname(self.wrkspc))

        # initializing arguments
        self.observationsVector = observationsVector
        self.hyParams, self.envSettings = hyParams, envSettings
        self.actionSpace = actionSpace
        self.actionSpaceSize = len(self.actionSpace)

        if self.envSettings:
            self.actionType = self.getActionType(self.envSettings['ENVTYPE'])
            # self.actionType = FloPyEnv(initWithSolution=False).getActionType(self.envSettings['ENVTYPE'])
        self.agentMode = mode
        self.maxTasksPerWorker = maxTasksPerWorker
        self.maxTasksPerWorkerMutate = maxTasksPerWorkerMutate
        self.maxTasksPerWorkerNoveltySearch = maxTasksPerWorkerNoveltySearch
        self.zFill = zFill

        # setting seeds
        if self.envSettings is not None:
            self.setSeeds()

        # creating required folders if inexistent
        self.modelpth = join(self.wrkspc, 'models')
        if not exists(self.modelpth):
            makedirs(self.modelpth)

        if self.agentMode == 'DQN':
            # initializing DQN agent
            self.initializeDQNAgent()

        if self.agentMode == 'genetic':
            if self.envSettings['NAGENTSPARALLEL'] == None:
                self.envSettings['NAGENTSPARALLEL'] = cpu_count()

            # creating required folders if inexistent
            self.tempModelpth = join(self.wrkspc, 'temp', 
                self.envSettings['MODELNAME'])

            if not self.envSettings['RESUME']:
                # removing previous models if existing to avoid loading outdated models
                modelFiles = glob(join(self.modelpth, self.envSettings['MODELNAME'] + '_' + '*'))
                for item in modelFiles:
                    try:
                        remove(item)
                    except:
                        files = glob(join(item, '*'))
                        for f in files:
                            remove(f)
                try:
                    rmtree(self.tempModelpth)
                except:
                    pass

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

            # initializing seeds for mutation
            self.mutationSeeds = []
            for iGen in range(self.hyParams['NGENERATIONS']):
                seedsInnerGeneration = list(randint(
                    low=1, high=10000000, size=self.hyParams['NAGENTS']))
                self.mutationSeeds.append(seedsInnerGeneration)
                self.setSeeds()

            # initializing seeds for games
            if self.envSettings['SEEDSRANDOM']:
                self.gamesSeeds = []
                self.bestRewards = []
                for iGen in range(self.hyParams['NGENERATIONS']):
                    seedsInnerGeneration = list(randint(
                        low=1, high=10000000, size=self.hyParams['NGAMESAVERAGED']))
                    self.gamesSeeds.append(seedsInnerGeneration)
                self.setSeeds()

            # initializing genetic agents and saving hyperparameters and
            # environment settings or loading them if resuming
            if not self.envSettings['RESUME']:
                self.geneticGeneration = 0
                print('Initializing genetic agents.')
                self.initializeGeneticAgents()
                # print(join(self.tempModelpth, self.envSettings['MODELNAME'] + '_hyParams.p'))
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

        if self.envSettings['KEEPMODELHISTORY']:
            chunksTotal = self.yieldChunks(arange(self.hyParams['NAGENTS']),
                self.envSettings['NAGENTSPARALLEL']*self.maxTasksPerWorkerMutate)
            for chunk in chunksTotal:
                _ = self.multiprocessChunks(self.randomAgentGenetic, chunk)
        else:
            self.mutationHistory = {}
            for iAgent in range(self.hyParams['NAGENTS']):
                creationSeed = iAgent+1
                agentNumber = iAgent+1
                # agent = self.createNNModel(self.actionType, seed=creationSeed)
                self.mutationHistory = self.updateMutationHistory(self.mutationHistory, agentNumber, creationSeed, mutationSeeds=[], mutationSeed=None)

    def setSeeds(self):
        """Set seeds."""
        self.SEED = self.envSettings['SEEDAGENT']
        numpySeed(self.SEED)
        randomSeed(self.SEED)
        set_random_seed(self.SEED)

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
            env.reset(MODELNAME=env.MODELNAME) # no need for seed?
            # MODELNAME=MODELNAMETEMP, _seed=SEEDTEMP


            # simulating, updating replay memory and training main network
            self.takeActionsUpdateAndTrainDQN(env)
            if env.success:
                self.gameReward = self.gameReward
            elif env.success == False:
                # overwriting simulation memory to zero if no success
                # to test: is it better to give reward an not reset to 0?
                # self.gameReward = 0.0
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

    class noveltyArchiver():

        def __init__(self):
            pass

        def prepareResume(self):
            pass

    def runGenetic(self, env, searchNovelty=False):
        """Run main pipeline for genetic agent optimisation.
        # Inspiration and larger parts of code modified after and inspired by:
        # https://github.com/paraschopra/deepneuroevolution
        # https://arxiv.org/abs/1712.06567
        """

        self.actionType = env.actionType

        # if self.hyParams['NOVELTYSEARCH'] and not self.envSettings['KEEPMODELHISTORY']:
        #     raise ValueError('Settings and hyperparameters require changes: ' + \
        #                      'If model history is not kept, novelty search cannot ' + \
        #                      'be performed as it updates novelty archive regularly ' + \
        #                      'and might require loading agents of previous generations.')

        if self.hyParams['NAGENTS'] <= self.hyParams['NNOVELTYELITES']:
            raise ValueError('Settings and hyperparameters require changes: ' + \
                             'The number of novelty elites considered during novelty search ' + \
                             'should be lower than the number of agents considered ' + \
                             'to evolve.')

        # setting environment and number of games
        self.env, n = env, self.hyParams['NGAMESAVERAGED']
        if searchNovelty:
            self.searchNovelty, self.noveltyArchive = searchNovelty, {}
            self.noveltyArchiver_ = self.noveltyArchiver()
            self.noveltyItemCount = 0
            self.agentsUnique, self.agentsUniqueIDs = [], []
            self.agentsDuplicate, self.agentsDuplicateID = [], []
            if env.actionType == 'discrete':
                self.actionsUniqueIDMapping = defaultdict(count().__next__)
        cores = self.envSettings['NAGENTSPARALLEL']
        # generating unique process ID from system time
        self.pid = str(uuid4())
        self.pidList = list(self.pid)
        shuffle(self.pidList)
        self.pid = ''.join(self.pidList)

        agentCounts = [iAgent for iAgent in range(self.hyParams['NAGENTS'])]
        self.rereturnChildrenGenetic = False
        for self.geneticGeneration in range(self.hyParams['NGENERATIONS']):
            self.flagSkipGeneration = False
            self.generatePathPrefixes()

            if self.envSettings['RESUME']:
                if self.searchNovelty:
                    if self.geneticGeneration > 0:
                        self.noveltyArchive = self.pickleLoad(join(
                            self.tempPrevModelPrefix + '_noveltyArchive.p'))
                        if not self.envSettings['KEEPMODELHISTORY']:
                            self.mutationHistory = self.pickleLoad(join(
                                self.tempPrevModelPrefix + '_mutationHistory.p'))
                        self.noveltyItemCount = len(self.noveltyArchive.keys())
                sortedParentIdxs, continueFlag, breakFlag = self.resumeGenetic()
                if continueFlag: continue
                if breakFlag: break

                if self.searchNovelty:
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
                                # self.agentsDuplicateID.apppend()

            print('########## started generation ' + str(self.geneticGeneration+1).zfill(self.zFill) + ' ##########')
            print('')

            # simulating agents in environment, returning average of n runs
            self.rewards = self.runAgentsRepeatedlyGenetic(agentCounts, n, env)
            print('lowest reward', min(self.rewards))
            print('average reward', mean(self.rewards))
            print('highest reward', max(self.rewards))
            if self.envSettings['SEEDSRANDOM']:
                self.bestRewards.append(max(self.rewards))
                print('highest sliding reward (n=50)', mean(self.bestRewards[-50:]))

            # sorting by rewards in reverse, starting with indices of top reward
            # https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
            sortedParentIdxs = argsort(
                self.rewards)[::-1][:self.hyParams['NAGENTELITES']]
            self.bestAgentReward = self.rewards[sortedParentIdxs[0]]
            self.pickleDump(join(self.tempModelPrefix +
                '_agentsSortedParentIndexes.p'), sortedParentIdxs)

            if self.searchNovelty:
                print('Performing novelty search')
                t0PreparingNoveltySearch = time()
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
                    resultsFile = tempAgentPrefix + '_results.p'

                    if env.actionType == 'discrete':
                        pth = join(tempAgentPrefix + '_results.p')
                        actions = self.pickleLoad(pth)['actions']
                        actionsAll = [action for actions_ in actions for action in actions_]
                        actionsUniqueID = self.actionsUniqueIDMapping[tuple(actionsAll)]
                        # https://stackoverflow.com/questions/38291372/assign-unique-id-to-list-of-lists-in-python-where-duplicates-get-the-same-id
                        self.noveltyArchive[agentStr]['actionsUniqueID'] = actionsUniqueID
                    self.noveltyArchive[agentStr]['itemID'] = itemID
                    self.noveltyArchive[agentStr]['modelFile'] = modelFile
                    self.noveltyArchive[agentStr]['resultsFile'] = resultsFile
                    # self.noveltyArchive[agentStr]['actions'] = actions

                    if env.actionType == 'discrete':
                        # if not self.noveltyItemCount > self.hyParams['NNOVELTYNEIGHBORS']:
                        # removing duplicate novelty calculation only in case neighbour limit is not reached
                        # as otherwise the same novelty might not apply
                        if actionsUniqueID not in self.agentsUniqueIDs:
                            # checking if unique ID from actions already exists
                            self.agentsUnique.append(k)
                            self.agentsUniqueIDs.append(actionsUniqueID)
                        else:
                            self.agentsDuplicate.append(k)

                        # is this necessary?

                        # else:
                        #     self.agentsUnique, self.agentsUniqueIDs, self.agentsDuplicate = [], [], []
                        #     # otherwise computed as if unique to avoid assigning novelty
                        #     # from duplicates with different nearest neighbours
                        #     for iNov in range(self.noveltyItemCount+1):
                        #         self.agentsUnique.append(iNov)
                        #         self.noveltyArchive['agent' + str(iNov+1)]['actionsUniqueID'] = iNov
                        #         # self.agentsUniqueIDs.append(iNov)
                    elif env.actionType == 'continuous':
                            self.agentsUnique, self.agentsUniqueIDs, self.agentsDuplicate = [], [], []
                            for iNov in range(self.noveltyItemCount+1):
                                self.agentsUnique.append(iNov)
                                self.noveltyArchive['agent' + str(iNov+1)]['actionsUniqueID'] = iNov
                    self.noveltyItemCount += 1
                tPreparingNoveltySearch = time() - t0PreparingNoveltySearch
                print('Preparing novelty search took', tPreparingNoveltySearch, 's')

                if not self.noveltyItemCount > self.hyParams['NNOVELTYNEIGHBORS']:
                    print('Novelty search:', len(self.agentsUnique), 'unique agents', len(self.agentsDuplicate), 'duplicate agents')
                else:
                    print('Novelty search (neighbor level reached):', len(self.agentsUnique), 'unique agents', len(self.agentsDuplicate), 'duplicate agents')
                # updating novelty of unique agents
                # Note: This can become a massive bottleneck with increasing
                # number of stored agent information and generations
                # despite parallelization
                noveltiesUniqueAgents, updateFlagsUniqueAgents = [], []

                # # loading all actions
                # # this can be improved by loading only once and then saving to disk between generations
                # t0LoadActionsOnce = time()
                # sharedArrayActions = self.loadActions()
                # tLoadActionsOnce = time() - t0LoadActionsOnce
                # print('Loading actions took', tLoadActionsOnce, 's')

                t0NoveltySearch = time()
                # args = [iAgent for iAgent in self.agentsUnique]
                self.neighborLimitReached = (len(self.agentsUnique) > self.hyParams['NNOVELTYNEIGHBORS'])

                t0CalcNoveltyArgs = time()
                args = []

                for iAgent in range(len(self.agentsUnique)):
                    if self.neighborLimitReached:
                        # includes identification of iAgents needing novelty update
                        rangeLower, rangeHigher, iAgentInCroppedArray, needsUpdate = self.calculateNoveltyNeighborBounds(iAgent)

                        # load only actions needing update here?
                        # so replace sharedArrayActions with a load function
                        # other approaches will still blow up memory

                        # arr = self.loadActions(agents=list(range(rangeLower, rangeHigher)))
                        # print('DEBUG LOADED ACTIONS SEPARATELY', self.neighborLimitReached)

                        # ranges were mapped these to indices considering uniques
                        # arr = sharedArrayActions[rangeLower:rangeHigher]
                    else:
                        needsUpdate = True
                        rangeLower, rangeHigher, iAgentInCroppedArray = 0, len(self.agentsUnique), iAgent
                        # arr = self.loadActions(agents=list(range(rangeLower, rangeHigher))
                        # print('DEBUG LOADED ACTIONS SEPARATELY', self.neighborLimitReached)
                        # arr = sharedArrayActions

                    # replace arr with generator reference?
                    if needsUpdate:
                        # print(arr, 'iAgent', iAgent, 'rangeLower', rangeLower, 'rangeHigher', rangeHigher)
                        # print('iAgent', iAgent, 'rangeLower', rangeLower, 'rangeHigher', rangeHigher)
                        # print('---')
                        args.append([iAgent, None, rangeLower, rangeHigher, iAgentInCroppedArray])
                        # args.append([iAgent, arr, rangeLower, rangeHigher, iAgentInCroppedArray])
                    updateFlagsUniqueAgents.append(needsUpdate)
                tCalcNoveltyArgs = time()-t0CalcNoveltyArgs
                print('Calculated novelty search arguments, took', int(tCalcNoveltyArgs), 's')

                chunksTotal = self.yieldChunks(args,
                    cores*self.maxTasksPerWorkerNoveltySearch)

                print('Started novelty search ...')
                for chunk in chunksTotal:
                    # sharing dictionary containing actions to avoid loading

                    # those are only the updated novelties
                    noveltiesPerAgent = self.multiprocessChunks(
                        # self.calculateNoveltyPerAgent, chunk, sharedArr=sharedArrayActions)
                        # self.calculateNoveltyPerAgent, chunk, sharedDict=sharedDictActions)
                        self.calculateNoveltyPerAgent, chunk)
                    noveltiesUniqueAgents += noveltiesPerAgent
                # print('DEBUG len(noveltiesUniqueAgents)', len(noveltiesUniqueAgents), 'len(args)', len(args))

                print('Finished novelty search, took', int(time()-t0NoveltySearch), 's')

                # calculating novelty of unique agents
                count_ = 0
                for i, iUniqueAgent in enumerate(self.agentsUnique):
                    agentStr = 'agent' + str(iUniqueAgent+1)
                    # actionsUniqueID = self.noveltyArchive[agentStr]['actionsUniqueID']
                    if updateFlagsUniqueAgents[i]:
                        # novelty = noveltiesUniqueAgents[actionsUniqueID]
                        novelty = noveltiesUniqueAgents[count_]
                        self.noveltyArchive[agentStr]['novelty'] = novelty
                        count_ += 1

                # updating novelty of duplicate agents from existing value
                count_ = 0
                for iDuplicateAgent in self.agentsDuplicate:
                    # finding ID of agent representing duplicate agent's actions
                    agentStr = 'agent' + str(iDuplicateAgent+1)
                    actionsUniqueID = self.noveltyArchive[agentStr]['actionsUniqueID']
                    itemID = self.noveltyArchive[agentStr]['itemID']
                    # print(self.agentsUnique, actionsUniqueID)
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

            if self.geneticGeneration+1 >= self.hyParams['ADDNOVELTYEVERY']:
                print('lowest novelty', min(self.novelties))
                print('average novelty', mean(self.novelties))
                print('highest novelty', max(self.novelties))

            # returning best-performing agents
            self.returnChildrenGenetic(sortedParentIdxs)

            if not self.envSettings['KEEPMODELHISTORY']:
                self.pickleDump(join(self.tempModelPrefix +
                    '_mutationHistory.p'), self.mutationHistory)

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
            # if not self.envSettings['KEEPMODELHISTORY']:
            #     # removing stored agent models of finished generation
            #     # as the storage requirements can be substantial
            #     for agentIdx in range(self.hyParams['NAGENTS']):
            #         remove(join(self.tempModelPrefix + '_agent' +
            #             str(agentIdx + 1).zfill(self.zFill) + '.h5'))

            if self.envSettings['ENVTYPE'] in ['0s-c']:
                self.env = FloPyEnv(
                    self.envSettings['ENVTYPE'],
                    self.envSettings['PATHMF2005'],
                    self.envSettings['PATHMP6'],
                    MODELNAME=self.envSettings['MODELNAME'],
                    flagRender=self.envSettings['RENDER'],
                    NAGENTSTEPS=self.hyParams['NAGENTSTEPS'],
                    nLay=self.envSettings['NLAY'],
                    nRow=self.envSettings['NROW'],
                    nCol=self.envSettings['NCOL'],
                    OBSPREP=self.hyParams['NNTYPE'],
                    initWithSolution=True,
                    PATHMF6DLL=None)

            print('')
            print('########## finished generation ' + str(self.geneticGeneration+1).zfill(self.zFill) + ' ##########')
            print('')

    def loadActions(self, agents=None):

        if agents == None:
            agents = self.agentsUnique

        sharedListActions = []

        # retrieving shape to allow padding with zeros
        # else conversion to array fails with different length of action collections
        # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros/35751834

        # pre-loop to determine the largest shape of actions
        pth = self.noveltyArchive['agent' + str(0+1)]['resultsFile']
        if self.hyParams['NGAMESAVERAGED'] == 1:
            actions = array(self.pickleLoad(pth)['actions'], dtype=object)
            dimShape = len(shape(actions))
        else:
            dimShape = 0
            actions = array(self.pickleLoad(pth)['actions'], dtype=object)
            for iActions in range(len(actions)):
                actions_ = actions[iActions]
                dimShape_ = len(shape(actions_))
                if dimShape_ >= dimShape:
                    dimShape = dimShape_

        maxShape = [0 for _ in range(dimShape)]
        for iAgent in agents:
            agentStr = 'agent' + str(iAgent+1)
            pth = self.noveltyArchive[agentStr]['resultsFile']
            if self.hyParams['NGAMESAVERAGED'] == 1:
                actions = self.pickleLoad(pth)['actions']
            # else:
            #     actions = self.pickleLoad(pth)['actions'][0]
            if self.hyParams['NGAMESAVERAGED'] == 1:
                actions = array(actions, dtype=object)
                for iDim in range(dimShape):
                    if shape(actions)[iDim] > maxShape[iDim]:
                        maxShape[iDim] = shape(actions)[iDim]
            elif self.hyParams['NGAMESAVERAGED'] != 1:
                actions = self.pickleLoad(pth)['actions']
                for iActions in range(shape(actions)[0]):
                    actions_ = array(actions[iActions], dtype=object)
                    # print(iActions, 'shape actions_', shape(actions_)[0])
                    dimShape = len(shape(actions_))
                    for iDim in range(dimShape):
                        if shape(actions_)[iDim] > maxShape[iDim]:
                            maxShape[iDim] = shape(actions_)[iDim]

        if self.hyParams['NGAMESAVERAGED'] != 1:
            maxShape = [self.hyParams['NGAMESAVERAGED']] + maxShape
        maxShape = tuple(maxShape)

        # print('MAXSHAPE', maxShape)

        # replace this with specific agents?
        for iAgent in agents:
        # for iAgent in range(len(agents)):
            agentStr = 'agent' + str(iAgent+1)
            pth = self.noveltyArchive[agentStr]['resultsFile']
            actions = self.pickleLoad(pth)['actions']
            if self.actionType == 'discrete':
                sharedListActions_ = chararray(shape=maxShape, itemsize=10)
                sharedListActions_[:] = 'keep'
                padValue ='keep'
                # padding action lists to same length
                maxLen = maxShape[1]
                for iActions in range(len(actions)):
                    actions[iActions] = actions[iActions] + [padValue] * (maxLen - len(actions[iActions]))
                sharedListActions_[:len(actions)] = actions
            elif self.actionType == 'continuous':
                sharedListActions_ = zeros(maxShape)
                if self.hyParams['NGAMESAVERAGED'] != 1:
                    for iActions in range(len(actions)):
                        a = array(actions[iActions], dtype=object)
                        sharedListActions_[iActions, :a.shape[0], :a.shape[1]] = array(a, dtype=object)
                        # sharedListActions_[iActions, :a.shape[0], :a.shape[1]] = array([a])
                else:
                    a = array(actions, dtype=object)
                    sharedListActions_[:a.shape[0], :a.shape[1]] = a
            sharedListActions.append(sharedListActions_)

        if self.actionType == 'discrete':
            self.actionsUniqueScheme, sharedListActions = unique(sharedListActions, return_inverse=True)
            # sharedListActions = reshape(sharedListActions, tuple([len(agentStr)] + list(maxShape)))
            sharedListActions = reshape(sharedListActions, tuple([len(agents)] + list(maxShape)))
            # recreated = reshape(b[c], tuple([len(sharedListActions)] + list(maxShape)))

            sharedArrayActions = array(sharedListActions, dtype=object)
        if self.actionType == 'continuous':
            sharedArrayActions = array(sharedListActions, dtype=object)

        return sharedArrayActions

    def calculateNoveltyNeighborBounds(self, iAgent):
        """Check if half of NNOVELTYNEIGHBORS are available surrounding the given index,
        if not agents are selected until the index boundary and more from the other end."""

        nAgentsUnique = len(self.agentsUnique)

        # defining how many neighbors to consider below and above
        # will be asymmetric in case of odd NNOVELTYNEIGHBORS
        nLower = int(floor(self.hyParams['NNOVELTYNEIGHBORS']/2))
        nHigher = int(ceil(self.hyParams['NNOVELTYNEIGHBORS']/2))
        bottomReached, topReached = False, False
        if iAgent - nLower >= 0:
            rangeLower = iAgent - nLower
            rangeHigher = iAgent + nHigher
        else:
            bottomReached = True
            rangeLower, rangeHigher = 0, self.hyParams['NNOVELTYNEIGHBORS']

        if not bottomReached:
            # if iAgent + nHigher <= self.noveltyItemCount:
            if iAgent + nHigher <= nAgentsUnique:
                rangeLower = iAgent - nLower
                rangeHigher = iAgent + nHigher
            else:
                topReached = True
                rangeLower = nAgentsUnique - self.hyParams['NNOVELTYNEIGHBORS']
                rangeHigher = nAgentsUnique
        if not bottomReached and not topReached:
            iAgentInCroppedArray = iAgent - rangeLower
        elif bottomReached:
            iAgentInCroppedArray = iAgent
        elif topReached:


            # DO UNIT TESTING IF THIS YIELDS CORRECT NOVELTIES

            iAgentInCroppedArray = iAgent+1-nAgentsUnique + nHigher-1
        else:
            print('Error iAgentInCroppedArray could not be defined. This is a bug, please report.')

        # checking if agent needs updating given the setting of NNOVELTYNEIGHBORS to consider
        # considering NAGENTS as an edge case avoids to look up existing novelty when not yet existing
        # nMaxUpdates = max([self.hyParams['NAGENTS'], self.hyParams['NNOVELTYNEIGHBORS']])
        nMaxUpdates = self.hyParams['NAGENTS'] + self.hyParams['NNOVELTYNEIGHBORS']
        # nMaxUpdates = max([self.hyParams['NAGENTS'], self.hyParams['NNOVELTYNEIGHBORS']])
        if (nAgentsUnique - (iAgent+1)) >= nMaxUpdates:
            needsUpdate = False
        else:
            needsUpdate = True
        # print('DEBUG nMaxUpdates, nAgentsUnique, iAgent, nAgentsUnique - iAgent', nMaxUpdates, nAgentsUnique, iAgent, nAgentsUnique - (iAgent+1))
        # print('needsUpdate', needsUpdate)

        return rangeLower, rangeHigher, iAgentInCroppedArray, needsUpdate

    def createNNModel(self, actionType, seed=None):
        """Create neural network."""
        if seed is None:
            seed = self.SEED
        model = Sequential()
        if actionType == 'discrete':
            # initializer = glorot_normal(seed=seed)
            # initializer = glorot_uniform(seed=seed)
            initializer = random_uniform(minval=-2.2, maxval=2.2, seed=seed)
            # initializer = random_normal(seed=seed)
        elif actionType == 'continuous':
            initializer = random_uniform(minval=-2.5, maxval=2.5, seed=seed)
            # initializer = random_normal(minval=-2.5, maxval=2.5, seed=seed)
        else:
            print('Chosen normal glorot initializer, as not specified.')
            initializer = glorot_normal(seed=seed)

        nHiddenNodes = copy(self.hyParams['NHIDDENNODES'])
        # resetting numpy seeds to generate reproducible architecture
        numpySeed(seed)

        if self.hyParams['NNTYPE'] == 'perceptron':
            # fully-connected feed-forward multi-layer neural network
            # applying architecture (variable number of nodes per hidden layer)
            if self.agentMode == 'genetic' and self.hyParams['ARCHITECTUREVARY']:
                for layerIdx in range(len(nHiddenNodes)):
                    nHiddenNodes[layerIdx] = randint(2, self.hyParams['NHIDDENNODES'][layerIdx]+1)
            for layerIdx in range(len(nHiddenNodes)):
                inputShape = shape(self.observationsVector) if layerIdx == 0 else []
                model.add(Dense(units=nHiddenNodes[layerIdx],
                    input_shape=inputShape,
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    use_bias=True))
                if self.hyParams['BATCHNORMALIZATION']:
                    model.add(BatchNormalization())
                model.add(Activation(self.hyParams['HIDDENACTIVATIONS'][layerIdx]))
                if 'DROPOUTS' in self.hyParams:
                    if self.hyParams['DROPOUTS'][layerIdx] != 0.0:
                        model.add(Dropout(self.hyParams['DROPOUTS'][layerIdx]))

        elif self.hyParams['NNTYPE'] == 'convolution':
            # convolutional feed-forward neural network
            
            inputShape = shape(self.observationsVector)
            model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                             activation='relu',
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             input_shape=inputShape, padding='same'))
            # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2),
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             activation='relu', padding='same'))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             activation='relu', padding='same'))
            model.add(Flatten())
            model.add(Dense(512,
                      kernel_initializer=initializer,
                      bias_initializer=initializer,
                      activation='relu'))
            
        # adding output layer
        if actionType == 'discrete':
            model.add(Dense(self.actionSpaceSize, activation='linear', # 'softmax'
                kernel_initializer=initializer))
        elif actionType == 'continuous':
            # sigmoid used here as actions are predicted as fraction of actionRange
            model.add(Dense(self.actionSpaceSize, activation='sigmoid',
                kernel_initializer=initializer))

        # compiling to avoid warning while saving agents in genetic search
        # specifics are irrelevant, as genetic models are not optimized
        # along gradients
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001),
                      metrics=['mean_squared_error'])

        return model

    def checkParameterMean(self, agent):
        
        weights, means = agent.get_weights(), []
        for iParam, parameters in enumerate(weights):
            means.append(mean(weights[iParam]))
        
        return mean(means)

    def loadModelFromMutationHistory(self, creationSeed, mutationSeeds):

        agent = self.createNNModel(self.actionType, seed=creationSeed)

        for mutationSeed in mutationSeeds:
            agent = self.mutateGenetic(agent, mutationSeed)
        
        return agent

    def updateMutationHistory(self, mutationHistory, agentNumber, creationSeed, mutationSeeds=[], mutationSeed=None):

        mutationHistory['agent' + str(agentNumber)] = {}
        mutationHistory['agent' + str(agentNumber)]['creationSeed'] = creationSeed
        mutationHistory['agent' + str(agentNumber)]['mutationSeeds'] = mutationSeeds
        if mutationSeed != None:
            mutationHistory['agent' + str(agentNumber)]['mutationSeeds'].append(mutationSeed)
        
        return mutationHistory

    def createMutationRecord(self, creationSeed, mutationSeeds=[], mutationSeed=None):

        record = {}
        record['creationSeed'] = creationSeed
        record['mutationSeeds'] = mutationSeeds
        if mutationSeed != None:
            record['mutationSeeds'].append(mutationSeed)
        
        return record

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

            new_state, reward, done, info = env.step(action)
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
                if env.ENVTYPE in ['0s-c']:
                    env.teardown()
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
                new_state, reward, done, info = env.step(action)
                self.gameReward += reward
                if self.envSettings['RENDER']:
                    if not iGame % self.envSettings['RENDEREVERY']:
                        if not done: env.render()
                current_state = new_state
                step += 1
                if done:
                    if env.ENVTYPE in ['0s-c']:
                        env.teardown()
                    break

            # if not env.success:
            #     self.gameReward = 0.0
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

        # if __name__ == 'FloPyArcade':
        if 'flopyarcade' in __name__:
            if self.envSettings['SURROGATESIMULATOR'] is not None:
                # removing environment in case of surrogate model
                # as TensorFlow model cannot be pickled
                if hasattr(self, 'env'):
                    del self.env

            t0Runs = time()
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
            print('------------------------------')
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

                if self.envSettings['ENVTYPE'] in ['0s-c']:
                    # necessary as some ctypes object related to the dll used with BMI is not pickleable
                    try:
                        self.env.teardown()
                    except:
                        pass
                    try:
                        del self.env
                    except:
                        pass

                reward_chunks = self.multiprocessChunks(
                    self.runAgentsGeneticSingleRun, chunk)
                reward_agents += reward_chunks
                runtimes.append(time() - t0)
                nChunksRemaining -= 1
            tRuns = time()-t0Runs
            print('Processing agents took', tRuns, 's')

        return reward_agents

    def runAgentsGeneticSingleRun(self, agentCount):
        """Run single game within genetic agent optimisation."""

        # print('debug running agentCount', agentCount)
        # import os
        # os.system("taskset -p 0xfffff %d" % os.getpid())

        tempAgentPrefix = join(self.tempModelPrefix + '_agent'
            + str(agentCount + 1).zfill(self.zFill))

        # loading specific agent and weights with given ID
        t0load_model = time()


        if self.envSettings['KEEPMODELHISTORY']:
            agent = load_model(join(tempAgentPrefix + '.h5'), compile=False)

        
        else:
            t0RecreateFromMutationHistory = time()
            agentOffset = self.hyParams['NAGENTS'] * (self.geneticGeneration)
            # print(self.mutationHistory)
            agentNumber = agentCount + 1 + agentOffset
            creationSeed = self.mutationHistory['agent' + str(agentNumber)]['creationSeed']
            mutationSeeds = self.mutationHistory['agent' + str(agentNumber)]['mutationSeeds']
            agent = self.loadModelFromMutationHistory(creationSeed, mutationSeeds)
            tRecreateFromMutationHistory = (self.hyParams['NAGENTS']/self.envSettings['NAGENTSPARALLEL']) * (t0RecreateFromMutationHistory - time())
            if agentCount == 1:
                print('model recreation took about', tRecreateFromMutationHistory, 's')
            # print('debug duration load_model compiled', time() - t0load_model)

        MODELNAMETEMP = ('Temp' + self.pid +
            '_' + str(agentCount + 1))

        iGame = self.currentGame-1
        if self.envSettings['SEEDSRANDOM']:
            SEEDTEMP = self.gamesSeeds[self.geneticGeneration][iGame]
        else:
            SEEDTEMP = self.envSettings['SEEDENV'] + iGame
        if self.envSettings['SURROGATESIMULATOR'] is None:
            if self.envSettings['ENVTYPE'] in ['0s-c']:

                lenCount = len(str(agentCount + 1))

                MODELNAMETEMP = self.pid[0:15-lenCount] + str(agentCount + 1)

                MODELNAMETEMP_ = list(MODELNAMETEMP)
                shuffle(MODELNAMETEMP_)
                MODELNAMETEMP = ''.join(MODELNAMETEMP_)

                # print(self.envSettings['ENVTYPE'], self.envSettings['PATHMF2005'], self.envSettings['PATHMP6'],
                #     MODELNAMETEMP, SEEDTEMP, self.envSettings['RENDER'], self.hyParams['NAGENTSTEPS'],
                #     self.envSettings['NLAY'], self.envSettings['NROW'], self.envSettings['NCOL'],
                #     self.hyParams['NNTYPE'])

                # loading anew with BMI mode, as environment was not pickleable
                env = FloPyEnv(
                    self.envSettings['ENVTYPE'],
                    self.envSettings['PATHMF2005'],
                    self.envSettings['PATHMP6'],
                    MODELNAME=MODELNAMETEMP,
                    _seed=SEEDTEMP,
                    flagRender=self.envSettings['RENDER'],
                    NAGENTSTEPS=self.hyParams['NAGENTSTEPS'],
                    nLay=self.envSettings['NLAY'],
                    nRow=self.envSettings['NROW'],
                    nCol=self.envSettings['NCOL'],
                    OBSPREP=self.hyParams['NNTYPE'],
                    initWithSolution=True,
                    PATHMF6DLL=None)

            else:
                env = self.env
            # env = deepcopy(self.env)
            # if env.ENVTYPE in ['0s-c']
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
            new_observation, reward, done, info = env.step(action)
            # print('debug duration step', time() - t0step)
            actions[self.currentGame-1].append(action)
            rewards[self.currentGame-1].append(reward)
            if not env.ENVTYPE in ['0s-c']:
                wellCoords[self.currentGame-1].append(env.wellCoords)
            r += reward
            if self.envSettings['RENDER']:
                env.render()

            if done or (step == self.hyParams['NAGENTSTEPS']-1): # or if reached end
                # print('debug duration game', time() - t0game)
                # if env.success == False:
                #     r = 0
                # saving specific simulation results pertaining to agent
                if not env.ENVTYPE in ['0s-c']:
                    trajectories[self.currentGame-1].append(env.trajectories)
                objects = [trajectories, actions, rewards, wellCoords]
                for i, objectCurrent in enumerate(objects):
                    results[keys[i]] = objectCurrent
                pth = join(tempAgentPrefix + '_results.p')
                self.pickleDump(pth, results)

                if env.ENVTYPE in ['0s-c']:
                    env.teardown()

                break
        # print('time game', time()-t0game, 'time step', (time()-t0game)/step, 'reward', r, 'steps', step)

        # print('debug reward agentCount', r, step, actions, rewards)

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
            # print(rewardsAgentsCurrent)
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

        # t0 = time()
        actionType = FloPyEnv(initWithSolution=False).getActionType(self.envSettings['ENVTYPE'])
        agent = self.createNNModel(actionType, seed=self.envSettings['SEEDAGENT']+agentIdx)
        agent.save(join(self.tempModelpth, self.envSettings['MODELNAME'] +
            '_gen' + str(generation).zfill(self.zFill) + '_agent' +
            str(agentIdx + 1).zfill(self.zFill) + '.h5'))
        # print(agentIdx, 'save', time()-t0)

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

        if self.searchNovelty:
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

        if self.envSettings['KEEPMODELHISTORY']:
            bestAgent = load_model(join(self.tempModelPrefix + '_agent' +
                str(sortedParentIdxs[0] + 1).zfill(self.zFill) + '.h5'),
                compile=False)
        else:
            agentOffsetBest = self.hyParams['NAGENTS'] * (self.geneticGeneration)
            agentNumberBest = sortedParentIdxs[0] + 1 + agentOffsetBest
            creationSeedBest = self.mutationHistory['agent' + str(agentNumberBest)]['creationSeed']
            mutationSeedsBest = self.mutationHistory['agent' + str(agentNumberBest)]['mutationSeeds']
            bestAgent = self.loadModelFromMutationHistory(creationSeedBest, mutationSeedsBest)

        if not self.rereturnChildrenGenetic:
            bestAgent.save(join(self.tempModelPrefix + '_agentBest.h5'))
        if generation < self.hyParams['NGENERATIONS']:
            if self.envSettings['KEEPMODELHISTORY']:
                bestAgent.save(join(tempNextModelPrefix + '_agent' +
                    str(self.hyParams['NAGENTS']).zfill(self.zFill) + '.h5'))
            nAgentElites = self.hyParams['NAGENTELITES']
            nNoveltyAgents = self.hyParams['NNOVELTYAGENTS']
            self.candidateParentIdxs = sortedParentIdxs[:nAgentElites]

            records = []
            chunksTotal = self.yieldChunks(arange(self.hyParams['NAGENTS']-1),
                self.envSettings['NAGENTSPARALLEL']*self.maxTasksPerWorkerMutate)
            for chunk in chunksTotal:
                records += self.multiprocessChunks(self.returnChildrenGeneticSingleRun,
                    chunk)

            if not self.envSettings['KEEPMODELHISTORY']:
                agentOffset = self.hyParams['NAGENTS'] * (self.geneticGeneration+1)
                # agentNumber = sortedParentIdxs[0] + 1 + agentOffset
                # storing updates to mutation history
                for iRecord, record in enumerate(records):
                    self.mutationHistory['agent' + str(iRecord+1+agentOffset)] = record
                # saving best agent to mutation history
                creationSeed = self.mutationHistory['agent' + str(agentNumberBest)]['creationSeed']
                mutationSeeds = self.mutationHistory['agent' + str(agentNumberBest)]['mutationSeeds']
                recordBest = self.createMutationRecord(creationSeed, mutationSeeds, None)
                self.mutationHistory['agent' + str(self.hyParams['NAGENTS']+agentOffset)] = recordBest

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

        if self.searchNovelty:
            if ((self.geneticGeneration+1) % self.hyParams['ADDNOVELTYEVERY']) == 0:
                remainingElites = self.hyParams['NAGENTS'] - (childIdx+1)
                if self.rereturnChildrenGenetic:
                    generation = self.geneticGeneration
                else:
                    generation = self.geneticGeneration + 1
                if (childIdx+1 ==
                    # remainingElites - self.hyParams['NNOVELTYELITES']):
                    remainingElites - self.hyParams['NNOVELTYAGENTS']):

                    print('Performing novelty evolution after generation',
                        generation)
                # if remainingElites <= self.hyParams['NNOVELTYELITES']:
                if remainingElites <= self.hyParams['NNOVELTYAGENTS']:
                    # selecting a novelty parent randomly
                    len_ = len(self.candidateNoveltyParentIdxs)
                    selected_agent_index = self.candidateNoveltyParentIdxs[randint(len_)]
                    agentPth = self.noveltyFilenames[selected_agent_index]
                if remainingElites == 1:
                    # Keeping agent with highest novelty
                    selected_agent_index = self.candidateNoveltyParentIdxs[int(
                        remainingElites)-1]
                    agentPth = self.noveltyFilenames[selected_agent_index]

        # loading given parent agent, current with retries in case of race
        # condition: https://bugs.python.org/issue36773
        success = False
        while not success:
            try:
                if self.envSettings['KEEPMODELHISTORY']:
                    agent = load_model(agentPth,
                        compile=False)
                else:
                    # agentOffset = self.hyParams['NAGENTS'] * (self.geneticGeneration)
                    # agentNumber = selected_agent_index + 1 + agentOffset
                    agentNumber = selected_agent_index + 1
                    # print('debug selected_agent_index', selected_agent_index, 'agentNumber', agentNumber)
                    creationSeed = self.mutationHistory['agent' + str(agentNumber)]['creationSeed']
                    mutationSeeds = self.mutationHistory['agent' + str(agentNumber)]['mutationSeeds']
                    agent = self.loadModelFromMutationHistory(creationSeed, mutationSeeds)
                success = True
            except Exception as e:
                success = False
                print('Retrying loading parent agent, possibly due to lock.')
                print(e)
                sleep(1)

        # activating later part makes mutationSeed different every run (not reproducible)
        mutationSeed = self.mutationSeeds[self.geneticGeneration][childIdx]
        # mutationSeed = selected_agent_index+1 + randint(0, 1000000000)
        # print('childIdx', childIdx, 'mutationSeed', mutationSeed)
        # altering parent agent to create child agent
        childrenAgent = self.mutateGenetic(agent, childIdx)

        if self.envSettings['KEEPMODELHISTORY']:
            childrenAgent.save(join(self.tempNextModelPrefix +
                '_agent' + str(childIdx + 1).zfill(self.zFill) + '.h5'))
            return ''
        else:
            creationSeed = self.mutationHistory['agent' + str(agentNumber)]['creationSeed']
            mutationSeeds = self.mutationHistory['agent' + str(agentNumber)]['mutationSeeds']
            record = self.createMutationRecord(creationSeed, mutationSeeds, mutationSeed)
            return record

    def countParameters(self, agent):
        """Count number of policy model parameters."""

        parameters = agent.get_weights()
        nParameters = 0
        for iParam, parameters in enumerate(parameters):
            nParameters += parameters.size

        return nParameters

    def mutateGenetic(self, agent, seed):
        """Mutate single agent model.
        Mutation power is a hyperparameter. Find example values at:
        https://arxiv.org/pdf/1712.06567.pdf
        """

        numpySeed(seed)

        mProb = self.hyParams['MUTATIONPROBABILITY']
        mPower = self.hyParams['MUTATIONPOWER']
        parameters = agent.get_weights()
        for iParam, parameters_ in enumerate(parameters):
            randn_ = mPower * randn(*shape(parameters_))
            mutateDecisions_ = self.mutateDecisions(mProb, shape(parameters_))
            permutation = randn_*mutateDecisions_
            parameters[iParam] = add(parameters_, permutation)
        agent.set_weights(parameters)

        return agent

    def mutateDecision(self, probability):
        """Return boolean defining whether to mutate or not."""

        return random() < probability

    def mutateDecisions(self, probability, shape):
        """Return boolean defining whether to mutate or not."""

        return random(shape) < probability

    def getqsGivenAgentModel(self, agentModel, state):
        """ Query given model for Q values given observations of state
        """
        # predict_on_batch robust in parallel operation?

        # t0 = time()
        state = array(state).reshape(-1, (*shape(state)))
        prediction = agentModel.predict_on_batch(state)[0]
        # if max(prediction) >= 1.0 or min(prediction) < 0.0:
        #     print('debug prediction', min(prediction), max(prediction))
        # tPrediction = time()-t0
        # print('debug time prediction', tPrediction)
        return prediction

    def loadAgentModel(self, modelNameLoad=None, compiled=False):
        """Load an agent model."""

        # Note: change model load and save as json mode is faster
        # this is a temporary quickfix

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
            if not self.envSettings['KEEPMODELHISTORY']:
                self.mutationHistory = self.pickleLoad(join(self.tempModelPrefix + '_mutationHistory.p'))
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

                    if self.envSettings['ENVTYPE'] in ['0s-c']:
                        # necessary as some ctypes object related to the dll used with BMI is not pickleable
                        try:
                            self.env.teardown()
                        except:
                            pass
                        try:
                            del self.env
                        except:
                            pass

                    self.rereturnChildrenGenetic = True
                    self.returnChildrenGenetic(self.sortedParentIdxs)
                    self.rereturnChildrenGenetic = False
            elif not self.envSettings['KEEPMODELHISTORY']:
                print('Resuming impossible with missing model history.')
                breakFlag = True
            # changing resume flag if resuming
            self.envSettings['RESUME'] = False

        return self.sortedParentIdxs, continueFlag, breakFlag

    def actionNoveltyMetric(self, actions1, actions2, actionType):
        # finding largest object, or determining equal length
        # do this conversion while loading or saving?
        actions1 = array(actions1)
        actions2 = array(actions2)

        if actionType == 'discrete':
            if len(actions1) > len(actions2):
                shorterObj, longerObj = actions2, actions1[:len(actions2)]
            elif len(actions1) <= len(actions2):
                shorterObj, longerObj = actions1, actions2[:len(actions1)]
            diffsCount = len(shorterObj) - numpySum(1 for x, y in zip(longerObj, shorterObj) if x == y)

            # enabling this might promote agents having acted longer but not
            # too different to begin with
            # diffsCount += float(numpyAbs(len(longerObj) - len(shorterObj)))

            # dividing by the length of it, to avoid rewarding longer objects
            novelty = diffsCount/len(shorterObj)

        elif actionType == 'continuous':
            # there are values per action in the action space

            # ln = min([len(actions1), len(actions2)])
            # nActionLists = len(actions1[0])
            # diffsCount = ln*nActionLists
            # actions1_, actions2_ = [], []
            # for iStep in range(ln):
            #     actions1_.append(actions1[iStep])
            #     actions2_.append(actions2[iStep])
            # actions1 = list(chain.from_iterable(actions1_))
            # actions2 = list(chain.from_iterable(actions2_))
            # diffs = numpySum(numpyAbs(subtract(actions1, actions2)))
            # novelty = diffs/diffsCount

            # alternative way
            ln = min([len(actions1[:,0]), len(actions2[:,0])])
            nActionLists = len(actions1)
            diffsCount = ln*nActionLists
            diffs = numpySum(numpyAbs(subtract(
                actions1[:ln,:].flatten(),
                actions2[:ln,:].flatten())
                ))
            novelty = diffs/diffsCount

        # print('novelty', novelty)
        return novelty

    def calculateNoveltyPerPair(self, args):
        # agentStr = 'agent' + str(args[0]+1)
        # agentStr2 = 'agent' + str(args[1]+1)
        # actionsDict = args[2]

        # actions = actionsDict[agentStr]['actions']
        # actions2 = actionsDict[agentStr2]['actions']

        actions = args[0]
        actions2 = args[1]

        # loading per agent becomes a real bottleneck if done repeatedly
        # better to know which needed beforehand
        # tempAgentPrefix = self.noveltyArchive[agentStr]['modelFile'].replace('.h5', '')
        # tempAgentPrefix2 = self.noveltyArchive[agentStr2]['modelFile'].replace('.h5', '')
        # pth = join(tempAgentPrefix + '_results.p')
        # pth2 = join(tempAgentPrefix2 + '_results.p')
        # actions = self.pickleLoad(pth)['actions']
        # actions2 = self.pickleLoad(pth2)['actions']

        # having shared this object across processes will increase used memory too much
        # actions = self.noveltyArchive[agentStr]['actions']
        # actions2 = self.noveltyArchive[agentStr2]['actions']

        novelty = 0.
        for g in range(len(actions)):
            novelty += self.actionNoveltyMetric(actions[g],
                actions2[g], self.actionType)

        return novelty

    def calculateNoveltyPerAgent(self, iAgent, actionsDict=None):

        # args.append([iAgent, sharedArrayActions[rangeLower:rangeHigher], rangeLower, rangeHigher, iAgentInCroppedArray])
        iAgentOriginal, arr, rangeLower, rangeHigher, iAgentInCroppedArray = iAgent[0], iAgent[1], iAgent[2], iAgent[3], iAgent[4]

        arr = self.loadActions(agents=list(range(rangeLower, rangeHigher)))
        # print(arr, 'self.neighborLimitReached', self.neighborLimitReached)

        agentStr = 'agent' + str(iAgentOriginal+1)
        iAgent = iAgentInCroppedArray

        t0 = time()
        # loading noveties for specific agent from disk
        noveltyFile = join(self.tempNoveltypth, agentStr + '_novelties.p')
        if exists(noveltyFile):
            agentNovelties = self.pickleLoad(noveltyFile, compressed=None)
        else:
            agentNovelties = {}

        # global arr, arr_lock
        # with arr_lock:

        # print('loading novelties took', time()-t0)

        # determine which ones need update?
        # pass necessary actionsDict?
        # load only parts of the noveltyArchive

        # load all actions in main process
        # is this not the last batch missing?
        # collect iAgent2 beforehand and pass corresponding actions in actionsDict
        # load large noveltyArchive
        # iActionsNeeded = []
        # select actions from iActionsNeeded
        # dump the rest of the novelty Archive
        # or use these indices to request from main process?

        # i1 = self.agentsUnique.index(iAgent)
        actions = arr[iAgent]

        # actions = arr[iAgent]
        # actions = actionsDict[agentStr]['actions']

        t0 = time()
        novelties = []
        if not self.neighborLimitReached:
            for iAgent2 in range(len(arr)):
                if iAgent != iAgent2:
                    agentStr2 = 'agent' + str(iAgent2+1)
                    t0GetActions = time()
                    # i2 = self.agentsUnique.index(iAgent2)
                    actions2 = arr[iAgent2]
                    # actions2 = actionsDict[agentStr2]['actions']
                    try:
                        # calculate novelties only if unavailable
                        # as keys mostly exists, try/except check should be performant here
                        novelty = agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+rangeLower+1)]
                    # except:
                    #     novelty = agentNovelties[str(iAgent2+1) + '_' + str(iAgent+1)]
                    except:
                        t0single = time()
                        novelty = self.calculateNoveltyPerPair([actions, actions2])
                        # novelty = self.calculateNoveltyPerPair([iAgent, iAgent2])
                        # print('calculating single took', time()-t0single)

                        agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+rangeLower+1)] = novelty
                    novelties.append(novelty)
            # print('calculating novelties took', time()-t0)

        elif self.neighborLimitReached:
            iArr = 0
            for iAgent2 in range(rangeLower, rangeHigher):
                if iAgentOriginal != iAgent2:
                    agentStr2 = 'agent' + str(iAgent2+1)
                    # actionsUniqueID2 = self.noveltyArchive[agentStr2]['actionsUniqueID']
                    # actions2 = arr[actionsUniqueID2]
                    actions2 = arr[iArr]
                    # actions2 = actionsDict[agentStr2]['actions']
                    try:
                        # calculate novelties only if unavailable
                        # as keys mostly exists, try/except check should be performant here
                        novelty = agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+1)]
                        # print('debug loaded novelty', novelty)
                    # except:
                    #     novelty = agentNovelties[str(iAgent2+1) + '_' + str(iAgent+1)]
                    except:
                        novelty = self.calculateNoveltyPerPair([actions, actions2])
                        # print('debug calculated novelty', novelty)
                        # novelty = self.calculateNoveltyPerPair([iAgent, iAgent2])
                        agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+1)] = novelty
                    novelties.append(novelty)
                iArr += 1

        t0 = time()
        dumpPath = join(self.tempNoveltypth, agentStr + '_novelties.p')
        self.pickleDump(dumpPath, agentNovelties, compress=None)
        # self.pickleDump(join(self.tempNoveltypth,
        #     agentStr + '_novelties.p'), agentNovelties, compress='lz4')
        # print('dumping novelties took', time()-t0)

        t0 = time()
        novelty = mean(novelties)
        # print('calculating mean novelty took', time()-t0)

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
            ENVTYPE = env.ENVTYPE,
            modelName=MODELNAMEGENCOUNT,
            animationFolder=MODELNAME,
            NAGENTSTEPS=self.hyParams['NAGENTSTEPS'],
            PATHMF2005=self.envSettings['PATHMF2005'],
            PATHMP6=self.envSettings['PATHMP6'],
            flagSavePlot=True,
            flagManualControl=False,
            flagRender=False,
            nLay=env.nLay, nRow=env.nRow, nCol=env.nCol,
            OBSPREP=self.hyParams['NNTYPE'])

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

    def init_arr(self, arr, arr_lock=None, shape=None):
        arr = array(list(arr))
        globals()['arr'] = frombuffer(arr, dtype='float64').reshape(shape)
        globals()['arr_lock'] = arr_lock

    # def multiprocessChunks(self, function, chunk, parallelProcesses=None, wait=False):
    #     """Process function in parallel given a chunk of arguments."""

    #     # Pool object from pathos instead of multiprocessing library necessary
    #     # as tensor.keras models are currently not pickleable
    #     # https://github.com/tensorflow/tensorflow/issues/32159
    #     if parallelProcesses == None:
    #         parallelProcesses = self.envSettings['NAGENTSPARALLEL']
    #     p = Pool(processes=parallelProcesses)
    #     pasync = p.map_async(function, chunk)
    #     # waiting is important to order results correctly when running
    #     # -- really?
    #     # in asynchronous mode (correct reward order is validated)
    #     pasync = pasync.get()
    #     if wait:
    #         pasync.wait()
    #     p.close()
    #     p.join()
    #     p.terminate()

    #     return pasync

    def multiprocessChunks(self, function, chunk, parallelProcesses=None, wait=False, async_=True, sharedArr=None, sharedDict=None):
        """Process function in parallel given a chunk of arguments."""

        # Pool object from pathos instead of multiprocessing library necessary
        # as tensor.keras models are currently not pickleable
        # https://github.com/tensorflow/tensorflow/issues/32159
        if parallelProcesses == None:
            parallelProcesses = self.envSettings['NAGENTSPARALLEL']
        # print('debug parallelProcesses', parallelProcesses)

        # manual setting
        # async_ = False


        if sharedDict == None and sharedArr is None:
            p = Pool(processes=parallelProcesses)
            if async_:
                # pasync = p.map_async(function, chunk)
                # pasync = pasync.get()

                # imap to use map with a generator and reduce memory impact
                # chunksize = int(max([int(len(chunk)/parallelProcesses), 1]))
                # pasync = p.imap(function, chunk, chunksize=chunksize)
                # pasync = p.imap(function, chunk)
                pasync = p.map_async(function, chunk)
                pasync = pasync.get()
                # pasync = p.map(function, chunk)
                # print(pasync)
                # pasync = list(pasync)
                try:
                    pasync = list(pasync)
                except Exception as e:
                    print('Error in multiprocessing:', e)
            else:
                pasync = p.map(function, chunk)
                # pasync = pasync.get()
            # waiting is important to order results correctly when running
            # -- really?
            # in asynchronous mode (correct reward order is validated)
            if wait:
                pasync.wait()
            p.close()
            p.join()
            p.terminate()

        if sharedDict != None:
            with pathosHelpers.mp.Manager() as manager:
                sharedMgrDict = manager.dict()
                # https://stackoverflow.com/questions/35353934/python-manager-dict-is-very-slow-compared-to-regular-dict
                sharedMgrDict.update(sharedDict)
                with manager.Pool(processes=parallelProcesses) as p:
                    input_ = zip(chunk, repeat(sharedDict, len(chunk)))
                    pasync = p.starmap(function, input_)

        if sharedArr is not None:
            # https://stackoverflow.com/questions/39322677/python-how-to-use-value-and-array-in-multiprocessing-pool
            # https://stackoverflow.com/questions/64222805/how-to-pass-2d-array-as-multiprocessing-array-to-multiprocessing-pool
            # https://stackoverflow.com/questions/1675766/combine-pool-map-with-shared-memory-array-in-python-multiprocessing/58208695#58208695
            # https://stackoverflow.com/questions/11652288/slower-execution-with-python-multiprocessing
            arr = pathosHelpers.mp.Array('d', sharedArr.flatten(), lock=False)
            arr_lock = pathosHelpers.mp.Lock()
            shape_ = shape(sharedArr)
            # arr = manager.Array('d', sharedArr.flatten(), lock=False)
            # arr_lock = manager.Lock()
            with Pool(processes=parallelProcesses, initializer=self.init_arr, initargs=(arr, arr_lock, shape_)) as p:
            # with Pool(processes=parallelProcesses, initializer=self.init_arr, initargs=(arr, arr_lock, shape_)) as p:
                input_ = zip(chunk, repeat(sharedDict, len(chunk)))
                pasync = p.starmap(function, input_)

        return pasync

    def multiprocessChunks_OLD(self, function, chunk, parallelProcesses=None,
        wait=False, async_=True, sharedArr=None, sharedDict=None, threadPool=False):
        """Process function in parallel given a chunk of arguments."""

        # Pool object from pathos instead of multiprocessing library necessary
        # as tensor.keras models are currently not pickleable
        # https://github.com/tensorflow/tensorflow/issues/32159
        if parallelProcesses == None:
            parallelProcesses = self.envSettings['NAGENTSPARALLEL']

        if sharedDict == None and sharedArr is None:
            if threadPool == False:
                # p = Pool(ncpus=parallelProcesses)
                # p = Pool(nodes=parallelProcesses, processes=len(chunk))
                # manager = pathosHelpers.mp.Manager()
                # p = manager.Pool(processes=parallelProcesses)
                p = Pool(processes=parallelProcesses)
                # p = Pool(parallelProcesses)
                # p = Pool(16)
                # p = Pool(processes=parallelProcesses, nodes=16)
            else:
                p = ThreadPool(processes=parallelProcesses)
            if async_:

                # import os
                # os.system("taskset -p -c %d %d" % (process_idx % cpu_count(), os.getpid()))

                # imap to use map with a generator and reduce memory impact
                # chunksize = int(max([int(len(chunk)/parallelProcesses), 1]))
                # pasync = p.imap(function, chunk, chunksize=chunksize)
                # pasync = p.imap(function, chunk, chunksize=1000000000)
                # try:
                #     pasync = list(pasync)
                #     print('debug past list')
                # except Exception as e:
                #     print('Error in multiprocessing:', e)

                # NOW USING MULTIPROCESSING NOT PATHOS

                pasync = p.map_async(function, chunk) # , chunksize=10)
                pasync = pasync.get()
                # pasync = p.imap(function, chunk)
            else:
                pasync = p.map(function, chunk)
                # print(pasync)
                # pasync = pasync.get()
            # waiting is important to order results correctly when running
            # -- really?
            # in asynchronous mode (correct reward order is validated)
            if wait:
                pasync.wait()
            p.close()
            p.join()
            p.terminate()

        if sharedDict != None:
            with pathosHelpers.mp.Manager() as manager:
                sharedMgrDict = manager.dict()
                # https://stackoverflow.com/questions/35353934/python-manager-dict-is-very-slow-compared-to-regular-dict
                sharedMgrDict.update(sharedDict)
                with manager.Pool(processes=parallelProcesses) as p:
                    input_ = zip(chunk, repeat(sharedDict, len(chunk)))
                    pasync = p.starmap(function, input_)

        if sharedArr is not None:
            # https://stackoverflow.com/questions/39322677/python-how-to-use-value-and-array-in-multiprocessing-pool
            # https://stackoverflow.com/questions/64222805/how-to-pass-2d-array-as-multiprocessing-array-to-multiprocessing-pool
            # https://stackoverflow.com/questions/1675766/combine-pool-map-with-shared-memory-array-in-python-multiprocessing/58208695#58208695
            # https://stackoverflow.com/questions/11652288/slower-execution-with-python-multiprocessing
            arr = pathosHelpers.mp.Array('d', sharedArr.flatten(), lock=False)
            arr_lock = pathosHelpers.mp.Lock()
            shape_ = shape(sharedArr)
            # arr = manager.Array('d', sharedArr.flatten(), lock=False)
            # arr_lock = manager.Lock()
            with Pool(processes=parallelProcesses, initializer=self.init_arr, initargs=(arr, arr_lock, shape_)) as p:
            # with Pool(processes=parallelProcesses, initializer=self.init_arr, initargs=(arr, arr_lock, shape_)) as p:
                input_ = zip(chunk, repeat(sharedDict, len(chunk)))
                pasync = p.starmap(function, input_)

        return pasync

    def pickleLoad(self, path, compressed=None):
        """Load pickled object from file."""
        if compressed == None:
            with open(path, 'rb') as f:
                objectLoaded = load(f)
        elif compressed in ['lz4', 'lzma']:
            objectLoaded = joblibLoad(path)
        else:
            print('Unknown compression algorithm specified.')

        return objectLoaded

    def pickleDump(self, path, objectToDump, compress=None, compressLevel=3):
        """Store object to file using pickle."""
        if compress == None:
            with open(path, 'wb') as f:
                dump(objectToDump, f)
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

import gym

class FloPyEnv(gym.Env):
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
    
    # necessary to render with rllib
    # https://discourse.aicrowd.com/t/how-to-save-rollout-video-render/3246/9
    metadata = {
        "render.modes": ["human", "rgb_array"]
      }

    def __init__(self,
                 ENVTYPE='1s-d', PATHMF2005=None, PATHMP6=None,
                 MODELNAME='FloPyArcade', ANIMATIONFOLDER='FloPyArcade',
                 _seed=None, flagSavePlot=False, flagManualControl=False,
                 manualControlTime=0.1, flagRender=False, NAGENTSTEPS=200,
                 nLay=1, nRow=100, nCol=100, OBSPREP='perceptron',
                 initWithSolution=True, PATHMF6DLL=None,
                 env_config={}):

    # def __init__(self, env_config={}, initWithSolution=True):

    #     # ENVTYPE='1s-d',
    #     PATHMF2005=None
    #     PATHMP6=None
    #     MODELNAME='FloPyArcade'
    #     ANIMATIONFOLDER='FloPyArcade'
    #     _seed=None
    #     flagSavePlot=False
    #     flagManualControl=False
    #     manualControlTime=0.1
    #     flagRender=False
    #     NAGENTSTEPS=200
    #     nLay=1
    #     nRow=100
    #     nCol=100
    #     OBSPREP='perceptron'
    #     initWithSolution=True
    #     PATHMF6DLL=None

    # def __init__(self, env_config={}, initWithSolution=True):
        """Constructor."""

        import numpy as np
        # print('env_config pre', env_config)

        # !!! seed currently not in env_config as other methods are still parsing it via argument

        # ENVTYPE='1s-d'
        # ENVTYPE='6s-c'
        ENVTYPE=env_config['ENVTYPE'] if 'ENVTYPE' in env_config.keys() else ENVTYPE
        PATHMF2005=env_config['PATHMF2005'] if 'PATHMF2005' in env_config.keys() else PATHMF2005
        PATHMP6=env_config['PATHMP6'] if 'PATHMP6' in env_config.keys() else PATHMP6
        MODELNAME=env_config['MODELNAME'] if 'MODELNAME' in env_config.keys() else MODELNAME
        ANIMATIONFOLDER=env_config['ANIMATIONFOLDER'] if 'ANIMATIONFOLDER' in env_config.keys() else ANIMATIONFOLDER
        _seed=env_config['_seed'] if '_seed' in env_config.keys() else _seed
        flagSavePlot=env_config['flagSavePlot'] if 'flagSavePlot' in env_config.keys() else flagSavePlot
        flagManualControl=env_config['flagManualControl'] if 'flagManualControl' in env_config.keys() else flagManualControl
        manualControlTime=env_config['manualControlTime'] if 'manualControlTime' in env_config.keys() else 0.1
        flagRender=env_config['flagRender'] if 'flagRender' in env_config.keys() else flagRender
        NAGENTSTEPS=env_config['NAGENTSTEPS'] if 'NAGENTSTEPS' in env_config.keys() else NAGENTSTEPS
        nLay=env_config['nLay'] if 'nLay' in env_config.keys() else nLay
        nRow=env_config['nRow'] if 'nRow' in env_config.keys() else nRow
        nCol=env_config['nCol'] if 'nCol' in env_config.keys() else nCol
        OBSPREP=env_config['OBSPREP'] if 'OBSPREP' in env_config.keys() else OBSPREP # 'perceptron'
        initWithSolution=env_config['initWithSolution'] if 'initWithSolution' in env_config.keys() else initWithSolution
        PATHMF6DLL=env_config['PATHMF6DLL'] if 'PATHMF6DLL' in env_config.keys() else PATHMF6DLL

        env_config['ENVTYPE'] = ENVTYPE
        env_config['PATHMF2005'] = PATHMF2005
        env_config['PATHMP6'] = PATHMP6
        env_config['MODELNAME'] = MODELNAME
        env_config['ANIMATIONFOLDER'] = ANIMATIONFOLDER
        # env_config['_seed'] = _seed
        env_config['flagSavePlot'] = flagSavePlot
        env_config['flagManualControl'] = flagManualControl
        env_config['manualControlTime'] = manualControlTime
        env_config['flagRender'] = flagRender
        env_config['NAGENTSTEPS'] = NAGENTSTEPS
        env_config['nLay'] = nLay
        env_config['nRow'] = nRow
        env_config['nCol'] = nCol
        env_config['OBSPREP'] = OBSPREP
        env_config['initWithSolution'] = initWithSolution
        env_config['PATHMF6DLL'] = PATHMF6DLL

        # print(env_config)

        self.env_config = env_config

        # print('env_config post', env_config)

        self.env_config = env_config

        self.max_episode_steps = 200

        self.ENVTYPE = ENVTYPE
        self.PATHMF2005, self.PATHMP6 = PATHMF2005, PATHMP6
        self.MODELNAME = 'FloPyArcade' + str(uuid4())[0:5] if (MODELNAME==None) else MODELNAME
        if self.ENVTYPE in ['0s-c']:
            self.MODELNAMEGENCOUNT = self.MODELNAME
            self.MODELNAME = self.MODELNAME[0:15]

        self.ANIMATIONFOLDER = ANIMATIONFOLDER
        self.SAVEPLOT = flagSavePlot

        self.MANUALCONTROL = flagManualControl
        self.MANUALCONTROLTIME = manualControlTime
        self.RENDER = flagRender
        self.NAGENTSTEPS = NAGENTSTEPS
        self.info, self.comments = {}, ''
        self.done = False
        self.nLay, self.nRow, self.nCol = nLay, nRow, nCol
        self.OBSPREP = OBSPREP
        self.initWithSolution = initWithSolution

        self.wrkspc = dirname(abspath(__file__))
        if 'library.zip' in self.wrkspc:
            # changing workspace in case of call from executable
            self.wrkspc = dirname(dirname(self.wrkspc))
        # setting up the model path and ensuring it exists
        # self.modelpth = join(self.wrkspc, 'models', self.MODELNAME)
        # print(self.modelpth)
        # if not exists(self.modelpth):
            # makedirs(self.modelpth)
        self.tempDir = TemporaryDirectory(suffix=str(uuid4())[0:5])
        self.modelpth = str(self.tempDir.name)

        self.actionType = self.getActionType(self.ENVTYPE)

        self._SEED = _seed
        # if self._SEED is not None:
        #     numpySeed(self._SEED)
        self.defineEnvironment()
        self.timeStep, self.keyPressed = 0, None
        self.reward, self.rewardCurrent = 0., 0.
        self.delFiles = True
        self.canvasInitialized = False

        if self.ENVTYPE in ['0s-c']:
            # print('PATHMF6DLL', PATHMF6DLL)
            self.initializeSimulators(PATHMF6DLL=PATHMF6DLL)
            self.wrkspc = dirname(abspath(__file__))
            # self.exdir = exdir
            self.exdir = self.MODELNAME[0:15]
            self.model_ws = join(self.wrkspc, 'models', self.exdir)
            if not exists(self.model_ws):
                makedirs(self.model_ws)
            self.mf6dll = self.PATHMF6DLL
            self.name = self.MODELNAME[0:15]
            self.simpath = None
            # self.seed = seed
            # self.env = self.defineEnvironment(self._SEED)
            self.defineEnvironment(self._SEED)
            self.actionSpaceSize = self.getActionSpaceSize(self.actionsDict)
            self.renderFlag = flagRender
            if system() == 'Windows':
                # currently only supported in win64
                self.add_lib_dependencies([join(self.wrkspc, 'simulators', 'win64', 'win-builds', 'bin')])

            self.constructModel()

        else:
            self.initializeSimulators(PATHMF2005, PATHMP6)
            if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '2s-d', '2s-c', '2r-d', '2r-c']:
                self.initializeAction()
            self.initializeParticle()

            # this needs to be transformed, yet not understood why
            self.particleCoords[0] = self.extentX - self.particleCoords[0]

            if self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                self.headSpecNorth = uniform(self.minH, self.maxH)
                self.headSpecSouth = uniform(self.minH, self.maxH)
            self.initializeModel()

            self.wellX, self.wellY, self.wellZ, self.wellCoords, self.wellQ = self.initializeWellRate(self.minQ, self.maxQ)
            if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                helperWells = {}
                for i in range(self.nHelperWells):
                    w = str(i+1)
                    helperWells['wellX'+w], helperWells['wellY'+w], helperWells['wellZ'+w], helperWells['wellCoords'+w], helperWells['wellQ'+w] = self.initializeWellRate(self.minQhelper, self.maxQhelper)
                self.helperWells = helperWells

            self.initializeWell()
            if self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                self.initializeAction()
            # initializing trajectories container for potential plotting
            self.trajectories = {}
            for i in ['x', 'y', 'z']:
                self.trajectories[i] = []

        self.success = False

        if self.initWithSolution:
            self.stepInitial()

            # print('shape(self.observations)', shape(self.observations))
            self.observation_space = gym.spaces.Box(
                -25.0, 25.0, shape=shape(self.observations), dtype=np.float32)
                # -1.0, 2.0, shape=(shape(self.observations),), dtype=np.float32)
            # print('self.actionSpace', self.actionSpace)

            self.action_space = gym.spaces.Discrete(len(self.actionSpace))

            if self.actionType == 'discrete':
                self.action_space = gym.spaces.Discrete(len(self.actionSpace))
            elif self.actionType == 'continuous':
                self.action_space = gym.spaces.Box(0.0, 1.0, shape=(len(self.actionSpace),), dtype=np.float32)

    # def seed(self, seed=None):
    #     from numpy.random import random
    #     random.seed(seed)

    def teardown(self):
        """
        Remove the example folder
        """
        chdir(self.wrkspc)

        if self.delFiles:
            # msg = sfmt.format('Tearing down', self.exdir)
            # print(msg)

            # # wait to delete on windows
            # if sys.platform.lower() == "win32":
            #     sleep(3)

            try:
                self.mf6.finalize()
            except: pass

            try:
                # print('self.simpath', self.model_ws)
                rmtree(self.model_ws)
                self.teardownSuccess = True
            except Exception as e:
                print('Could not remove test ' + self.model_ws)
                print('error:', e)
                self.teardownSuccess = False
            # assert self.teardownSuccess
        else:
            print('Retaining files')

    def bmi_return(self, success, model_ws, modelname):
        """
        parse libmf6.so and libmf6.dll stdout file
        """

        fpth = join(model_ws, modelname + '.stdout')
        fpth = join(model_ws, 'mfsim' + '.stdout')
        return success, open(fpth).readlines()

    def stepInitial(self):
        """Initialize with the steady-state solution.
        
        Note: If just initializing environments without intention of solving,
        or intentions of solving later, this can be a massive throttleneck.
        """

        if self.ENVTYPE in ['0s-c']:

            self.model_ws = join(self.wrkspc, 'models', self.exdir)
            self.success = False
            modelname = self.name
            exe = self.mf6dll

            self.alive = True

            self.nameUpper = self.name.upper()
            if self.model_ws is not None:
                chdir(self.model_ws)
            chdir(self.model_ws)

            t0SteadyState = time()
            # steady state solution
            self.sim_steadyState.write_simulation(silent=True)

            mf6_config_file = join(self.model_ws, modelname + '.nam')
            chdir(self.model_ws)
            try:
                self.mf6 = XmiWrapper(exe)
            except Exception as e:
                print("Failed to load " + exe)
                print("with message: " + str(e))
                return self.bmi_return(self.success, self.model_ws, modelname)

            # print('3', 'mf6_config_file', mf6_config_file)
            # initialize the model
            try:
                # self.mf6.initialize(mf6_config_file)
                self.mf6.initialize(modelname + '.nam')
            except Exception as e:
                return self.bmi_return(self.success, self.model_ws)

            # maximum outer iterations
            mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
            self.max_iter = self.mf6.get_value(mxit_tag)

            dt = self.mf6.get_time_step()
            self.mf6.prepare_time_step(dt)

            # convergence loop
            kiter = 0
            self.mf6.prepare_solve(1)
            while kiter < self.max_iter:
                # solve with updated well rate
                has_converged = self.mf6.solve(1)
                kiter += 1

                if has_converged:
                    # msg = "Component {}".format(1) + \
                    #       " converged in {}".format(kiter) + " outer iterations"
                    # print(msg)
                    break
            if not has_converged:
                print('failed converging')
                return self.bmi_return(self.success, self.model_ws, modelname)

            # get pointer to simulated heads
            head_tag = self.mf6.get_var_address("X", self.nameUpper)
            self.head_steadyState_flat = deepcopy(self.mf6.get_value_ptr(head_tag))
            self.head_steadyState = deepcopy(self.head_steadyState_flat).reshape((self.nlay, self.nrow, self.ncol))

            # necessary to avoid error with file locks
            self.mf6.finalize()
            self.tSteadyState = time()-t0SteadyState


            chdir(self.model_ws)
            # transient simulation
            self.sim.write_simulation(silent=True)
            self.timeStep = int(0)

            mf6_config_file = join(self.model_ws, modelname + '.nam')
            try:
                self.mf6 = XmiWrapper(exe)
            except Exception as e:
                print("Failed to load " + exe)
                print("with message: " + str(e))
                return self.bmi_return(self.success, self.model_ws, modelname)

            # initialize the model
            try:
                # self.mf6.initialize(mf6_config_file)
                self.mf6.initialize(modelname + '.nam')
            except:
                return self.bmi_return(self.success, self.model_ws)

            self.dt = self.mf6.get_time_step()

            # time loop
            self.current_time = self.mf6.get_current_time()
            self.end_time = self.mf6.get_end_time()

            # get pointer to simulated heads
            head_tag = self.mf6.get_var_address("X", self.nameUpper)
            head = self.mf6.get_value_ptr(head_tag)
            self.head = head

            # setting steady-state heads as initial values
            head_tag = self.mf6.get_var_address("X", self.nameUpper)
            # self.mf6.set_value(head_tag, self.head_steadyState.flatten())
            self.mf6.set_value(head_tag, self.head_steadyState)

            if self.renderFlag or self.SAVEPLOT:
                self.render()

            # options for available variables
            # print(self.mf6.get_component_name())
            # print(self.mf6.get_input_var_names())
            # print(self.mf6.get_output_var_names())

            # retrieving copy of recharge array
            rch_tag = self.mf6.get_var_address("BOUND", self.nameUpper, "RCHA_0")
            new_recharge = self.mf6.get_value(rch_tag).copy()

            grid_shape = shape(head.reshape((self.nlay, self.nrow, self.ncol)))

            # maximum outer iterations
            mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
            self.max_iter = self.mf6.get_value(mxit_tag)

            # get copy of well data

            # currently requires copying of set_value function from https://github.com/Deltares/xmipy/blob/develop/xmipy/xmiwrapper.py
            # to local installation

            self.well_tag = self.mf6.get_var_address("BOUND", self.nameUpper, "WEL_0")
            self.well = self.mf6.get_value(self.well_tag)

            # needs to observe more, potentially storm parameters?
            observations_wells = []
            for iHelper in range(self.nHelperWells):
                x = self.actionsDict['well' + str(iHelper) + 'x']
                x = self.normalize(x, 0, self.extentX)
                observations_wells.append(x)
                y = self.actionsDict['well' + str(iHelper) + 'y']
                y = self.normalize(y, 0, self.extentY)
                observations_wells.append(y)
                Q = self.actionsDict['well' + str(iHelper) + 'Q']
                Q = self.normalize(Q, self.minQ, self.diffQ)
                observations_wells.append(Q)

            observations_storms = []
            for iStorm in range(self.nStorms):
                stormStart = self.stormStarts[iStorm]/self.nstp
                observations_storms.append(stormStart)
                observations_storms.append(self.timeStep / self.nstp)
                stormDuration = self.normalize(self.stormDurations[iStorm], 0, self.maxStormDuration)
                observations_storms.append(stormDuration)
                stormIntensity = self.normalize(self.stormIntensities[iStorm], self.storm_rch_min, numpyAbs(self.storm_rch_min-self.storm_rch_max))
                observations_storms.append(stormIntensity)
                stormCenterX = self.normalize(self.stormCentersX[iStorm], 0., self.extentX)
                observations_storms.append(stormCenterX)
                stormCenterY = self.normalize(self.stormCentersY[iStorm], 0., self.extentY)
                observations_storms.append(stormCenterY)

            # self.normalize(head, self.chd_east, self.chd_diff)
            observation = []
            observation += list(self.normalize(head, self.chd_east, self.chd_diff).flatten())
            observation += list(self.normalize(new_recharge, self.avg_rch_min, self.rch_diff).flatten())
            observation += observations_wells
            observation += observations_storms

            self.observations = observation
            self.observationsVectorNormalized = observation
            reward = 0.
            self.done = False
            info = None

            return self.observationsVectorNormalized, reward, self.done, info

        else:
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
            elif self.ENVTYPE in ['5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c']:
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
            self.observations['headsSampledField'] = self.heads[0::self.sampleHeadsEvery,
                                                    0::self.sampleHeadsEvery,
                                                    0::self.sampleHeadsEvery]
            self.observations['heads'] = self.heads
            lParticle, cParticle, rParticle = self.cellInfoFromCoordinates(
                [self.particleCoords[0], self.particleCoords[1], self.particleCoords[2]])
            lWell, cWell, rWell = self.cellInfoFromCoordinates(
                [self.wellX, self.wellY, self.wellZ])

            if self.OBSPREP == 'perceptron':
                # note: these heads from actions are not necessary to return as observations for surrogate modeling
                # but for reinforcement learning
                if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
                    self.observations['heads'] = [self.actionValueNorth,
                                                  self.actionValueSouth]
                elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
                    # this can cause issues with unit testing, as model expects different input 
                    self.observations['heads'] = [self.actionValue]
                elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
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
            elif self.OBSPREP == 'convolution':
                self.observations['heads'] = self.heads

            self.observations['wellQ'] = self.wellQ
            self.observations['wellCoords'] = self.wellCoords
            if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                for i in range(self.nHelperWells):
                    w = str(i+1)
                    self.observations['wellQ'+w] = self.helperWells['wellQ'+w]
                    self.observations['wellCoords'+w] = self.helperWells['wellCoords'+w]
            self.observationsNormalized['particleCoords'] = divide(
                copy(self.particleCoords), numpyAbs(self.minX + self.extentX))
            self.observationsNormalized['heads'] = divide(array(self.observations['heads']) - self.minH,
                self.maxH - self.minH)
            self.observationsNormalized['wellQ'] = self.wellQ / self.minQ
            self.observationsNormalized['wellCoords'] = divide(
                self.wellCoords, numpyAbs(self.minX + self.extentX))
            if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                for i in range(self.nHelperWells):
                    w = str(i+1)
                    self.observationsNormalized['wellQ'+w] = self.helperWells['wellQ'+w] / self.minQhelper
                    self.observationsNormalized['wellCoords'+w] = divide(
                        self.helperWells['wellCoords'+w], numpyAbs(self.minX + self.extentX))
            self.observationsNormalizedHeads['heads'] = divide(array(self.heads) - self.minH,
                self.maxH - self.minH)

            if self.OBSPREP == 'perceptron':
                self.observationsVector = self.observationsDictToVector(
                    self.observations)
                self.observationsVectorNormalized = self.observationsDictToVector(
                    self.observationsNormalized)
                self.observationsVectorNormalizedHeads = self.observationsDictToVector(
                    self.observationsNormalizedHeads)
            elif self.OBSPREP == 'convolution':
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
            elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                self.stressesVectorNormalized = [(self.headSpecSouth - self.minH)/(self.maxH - self.minH),
                                                 (self.headSpecNorth - self.minH)/(self.maxH - self.minH)]
                for i in range(self.nHelperWells):
                    w = str(i+1)
                    self.stressesVectorNormalized += [self.helperWells['wellQ'+w]/self.minQ, self.helperWells['wellX'+w]/(self.minX+self.extentX),
                        self.helperWells['wellY'+w]/(self.minY+self.extentY), self.helperWells['wellZ'+w]/(self.zBot+self.zTop)]

            self.observations = array(self.observationsVectorNormalized)

            # self.timeStepDuration = []
            # print('initial shape(self.observations)', shape(self.observations))

    # def step(self, observations, action, rewardCurrent, teardownOnFinish=False):
    def step(self, action, teardownOnFinish=True):

        """Perform a single step of forwards simulation."""

        if self.actionType == 'discrete':
            # if actions are given in integers
            if action not in self.actionSpace:
                action = self.actionSpace[action-1]

        # print('min(action)', min(action), max(action))

        if self.ENVTYPE in ['0s-c']:

            # model time loop
            if self.current_time <= self.end_time:
                t0Step = time()
                self.timeStep += int(1)

                # if self.timeStep == self.nstp-1:
                if self.timeStep == self.nstp:
                    self.done = True

                # get dt and prepare for non-linear iterations
                # self.dt = self.mf6.get_time_step()
                self.mf6.prepare_time_step(self.dt)
                self.current_time = self.mf6.get_current_time()
         
                # get pointer to simulated heads
                head_tag = self.mf6.get_var_address("X", self.nameUpper)
                head = self.mf6.get_value_ptr(head_tag)
                self.head = head

                rch_tag = self.mf6.get_var_address("BOUND", self.nameUpper, "RCHA_0")
                new_recharge = self.mf6.get_value(rch_tag)

                # updating well location
                # not add up pumping rates if overlapping
                wellCells = []
                for iHelper in range(self.nHelperWells):
                    wellCells.append(self.actionsDict['well' + str(iHelper) + 'cellID'])
                    # if iHelper == 0:
                    #     print('x', self.actionsDict['well' + str(iHelper) + 'x'])
                # print('time 1', time()-t0Step)
                wellCells = array(wellCells).astype(int32)

                # print(self.mf6.get_component_name())
                # print(self.mf6.get_input_var_names())
                # print(self.mf6.get_output_var_names())

                # seems like on unix, theis is currently not available
                wel_xtag = self.mf6.get_var_address('NODELIST', self.nameUpper, 'WEL_0')
                wel_x = self.mf6.get_value(wel_xtag)
                # print('NODELIST', wel_x, type(wel_x), type(wel_x[0]))
                self.mf6.set_value(wel_xtag, add(multiply(wel_x, 0), wellCells))
                # print('time 2', time()-t0Step)

                # updating well rate
                self.actionsDict = self.changeActionDict(self.actionsDict, action)
                self.actionsDict = self.getActionValues(self.actionsDict)

                Qs = []
                for iHelper in range(self.nHelperWells):
                    dxLeft = self.well_dxMax*self.actionsDict['well' + str(iHelper) + 'actiondxLeft']
                    dxRight = self.well_dxMax*self.actionsDict['well' + str(iHelper) + 'actiondxRight']
                    x = self.actionsDict['well' + str(iHelper) + 'x'] # + dxRight - dxLeft # has moved in getActionValues already
                    dyUp = self.well_dyMax*self.actionsDict['well' + str(iHelper) + 'actiondyUp']
                    dyDown = self.well_dyMax*self.actionsDict['well' + str(iHelper) + 'actiondyDown']
                    y = self.actionsDict['well' + str(iHelper) + 'y'] # + dyUp - dyDown # has moved in getActionValues already
                    dQUp = self.well_dyMax*self.actionsDict['well' + str(iHelper) + 'actiondQUp']
                    dQDown = self.well_dyMax*self.actionsDict['well' + str(iHelper) + 'actiondQDown']
                    Q = self.actionsDict['well' + str(iHelper) + 'Q']
                    Qs.append(Q)

                    # print('debug',
                    #       'left action', '%.2f' % (dxLeft-dxRight),
                    #       'up action', '%.2f' % (dyUp-dyDown),
                    #       'Q action', '%.2f' % (dQUp-dQDown))

                # print(self.current_time, Qs)

                self.well[:, 0] = Qs

                # print('NODELIST', wel_x, type(wel_x), type(wel_x[0]))
                # print(dir(self.mf6))
                import numpy as np
                _values = add(multiply(wel_x, 0), wellCells)
                _values = np.int32(_values)
                # for i, v in enumerate(_values):
                    # _values[i] = np.float64(float(v))
                # print(type(_values[0]))
                # print(wel_xtag, _values)

                self.mf6.set_value(self.well_tag, self.well)

                # updating recharge randomly
                new_recharge = self.rechTimeSeries[self.timeStep]
                self.mf6.set_value(rch_tag, new_recharge)
                # print('time 6', time()-t0Step)

                # convergence loop
                # t0Solve = time()
                kiter = 0
                self.mf6.prepare_solve(1)
                while kiter < self.max_iter:
                    # solve with updated well rate
                    has_converged = self.mf6.solve(1)
                    kiter += 1

                    if has_converged:
                        break
                if not has_converged:
                    print('failed converging')
                    _, _ = self.bmi_return(self.success, self.model_ws, modelname)
                # print('time solve', time()-t0Solve)
                # print('time 7', time()-t0Step)

                reward = self.calculateGameRewardHeadChange(self.head_steadyState_flat, head)
                # print('zgoih', self.timeStep, mean(self.head_steadyState_flat), mean(head))
                self.reward = copy(reward)
                self.rewardCurrent += self.reward

                # print('time 8', time()-t0Step)

                if self.renderFlag or self.SAVEPLOT:
                    self.render()

                # finalize time step
                # t0 = time()
                # self.mf6.finalize_solve(1)
                # print('finalize_solve', time()-t0)

                # t0 = time()
                # finalize time step and update time
                # self.mf6.finalize_time_step()
                # print('finalize_time_step', time()-t0)

            # t0Postprocess = time()
            # needs to observe more, potentially storm parameters?
            observations_wells = []
            for iHelper in range(self.nHelperWells):
                x = self.actionsDict['well' + str(iHelper) + 'x']
                x = self.normalize(x, 0, self.extentX)
                observations_wells.append(x)
                y = self.actionsDict['well' + str(iHelper) + 'y']
                y = self.normalize(y, 0, self.extentY)
                observations_wells.append(y)
                Q = self.actionsDict['well' + str(iHelper) + 'Q']
                Q = self.normalize(Q, self.minQ, self.diffQ)
                observations_wells.append(Q)

            observations_storms = []
            for iStorm in range(self.nStorms):
                stormStart = self.stormStarts[iStorm]/self.nstp
                observations_storms.append(stormStart)
                observations_storms.append(self.timeStep / self.nstp)
                stormDuration = self.normalize(self.stormDurations[iStorm], 0, self.maxStormDuration)
                observations_storms.append(stormDuration)
                stormIntensity = self.normalize(self.stormIntensities[iStorm], self.storm_rch_min, numpyAbs(self.storm_rch_min-self.storm_rch_max))
                observations_storms.append(stormIntensity)
                stormCenterX = self.normalize(self.stormCentersX[iStorm], 0., self.extentX)
                observations_storms.append(stormCenterX)
                stormCenterY = self.normalize(self.stormCentersY[iStorm], 0., self.extentY)
                observations_storms.append(stormCenterY)

            observation = []
            observation += list(self.normalize(head, self.chd_east, self.chd_diff).flatten())
            observation += list(self.normalize(new_recharge, self.avg_rch_min, self.rch_diff).flatten())
            observation += observations_wells
            observation += observations_storms

            info = None

            observation = copy(observation)
            # self.done = copy(self.done)

            if self.done:
                # cleanup

                try:
                    self.mf6.finalize()
                    self.success = True
                    # print('success finalizing')
                except:
                    _, _ =  self.bmi_return(self.success, self.model_ws)

                chdir(self.model_ws)
                if teardownOnFinish:
                    self.teardown()
            # print('time 9', time()-t0Step)

            # print('time per step', time()-t0Step)
            self.observations = observation
            self.observationsVectorNormalized = observation

            return self.observationsVectorNormalized, reward, self.done, info

        else:

            t0 = time()
            self.debugTimeBlind = time()-t0

            if self.timeStep == 0:
                if not self.initWithSolution:
                    self.stepInitial()
                # rendering initial timestep
                if self.RENDER or self.MANUALCONTROL or self.SAVEPLOT:
                    self.render()
            self.timeStep += 1
            self.keyPressed = None
            self.periodSteadiness = False
            # t0total = time()

            # print('debug action step', action)
            self.setActionValue(action)
            observations = self.observationsVectorToDict(list(self.observations))
            rewardCurrent = self.rewardCurrent
            # observations = self.observationsVectorToDict(observations)
            self.particleCoordsBefore = observations['particleCoords']

            # it might be obsolete to feed this back,
            # as it can be stored with the object
            self.rewardCurrent = rewardCurrent

            # does this need to be enabled? It disables numpy finding different
            # random numbers throughout the game,
            # for example for random action exploration
            # if self._SEED is not None:
            #     numpySeed(self._SEED)

            self.debugTime1 = time()-t0

            self.initializeState(self.state)
            self.debugTime2 = time()-t0
            self.updateModel()
            self.debugTime3 = time()-t0
            self.updateWellRate()
            self.debugTime4 = time()-t0
            self.updateWell()
            self.debugTime5 = time()-t0
            self.runMODFLOW()
            self.debugTime6 = time()-t0
            self.runMODPATH()
            self.debugTime7 = time()-t0
            self.evaluateParticleTracking()
            self.debugTime8 = time()-t0

            # calculating game reward
            if self.ENVTYPE in ['5s-c-cost']:
                self.reward = self.calculateGameRewardOperationCost(self.nHelperWells, self.helperWells)
            else:
                self.reward = self.calculateGameRewardTrajectory(self.trajectories)

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
            elif self.ENVTYPE in ['5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c']:
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
            self.observations['headsSampledField'] = self.heads[0::self.sampleHeadsEvery,
                                                    0::self.sampleHeadsEvery,
                                                    0::self.sampleHeadsEvery]
            self.observations['heads'] = self.heads
            lParticle, cParticle, rParticle = self.cellInfoFromCoordinates(
                [self.particleCoords[0], self.particleCoords[1], self.particleCoords[2]])
            lWell, cWell, rWell = self.cellInfoFromCoordinates(
                [self.wellX, self.wellY, self.wellZ])

            if self.OBSPREP == 'perceptron':
                # note: these heads from actions are not necessary to return as observations for surrogate modeling
                # but for reinforcement learning
                if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
                    self.observations['heads'] = [self.actionValueNorth,
                                                  self.actionValueSouth]
                elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
                    # this can cause issues with unit testing, as model expects different input 
                    self.observations['heads'] = [self.actionValue]
                elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
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
            elif self.OBSPREP == 'convolution':
                self.observations['heads'] = self.heads

            self.observations['wellQ'] = self.wellQ
            self.observations['wellCoords'] = self.wellCoords
            if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                for i in range(self.nHelperWells):
                    w = str(i+1)
                    self.observations['wellQ'+w] = self.helperWells['wellQ'+w]
                    self.observations['wellCoords'+w] = self.helperWells['wellCoords'+w]
            self.observationsNormalized['particleCoords'] = divide(
                copy(self.particleCoordsAfter), numpyAbs(self.minX + self.extentX))
            self.observationsNormalized['heads'] = divide(array(self.observations['heads']) - self.minH,
                self.maxH - self.minH)
            # self.observationsNormalized['heads'] = divide(array(self.observations['heads']) - self.minH,
            #     self.maxH - self.minH)
            self.observationsNormalized['wellQ'] = self.wellQ / self.minQ
            self.observationsNormalized['wellCoords'] = divide(
                self.wellCoords, numpyAbs(self.minX + self.extentX))
            if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                for i in range(self.nHelperWells):
                    w = str(i+1)
                    self.observationsNormalized['wellQ'+w] = self.helperWells['wellQ'+w] / self.minQhelper
                    self.observationsNormalized['wellCoords'+w] = divide(
                        self.helperWells['wellCoords'+w], numpyAbs(self.minX + self.extentX))
            self.observationsNormalizedHeads['heads'] = divide(array(self.heads) - self.minH,
                self.maxH - self.minH)

            if self.OBSPREP == 'perceptron':
                self.observationsVector = self.observationsDictToVector(
                    self.observations)
                self.observationsVectorNormalized = self.observationsDictToVector(
                    self.observationsNormalized)
                self.observationsVectorNormalizedHeads = self.observationsDictToVector(
                    self.observationsNormalizedHeads)
            elif self.OBSPREP == 'convolution':
                self.observationsVector = self.observationsDictToVector(
                    self.observations)
                self.observationsVectorNormalized = self.observationsDictToVector(
                    self.observationsNormalized)

            if self.observations['particleCoords'][0] >= self.extentX - self.dCol:
                self.alive = True
            else:
                self.alive = False

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
            elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
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
                if self.ENVTYPE in ['5s-c-cost']:
                    self.reward = self.costQFail + self.rewardCurrent
                else:
                    # resetting to zero if reward is positive
                    if self.rewardCurrent >= 0.:
                        self.reward = (self.rewardCurrent) * (-1.0)
            if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
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
                        if self.ENVTYPE in ['5s-c-cost']:
                            self.reward = self.costQFail + self.rewardCurrent
                        else:
                            # resetting to zero if reward is positive
                            if self.rewardCurrent >= 0.:
                                self.reward = (self.rewardCurrent) * (-1.0)

            # checking if particle has reached eastern boundary
            if self.particleCoordsAfter[0] >= self.minX + self.extentX - self.dCol:
                self.done = True

            # checking if particle has returned to western boundary
            if self.particleCoordsAfter[0] <= self.minX + self.dCol:
                self.done = True
                if self.ENVTYPE in ['5s-c-cost']:
                    self.reward = self.costQFail + self.rewardCurrent
                else:
                    # resetting to zero if reward is positive
                    if self.rewardCurrent >= 0.:
                        self.reward = (self.rewardCurrent) * (-1.0)

            if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                # checking if particle has reached northern boundary
                if self.particleCoordsAfter[1] >= self.minY + self.extentY - self.dRow:
                # if self.particleCoordsAfter[1] >= self.minY + \
                #         self.extentY - self.dRow:
                    self.done = True
                    if self.ENVTYPE in ['5s-c-cost']:
                        self.reward = self.costQFail + self.rewardCurrent
                    else:
                        # resetting to zero if reward is positive
                        if self.rewardCurrent >= 0.:
                            self.reward = (self.rewardCurrent) * (-1.0)

            # checking if particle has reached southern boundary
            if self.particleCoordsAfter[1] <= self.minY + self.dRow:
                self.done = True
                if self.ENVTYPE in ['5s-c-cost']:
                    self.reward = self.costQFail + self.rewardCurrent
                else:
                    # resetting to zero if reward is positive
                    if self.rewardCurrent >= 0.:
                        self.reward = (self.rewardCurrent) * (-1.0)

            # aborting game if a threshold of steps have been taken
            if self.timeStep == self.maxSteps:
                if self.done != True:
                    self.done = True
                    if self.ENVTYPE in ['5s-c-cost']:
                        self.reward = self.costQFail + self.rewardCurrent
                    else:
                        # resetting to zero if reward is positive
                        if self.rewardCurrent >= 0.:
                            self.reward = (self.rewardCurrent) * (-1.0)

            self.rewardCurrent += self.reward
            # self.timeStepDuration.append(time() - t0total)

            if self.RENDER or self.MANUALCONTROL or self.SAVEPLOT:
                self.render()

            # if self.done:
                # if not self.ENVTYPE in ['5s-c-cost']:
                #     if not self.success:
                #         if 
                # print('debug average timeStepDuration', mean(self.timeStepDuration))

                # necessary to remove these file handles to release file locks
                # del self.mf
                # self.cbb, self.hdobj

                # for f in listdir(self.modelpth):
                #     # removing files in folder
                #     remove(join(self.modelpth, f))
                # if exists(self.modelpth):
                #     # removing folder with model files after run
                #     rmdir(self.modelpth)

            # self.observations = array(self.observationsVectorNormalized)
            self.observations = array(self.observationsVectorNormalized)
            # self.rewardCurrent = self.reward
            # self.state = self.observationsVectorNormalized

            # print('werb', self.observations, self.reward, self.done, self.info)
            # return [0.4], -0.1, self.done, {}
            # print('max obs', np.max(self.observationsVectorNormalized))

            # print('shape(self.observations)', shape(self.observations))
            return self.observations, self.reward, self.done, self.info

    def defineEnvironment(self, seed=None):
        """Define environmental variables."""

        numpySeed(self._SEED)

        if self.ENVTYPE in ['0s-c']:

            # temporal discretization
            self.perlen = 50
            # self.perlen = 200
            self.nper = 1
            # self.nstp = 200
            self.nstp = 50
            self.maxSteps = self.nstp


            self.tdis_rc = []
            self.tdis_rc.append((self.perlen, self.nstp, 1))

            # model spatial dimensions
            self.nlay, self.nrow, self.ncol = 1, 25, 25
            # self.nlay, self.nrow, self.ncol = 1, 20, 20
            # self.nlay, self.nrow, self.ncol = 1, 100, 100

            # cell spacing
            self.delr = 100.
            self.delc = 100.
            self.extentX = self.ncol * self.delr
            self.extentY = self.nrow * self.delc
            self.domainArea = self.delr * self.delc

            # top of the aquifer
            self.top = 50.

            # bottom of the aquifer
            self.botm = 0.

            # hydraulic conductivity
            self.hk = 0.05

            # starting head
            self.strt = 58.
            self.chd_west, self.chd_east = 58.5, 57.5
            self.chd_diff = numpyAbs(self.chd_west-self.chd_east)

            self.nHelperWells = 5
            self.nStorms = 2

            # allowed pumping rates
            self.actionsDict = {}
            self.well_dxMax = 2*self.delr
            self.well_dyMax = 2*self.delc
            self.minQ, self.maxQ = -15., 15.
            self.diffQ = numpyAbs(self.minQ-self.maxQ)
            self.well_dQMax = 5.
            for iHelper in range(self.nHelperWells):
                x = uniform(0. + 1.1 * self.delr, self.extentX - 1.1 * self.delr)
                self.actionsDict['well' + str(iHelper) + 'x'] = x
                y = uniform(0. + 1.1 * self.delc, self.extentY - 1.1 * self.delc)
                self.actionsDict['well' + str(iHelper) + 'y'] = y
                self.actionsDict['well' + str(iHelper) + 'cellID'] = self.get_cellID(x, y)
                iRow, iCol = self.get_cellColRow(x, y)
                self.actionsDict['well' + str(iHelper) + 'iRow'] = iRow
                self.actionsDict['well' + str(iHelper) + 'iCol'] = iCol
                # self.actionsDict['well' + str(iHelper) + 'Q'] = uniform(self.minQ, self.maxQ)
                self.actionsDict['well' + str(iHelper) + 'Q'] = 0.0

            self.avg_rch_min, self.avg_rch_max = 0., 0.
            # self.avg_rch_min, self.avg_rch_max = 1e-8, 1e-6
            # self.avg_rch_min, self.avg_rch_max = 1e-6, 1e-2
            self.storm_rch_min, self.storm_rch_max = 1e-1, 2e1
            self.rch_diff = numpyAbs(self.avg_rch_min-self.storm_rch_max)
            self.minStormDuration, self.maxStormDuration = 3, 20
            self.minStormRadius, self.maxStormRadius = 300, 1000
            self.stormCentersX, self.stormCentersY = [], []
            self.stormStarts, self.stormDurations, self.stormIntensities, self.stormRadii = [], [], [], []
            for iStorm in range(self.nStorms):
                stormStart = int(uniform(2, self.nstp-1))
                stormDuration = int(uniform(self.minStormDuration, self.maxStormDuration))
                stormIntensity = uniform(self.storm_rch_min, self.storm_rch_max)
                self.stormCenterX = uniform(0, self.extentX)
                self.stormCenterY = uniform(0, self.extentY)
                stormRadius = uniform(self.minStormRadius, self.maxStormRadius)
                self.stormStarts.append(stormStart)
                self.stormDurations.append(stormDuration)
                self.stormIntensities.append(stormIntensity)
                self.stormCentersX.append(self.stormCenterX)
                self.stormCentersY.append(self.stormCenterY)
                self.stormRadii.append(stormRadius)

            # average recharge rate
            self.shapeLayer = (1, self.nrow, self.ncol)
            self.shapeGrid = (self.nlay, self.nrow, self.ncol)
            self.rech0 = uniform(low=self.avg_rch_min, high=self.avg_rch_max, size=self.shapeLayer)
            # self.rech0 = uniform(low=0.0, high=0.0, size=self.shapeLayer)
            self.rechTimeSeries, activeStormsIdx = [], []
            self.rechTimeSeries.append(self.rech0)
            for iStp in range(self.nstp):
                if iStp+1 in self.stormStarts:
                    iStorm = self.stormStarts.index(iStp+1)
                    activeStormsIdx.append(iStorm)
                rech = copy(self.rech0)
                stormDurationsTemp = copy(self.stormDurations)
                for iStorm in activeStormsIdx:
                    if stormDurationsTemp[iStorm] > 0:
                        ID = self.get_cellID(self.stormCentersX[iStorm], self.stormCentersY[iStorm])
                        iArray = ID-1
                        rech = rech.flatten()
                        rech[iArray] = self.stormIntensities[iStorm]
                        rech = rech.reshape(self.shapeLayer)
                        # modify storm region (just 5 cells around, if possible)?
                    stormDurationsTemp[iStorm] -= 1
                self.rechTimeSeries.append(rech)
            # print('debug', self.stormStarts, self.stormDurations, self.stormIntensities, self.stormRadii)

            self.actionSpaceIndividual = ['up', 'down', 'left', 'right', 'Qup', 'Qdown']
            self.actionSpace = []
            for i in range(self.nHelperWells):
                self.actionSpace += [j+str(i+1) for j in self.actionSpaceIndividual]

        else:
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
            self.periods, self.periodLength, self.periodSteps = 1, 1.0, 11 # 11
            self.periodSteadiness = True # to get steady-state starting solution, will change later
            self.maxSteps = self.NAGENTSTEPS
            self.sampleHeadsEvery = 3 # 5

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
                self.deviationPenaltyFactor = 10.0
                self.actionRange = 0.5
            elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c']:
                self.minH = 56.0
                self.maxH = 60.0
                self.nHelperWells = 0
                self.deviationPenaltyFactor = 10.0
                self.actionRange = 10.0
            elif self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                self.helperWellRadius = self.wellRadius/4
                self.minH = 56.0
                self.maxH = 60.0
                self.nHelperWells = 20 # 7
                self.minQhelper = -600.0
                self.maxQhelper = 600.0
                self.deviationPenaltyFactor = 10.0
                self.costQm3dPump = 0.48 # Euro / m3/d
                self.costQm3dRecharge = 0.70 # Euro / m3/d
                self.costQFail = -1.*2.*self.maxSteps*self.nHelperWells*self.maxQhelper*self.costQm3dRecharge

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

            elif self.ENVTYPE in ['4s-c', '4r-c', '5s-c', '5s-c-cost', '5r-c', '6s-c', '6r-c']:
                if self.ENVTYPE in ['4s-c', '4r-c']:
                    self.actionSpaceIndividual = ['up', 'down', 'left', 'right']
                    self.actionRange = 2.5
                elif self.ENVTYPE in ['5s-c', '5s-c-cost', '5r-c']:
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

            self.rewardMax = 1000 # 1000
            self.distanceMax = 98 # 97.9

        # solver data
        self.nouter, self.ninner = 100, 100
        # self.hclose, self.rclose, self.relax = 1e-6, 1e-3, 0.97
        # self.hclose, self.rclose, self.relax = 1e-12, 1e-9, 0.999
        self.hclose, self.rclose, self.relax = 1e-8, 1e-8, 1.

    def initializeSimulators(self, PATHMF2005=None, PATHMP6=None, PATHMF6DLL=None):
        """Initialize simulators depending on operating system.
        Executables have to be specified or located in simulators subfolder.
        """

        from ctypes import sizeof, c_voidp
        from platform import system as platformSystem

        c_voidp = sizeof(c_voidp)
        if c_voidp == 4:
            bits = 32
        elif c_voidp == 8:
            bits = 64

        # setting name of MODFLOW and MODPATH executables
        if platformSystem() == 'Windows':
            
            if bits == 32:
                subfolder = 'win32'
            elif bits == 64:
                subfolder = 'win64'

            if PATHMF2005 is None:
                self.exe_name = join(self.wrkspc, 'simulators',
                                     subfolder, 'mf2005'
                                     ) + '.exe'
            elif PATHMF2005 is not None:
                self.exe_name = PATHMF2005
            if PATHMP6 is None:
                self.exe_mp = join(self.wrkspc, 'simulators',
                                   subfolder, 'mp6'
                                   ) + '.exe'
            elif PATHMP6 is not None:
                self.exe_mp = PATHMP6
            if PATHMF6DLL is None:
                self.PATHMF6DLL = join(self.wrkspc, 'simulators',
                                       subfolder, 'libmf6'
                                       ) + '.dll'
            elif PATHMF6DLL is not None:
                self.PATHMF6DLL = PATHMF6DLL

        elif platformSystem() == 'Linux' or platformSystem() == 'Darwin':
            
            if platformSystem() == 'Linux':
                subfolder = 'linux'
            elif platformSystem() == 'Darwin':
                subfolder = 'mac'

            if PATHMF2005 is None:
                self.exe_name = join(self.wrkspc, 'simulators', subfolder, 'mf2005')
                chmod(self.exe_name, 0o775)
            elif PATHMF2005 is not None:
                self.exe_name = PATHMF2005
                chmod(self.exe_name, 0o775)
            if PATHMP6 is None:
                self.exe_mp = join(self.wrkspc, 'simulators', subfolder, 'mp6')
                chmod(self.exe_mp, 0o775)
            elif PATHMP6 is not None:
                self.exe_mp = PATHMP6
                chmod(self.exe_mp, 0o775)
            if PATHMF6DLL is None:
                self.PATHMF6DLL = join(self.wrkspc, 'simulators', subfolder, 'libmf6') + '.so'
                chmod(self.PATHMF6DLL, 0o775)
            elif PATHMF6DLL is not None:
                self.PATHMF6DLL = PATHMF6DLL
                chmod(self.PATHMF6DLL, 0o775)

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
        elif self.ENVTYPE in ['5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c']:
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
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.helperWells['l'+w], self.helperWells['c'+w], self.helperWells['r'+w] = self.cellInfoFromCoordinates(
                    [self.helperWells['wellX'+w], self.helperWells['wellY'+w], self.helperWells['wellZ'+w]])

        # print('debug well cells', self.wellCellLayer, self.wellCellColumn, self.wellCellRow)
        # adding WEL package to the MODFLOW model
        lrcq = {0: [[l-1, r-1, c-1, self.wellQ]]}
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
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

        if self.ENVTYPE in ['5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
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
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            for i in range(self.nHelperWells):
                w = str(i+1)
                self.helperWells['l'+w], self.helperWells['c'+w], self.helperWells['r'+w] = self.cellInfoFromCoordinates(
                    [self.helperWells['wellX'+w], self.helperWells['wellY'+w], self.helperWells['wellZ'+w]])

        lrcq = {0: [[l-1, r-1, c-1, self.wellQ]]}
        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            lrcq_ = [[l-1, r-1, c-1, self.wellQ]]
            for i in range(self.nHelperWells):
                w = str(i+1)
                lrcq_.append([self.helperWells['l'+w]-1, self.helperWells['r'+w]-1, self.helperWells['c'+w]-1, self.helperWells['wellQ'+w]])
            lrcq = {0: lrcq_}
        self.wel = ModflowWel(self.mf, stress_period_data=lrcq, options=['NOPRINT'])

    def constructModel(self):
        """Construct the groundwater flow model used for the arcade game.
        Flopy is used as a MODFLOW wrapper for input file construction.
        A specified head boundary condition is situated on the western, eastern
        and southern boundary. The southern boundary condition can be modified
        during the game. Generally, the western and eastern boundaries promote
        groundwater to flow towards the west. To simplify, all model parameters
        and aquifer thickness is homogeneous throughout.
        """

        if self.ENVTYPE in ['0s-c']:

            self.sim = MFSimulation(sim_name=self.name, version='mf6',
                exe_name=self.mf6dll, sim_ws=self.model_ws, memory_print_option='all')

            # long period length to retrieve steady-state result
            tdis = ModflowTdis(self.sim, time_units="DAYS",
                               nper=1, perioddata=[(10000000000000.0, 100, 1)])
                               # verbose=False

            # create iterative model solution and register the gwf model with it
            ims = ModflowIms(self.sim,
                             print_option='SUMMARY',
                             outer_dvclose=self.hclose,
                             outer_maximum=self.nouter,
                             under_relaxation='SIMPLE',
                             under_relaxation_gamma=0.98,
                             inner_maximum=self.ninner,
                             inner_dvclose=self.hclose, rcloserecord=self.rclose,
                             linear_acceleration='BICGSTAB',
                             relaxation_factor=self.relax)

            # create gwf model
            newtonoptions = ['NEWTON', 'UNDER_RELAXATION']
            # newtonoptions = ['NEWTON']

            self.gwf = ModflowGwf(self.sim, newtonoptions=newtonoptions,
                                  modelname=self.name, print_input=False, 
                                  save_flows=True
                                  )

            dis = ModflowGwfdis(self.gwf, nlay=self.nlay, nrow=self.nrow, ncol=self.ncol,
                                          delr=self.delr, delc=self.delc, top=self.top, botm=self.botm)

            # initial conditions
            ic = ModflowGwfic(self.gwf, strt=self.strt)

            # setting constant head boundary condition
            chd_rec = []
            for ilay in range(self.nlay):
                for irow in range(self.nrow):
                    chd_rec.append([(ilay, irow, 0), self.chd_west])
                    chd_rec.append([(ilay, irow, self.ncol-1), self.chd_east])
            chd = ModflowGwfchd(self.gwf,
                                          maxbound=len(chd_rec),
                                          stress_period_data=chd_rec,
                                          save_flows=True)

            # node property flow
            npf = ModflowGwfnpf(self.gwf,
                save_flows=True,
                icelltype=1,
                k=self.hk
                )

            rch = ModflowGwfrcha(self.gwf, recharge=self.rechTimeSeries[0],
                                 # stress_period_data={0: rd}
                                 )

            # output control, budget not needed without particle tracking
            oc = ModflowGwfoc(self.gwf,
                head_filerecord='{}.hds'.format(self.name),
                # budget_filerecord='{}.cbb'.format(name),
                headprintrecord=[
                ('COLUMNS', 10, 'WIDTH', 15,
                 'DIGITS', 6, 'GENERAL')],
                saverecord=[('HEAD', 'ALL'),
                # ('BUDGET', 'ALL')
                ],
                # saverecord=[('HEAD', 'ALL')],
                # printrecord=[('HEAD', 'ALL'),
                # ('BUDGET', 'ALL')
                # ]
                )

            self.sim_steadyState = deepcopy(self.sim)
            # self.sim_steadyState = deepcopy(self.sim)

            # del(self.sim.tdis)
            self.sim.remove_package(tdis)
            # print(dir(self.sim))

            # create tdis package
            tdis = ModflowTdis(self.sim, time_units='DAYS', nper=self.nper,
                perioddata=self.tdis_rc)

            sto = ModflowGwfsto(self.gwf, save_flows=True, iconvert=1,
                                ss=1e-5, sy=0.2, transient={0: True})

            wd = []
            for iHelper in range(self.nHelperWells):
                wd.append(((int(0),
                            self.actionsDict['well' + str(iHelper) + 'iCol'],
                            self.actionsDict['well' + str(iHelper) + 'iRow']),
                           self.actionsDict['well' + str(iHelper) + 'Q']))
            ModflowGwfwel(self.gwf, # maxbound=1,
                          stress_period_data={0: wd}, save_flows=True
                          )

            self.gwf.remove_package(self.gwf.package_key_dict['rcha'])
            rch = ModflowGwfrcha(self.gwf, recharge=self.rechTimeSeries[0],
                                 )

        else:

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
            if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                self.ibound[:, 1:-1, 0] = -1
                self.ibound[:, 1:-1, -1] = -1
                self.ibound[:, 0, :] = -1
                self.ibound[:, -1, :] = -1
            elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
                self.ibound[:, :-1, 0] = -1
                self.ibound[:, :-1, -1] = -1
                self.ibound[:, -1, :] = -1

            if self.periodSteadiness:
                # steady-state case: assigning ones as initial heads as they will be solved for
                self.strt = ones((self.nLay, self.nRow, self.nCol),
                                 dtype=float32
                                 )
            elif self.periodSteadiness == False:
                # transient case: assigning last solution as initial heads
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
            elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
                self.strt[:, 1:-1, 0] = self.headSpecWest
                self.strt[:, 1:-1, -1] = self.headSpecEast
                self.strt[:, 0, :] = self.headSpecSouth
                self.strt[:, -1, :] = self.headSpecNorth

            self.mf_bas = ModflowBas(self.mf, ibound=self.ibound, strt=self.strt)

            # adding LPF package to the MODFLOW model
            # for cell-variable k field
            # hk = multiply(random.random_sample(np.shape(self.strt)), 10.)
            hk = 10.
            # hk = np.add(np.multiply(self.strt, 0.), 10)
            self.mf_lpf = ModflowLpf(self.mf, hk=hk, vka=10., ss=1e-05, sy=0.15, ipakcb=53)

            # why is this relevant for particle tracking?
            # stress_period_data = {}
            # for kper in range(self.periods):
            #     for kstp in range([self.periodSteps][kper]):
            #         # budget is necessary for particle tracking
            #         stress_period_data[(kper, kstp)] = ['save head',
            #                                             'save budget',
            #                                             # 'save drawdown',
            #                                             ]

            # print(stress_period_data)
            # this only saves first and last result of stress period
            stress_period_data = {(0,0): ['save head', 'save budget'], (0,self.periodSteps-1): ['save head', 'save budget']}
            # print(stress_period_data)

            # adding OC package to the MODFLOW model for output control
            self.mf_oc = ModflowOc(self.mf, stress_period_data=stress_period_data, # stress_period_data
                compact=True)

            # from flopy.modflow import ModflowPcgn, ModflowGmg, ModflowSms

            # adding PCG package to the MODFLOW model
            # termination criteria currently empirically set
            self.mf_pcg = ModflowPcg(self.mf)
            # self.mf_pcg = ModflowPcg(self.mf, hclose=1e-1, rclose=1e-1)
            # self.mf_pcg = ModflowPcgn(self.mf, close_h=self.hclose, close_r=self.rclose)
            # self.mf_pcg = ModflowPcg(self.mf)

            # for speed-ups try different solvers

            # too large termination criterion might cause unrealistic near-well particle trajectories
            # self.mf_pcg = ModflowPcgn(self.mf, close_h=1e-6, close_r=1e-6)
            # self.mf_pcg = ModflowPcgn(self.mf, close_h=self.hclose, close_r=self.rclose)

            # this solver seems slightly quicker
            # from flopy.modflow import ModflowGmg
            # self.mf_pcg = ModflowGmg(self.mf, hclose=1e-1, rclose=1e-1)

            # from flopy.modflow import ModflowSms
            # self.mf_pcg = ModflowSms(self.mf)

            # from flopy.modflow import ModflowDe4
            # self.mf_pcg = ModflowDe4(self.mf)

            # from flopy.modflow import ModflowNwt
            # self.mf_pcg = ModflowNwt(self.mf)

    def get_cellID(self, x, y):
        # function to get correct cell for use in BMI call

        iCol = int(x/self.delc)
        iRow = self.nrow-1 - int(y/self.delr)
        cellID = (iRow*self.ncol)+iCol+1

        return cellID

    def get_cellColRow(self, x, y):
        # function to get correct cell for use in BMI call

        iCol = int(x/self.delc)
        iRow = self.nrow-1 - int(y/self.delr)

        return iCol, iRow

    def changeActionDict(self, actionsDict, action):
        iOffset = 0
        offset = 6
        for iHelper in range(self.nHelperWells):
            actionsDict['well' + str(iHelper) + 'actiondxRight'] = action[iOffset*offset]
            actionsDict['well' + str(iHelper) + 'actiondxLeft'] = action[iOffset*offset + 1]
            actionsDict['well' + str(iHelper) + 'actiondyUp'] = action[iOffset*offset + 2]
            actionsDict['well' + str(iHelper) + 'actiondyDown'] = action[iOffset*offset + 3]
            actionsDict['well' + str(iHelper) + 'actiondQUp'] = action[iOffset*offset + 4]
            actionsDict['well' + str(iHelper) + 'actiondQDown'] = action[iOffset*offset + 5]
            iOffset += 1

        return actionsDict

    def getActionSpaceSize(self, actionsDict):
        return len(actionsDict.keys())

    def getActionValues(self, actionsDict):

        for iHelper in range(self.nHelperWells):
            dxRight = self.well_dxMax*actionsDict['well' + str(iHelper) + 'actiondxRight']
            dxLeft = self.well_dxMax*actionsDict['well' + str(iHelper) + 'actiondxLeft']
            x = actionsDict['well' + str(iHelper) + 'x'] + dxRight - dxLeft
            if x < 0. + 1.1*self.delr:
                x = 0. + 1.1*self.delr
            elif x > self.extentX - 1.1*self.delr:
                x = self.extentX - 1.1*self.delr
            actionsDict['well' + str(iHelper) + 'x'] = x

            dyUp = self.well_dyMax*actionsDict['well' + str(iHelper) + 'actiondyUp']
            dyDown = self.well_dyMax*actionsDict['well' + str(iHelper) + 'actiondyDown']
            y = actionsDict['well' + str(iHelper) + 'y'] + dyUp - dyDown
            if y < 0. + 0.1*self.delc:
                y = 0. + 0.1*self.delc
            elif y > self.extentY - 0.1*self.delc:
                y = self.extentY - 0.1*self.delc
            actionsDict['well' + str(iHelper) + 'y'] = y

            actionsDict['well' + str(iHelper) + 'cellID'] = self.get_cellID(x, y)

            dQUp = self.well_dQMax*actionsDict['well' + str(iHelper) + 'actiondQUp']
            dQDown = self.well_dQMax*actionsDict['well' + str(iHelper) + 'actiondQDown']
            Q = actionsDict['well' + str(iHelper) + 'Q'] + dQUp - dQDown
            if Q > self.maxQ:
                Q = self.maxQ
            elif Q < self.minQ:
                Q = self.minQ
            actionsDict['well' + str(iHelper) + 'Q'] = Q

        return actionsDict

    def add_lib_dependencies(self, lib_dependencies):
        # for an example, see:
        # https://github.com/JoerivanEngelen/wflow_modflow6_bmi
        # on Windows, install: http://mingw-w64.org/doku.php
        # http://win-builds.org/doku.php/download_and_installation_from_windows
        # print('lib_dependencies DEBUG IN XMIPYWRAPPER', lib_dependencies)
        # print('lib_path DEBUG IN XMIPYWRAPPER', lib_path)

        # bmi installation linux
        # https://github.com/MODFLOW-USGS/executables/releases/download/5.0/linux.zip
        # maybe replace locally:
        # or force to install current xmipy
        # https://github.com/Deltares/xmipy/blob/develop/xmipy/xmiwrapper.py
        # sudo install libifport
        # https://software.intel.com/content/www/us/en/develop/articles/redistributable-libraries-for-intel-c-and-fortran-2020-compilers-for-linux.html

        # use nightly builds
        # https://github.com/MODFLOW-USGS/modflow6-nightly-build/releases

        if lib_dependencies:
            if system() == 'Windows':
                for dep_path in lib_dependencies:
                    if dep_path not in environ['PATH']:
                        environ['PATH'] = dep_path + pathsep + environ['PATH']
            else:
                # Assume a Unix-like system
                for dep_path in lib_dependencies:
                    if dep_path not in environ['PATH']:
                        # environ['LD_LIBRARY_PATH'] = (dep_path + pathsep + environ['LD_LIBRARY_PATH'])
                        environ['PATH'] = (dep_path + pathsep + environ['PATH'])

    def runMODFLOW(self, check=False):
        """Execute forward groundwater flow simulation using MODFLOW."""

        # writing MODFLOW input files
        t0 = time()
        if self.timeStep >= 1:
            # https://stackoverflow.com/questions/59187633/load-existing-mf2005-model-into-flopy-and-change-parameters
            self.debugWritten = 'partial'
            # self.dis.write_file()
            # self.mf_lpf.write_file()
            # self.mf_oc.write_file()
            # self.mf_pcg.write_file()
            # it could help if there was a method to write only changes to this file
            # https://github.com/modflowpy/flopy/blob/fdcc35ff5a7e7a809d7d10eff914881351a8f738/flopy/modflow/mfbas.py
            self.mf_bas.write_file(check=False)
            self.wel.write_file()

            # this will force finding a solution to take twice as long with no visible difference in heads and particle tracking
            # self.mf.write_input()
        elif self.timeStep == 0:
            self.debugWritten = 'full'
            self.mf.write_input()
        self.tWrite = time() - t0

        # debugging model setup if enabled
        if check:
            self.check = self.mf.check(verbose=True)

        # running the MODFLOW model
        t0 = time()
        self.successMODFLOW, self.buff = self.mf.run_model(silent=True)
        self.tSimulate = time() - t0
        if not self.successMODFLOW:
            raise Exception('MODFLOW did not terminate normally.')
        # print('debug tSimulate', self.tSimulate)

        # loading simulation heads and times
        self.fnameHeads = join(self.modelpth, self.MODELNAME + '.hds')
        with HeadFile(self.fnameHeads) as hf:
            # shouldn't we pick the heads at a specific runtime?
            self.times = hf.get_times()
            # self.realTime = self.times[-1]
            self.heads = hf.get_data(totim=self.times[-1])

        # loading discharge data
        # self.fnameBudget = join(self.modelpth, self.MODELNAME + '.cbc')
        # with CellBudgetFile(self.fnameBudget) as cbf:
            # self.cbb = cbf
            # self.frf = self.cbb.get_data(text='FLOW RIGHT FACE')[0]
            # self.fff = self.cbb.get_data(text='FLOW FRONT FACE')[0]

    def runMODPATH(self):
        """Execute forward particle tracking simulation using MODPATH."""

        t0 = time()

        # this needs to be transformed, yet not understood why
        self.particleCoords[0] = self.extentX - self.particleCoords[0]

        if self.timeStep == 1:
            # creating MODPATH simulation objects
            # only needs to be created initially, when first simulation timestep has passed
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
            self.mp.write_input()

        # manipulating input file to contain custom particle location
        # refer to documentation https://pubs.usgs.gov/tm/6a41/pdf/TM_6A_41.pdf
        out, keepFlag = [], True
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
        self.tMPInitialize = time() - t0

        t0 = time()
        # running the MODPATH model
        self.mp.run_model(silent=True)
        self.tMPSimulate = time() - t0
        # print(self.timeStep, 'time MF write', self.tWrite, 'time MF sim', self.tSimulate,
        #     'timeMP init', self.tMPInitialize, 'time MP sim', self.tMPSimulate, self.debugWritten)

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

    def calculateGameRewardTrajectory(self, trajectories):
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
        self.rewardMaxSegment = (distanceFraction * self.rewardMax)

        # pathLengthRatio defines the fraction of the highest possible reward
        # if lengthActual != 0.:
        if lengthShortest > 0.:
            pathLengthRatio = lengthShortest / lengthActual
            # self.gameReward = self.rewardMaxSegment * \
            #     (pathLengthRatio**self.deviationPenaltyFactor)
            self.gameReward = self.rewardMaxSegment * (pathLengthRatio**self.deviationPenaltyFactor)
        elif lengthShortest == 0.:
        # elif lengthActual == 0.:
            # returning no reward if travelled neither forwards nor backwards
            pathLengthRatio = 1.0
            self.gameReward = 0.
        # negative reward for traveling backwards
        # potential problem: maybe reward for going backward and then going
        # less deviating way forward?
        if lengthShortest < 0.:
            # self.gameReward *= -1.0 * self.rewardMaxSegment * \
            #     (pathLengthRatio**2)
            distanceFraction = lengthShortest / self.distanceMax
            self.rewardMaxSegment = (distanceFraction * self.rewardMax)
            self.gameReward = -1.0 * 5 * numpyAbs(self.rewardMaxSegment)
            # self.gameReward = -1.0 * (5*self.gameReward)
            # self.gameReward *= -1.0 * self.gameReward

        return self.gameReward

    def calculateGameRewardHeadChange(self, targetField, realizedField):
        # reward function

        # t0 = time()
        reward = -1.*numpySum(numpyAbs(subtract(targetField, realizedField)))
        # tCalculateReward = time()-t0
        # print('time get reward', tCalculateReward)

        return reward

    def calculateGameRewardOperationCost(self, nWells, wellDict):
        # look at Qs

        # t0 = time()
        QsumPump = 0.
        QsumRecharge = 0.
        for i in range(nWells):
            w = str(i+1)
            Q = wellDict['wellQ'+w]
            if Q <= 0.:
                QsumPump += -1.*wellDict['wellQ'+w]
            else:
                QsumRecharge += wellDict['wellQ'+w]

        reward = -1.*QsumPump*self.costQm3dPump
        reward = -1.*QsumRecharge*self.costQm3dRecharge
        # tCalculateReward = time()-t0
        # print('time get reward', tCalculateReward)

        return reward

    def reset(self, _seed=None, MODELNAME=None, initWithSolution=None):
        """Reset environment with same settings but potentially new seed."""

        if initWithSolution == None:
            initWithSolution=self.initWithSolution
        
        if self.ENVTYPE in ['0s-c']:
            # self.model_ws = join(self.wrkspc, 'models', MODELNAME)
            self.model_ws = join(self.wrkspc, 'models', self.exdir)
            self.defineEnvironment(_seed)
            self.constructModel()
            if self.timeStep == self.nstp:
                # on repeated reset
                # to reset timesteps etc.
                self.stepInitial()

            # env = self.env
            # # resetting to unique temporary folder to enable parallelism
            # # Note: This will resimulate the initial environment state
            # env.reset(MODELNAME=MODELNAMETEMP, _seed=SEEDTEMP)

        else:

            env_config = {
                'ENVTYPE': self.ENVTYPE,
                'PATHMF2005': self.PATHMF2005,
                'PATHMP6': self.PATHMP6,
                'MODELNAME': self.MODELNAME if MODELNAME is None else MODELNAME,
                '_seed': _seed,
                'flagSavePlot': self.SAVEPLOT,
                'flagManualControl': self.MANUALCONTROL,
                'flagRender': self.RENDER,
                'nLay': self.nLay,
                'nRow': self.nRow,
                'nCol': self.nCol,
                'OBSPREP': self.OBSPREP,
                'initWithSolution': initWithSolution
                }

            # print('env_config reset', env_config)

            self.__init__(env_config=env_config)
            close()

        return array(self.observationsVectorNormalized)

    def render(self, mode='human', dpi=120):
        """Plot the simulation state at the current timestep.
        Displaying and/or saving the visualisation. The active display can take
        user input from the keyboard to control the environment.
        """

        if mode == 'human':
            returnFigure = False
        elif mode == 'rgb_array':
            returnFigure = True

        if self.SAVEPLOT:
            returnFigure = True

        if self.ENVTYPE in ['0s-c']:
            # def render(self, head_, model_name, fig, ax):
            # print('plot head_.shape', head_.shape)
            if self.timeStep == 0:
                self.fig = figure(figsize=(8, 8))
                self.ax = self.fig.add_subplot(1, 1, 1, aspect='equal')
                self.plotArrays = []
                # self.render(self.head_steadyState, self.name, self.fig, self.ax)

            head = self.head
            ax = self.ax

            ax.cla()
            ax.set_title('step ' + str(self.timeStep) + ', reward ' + '%.2f' % (float(self.rewardCurrent)))

            # fname = join(self.wrkspc, 'models', self.MODELNAME, self.MODELNAME + '.dis.grb')
            fname = join(self.model_ws, self.MODELNAME + '.dis.grb')
            grd = MfGrdFile(fname, verbose=False)
            mg = grd.modelgrid # get_modelgrid()
            modelmap = PlotMapView(modelgrid=mg, ax=ax) # modelgrid=mg
            modelmap.plot_grid(alpha=0.1)
            cmap = ListedColormap(['r','g',])

            try:
                import flopy.utils.binaryfile as bf
                fname = join(self.wrkspc, 'temp', self.MODELNAME, self.MODELNAME + '.hds')
                # print('plot fname', fname)
                # hdobj = flopy.utils.HeadFile(fname)
                hds = bf.HeadFile(fname)
                # times = hds.get_times()
                # head_loaded = hds.get_data(totim=times[-1])
                # print(head_loaded)
            except:
                pass

            if self.timeStep in self.stormStarts:
                iStorm = self.stormStarts.index(int(self.timeStep))
                ax.scatter(self.stormCentersX[iStorm],
                            self.stormCentersY[iStorm],
                            color='red', s=10, zorder=10)

            # head = head_.reshape((self.nlay, self.nrow, self.ncol))
            head = head.reshape((self.nlay, self.nrow, self.ncol))
            pa = modelmap.plot_array(head, vmax=self.chd_west, vmin=self.chd_east)
            cb = self.fig.colorbar(pa, ax=ax) # plt.

            for iHelper in range(self.nHelperWells):
                x = self.actionsDict['well' + str(iHelper) + 'x']
                y = self.actionsDict['well' + str(iHelper) + 'y']
                Q = self.actionsDict['well' + str(iHelper) + 'Q']
                if Q <= 0.0:
                    color = 'red'
                else:
                    color = 'blue'
                self.ax.scatter(x, y, color=color, s=5, zorder=10)

                label_nextToCircle = '{:.2f}'.format(Q)
                self.ax.text((x + 3.0), (y), label_nextToCircle,
                    fontsize=12, color=color, zorder=12, alpha=0.5)

            if not returnFigure:
                show(block=False)
                if self.timeStep in self.stormStarts:
                    waitforbuttonpress(timeout=(2.5))
                else:
                    waitforbuttonpress(timeout=(0.5))
            elif returnFigure:
                self.fig.canvas.draw()
                data = self.fig.canvas.tostring_rgb()
                rows, cols = self.fig.canvas.get_width_height()
                imarray = copy(frombuffer(data, dtype=uint8).reshape(cols, rows, 3))
                # self.fig.set_size_inches(s)
            cb.remove()

            if self.SAVEPLOT:
                self.plotArrays.append(imarray)
                # if self.env.done or self.timeStep == self.NAGENTSTEPS:
                if self.done or self.timeStep == self.nstp:
                    # print('DEBUG self.MODELNAME', self.MODELNAME)
                    pathGIF = join(self.wrkspc, 'runs', self.ANIMATIONFOLDER, self.MODELNAMEGENCOUNT + '.gif')
                    # print('pathGIF', pathGIF)
                    self.writeGIFtodisk(pathGIF, self.plotArrays, optimizeSize=True)

            if returnFigure:
                return imarray

        else:
            if self.timeStep == 0 or self.canvasInitialized == False:
                self.renderInitializeCanvas()
                self.plotfilesSaved = []
                self.plotArrays = []
                self.extent = (self.dRow / 2.0, self.extentX - self.dRow / 2.0,
                 self.extentY - self.dCol / 2.0, self.dCol / 2.0)
                self.fig.canvas.draw()

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
                imarray = copy(frombuffer(data, dtype=uint8).reshape(cols, rows, 3))
                self.fig.set_size_inches(s)

            # temp
            # self.fig.savefig('C:\\Users\\admin\\Downloads\\forGitHub\\' + str(self.timeStep) + '.png')

            if not returnFigure:
                if self.MANUALCONTROL:
                    self.renderUserInterAction()
                else:
                    if not self.MANUALCONTROL:
                        if self.RENDER:
                            self.fig.canvas.draw()
                            show(block=False)
                            pause(self.MANUALCONTROLTIME)
            elif returnFigure:
                self.plotArrays.append(imarray)
                if self.SAVEPLOT:
                    if self.done or self.timeStep == self.NAGENTSTEPS:
                        pathGIF = join(self.wrkspc, 'runs', self.ANIMATIONFOLDER, self.MODELNAME + '.gif')
                        self.writeGIFtodisk(pathGIF, self.plotArrays, optimizeSize=True)
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
        self.canvasInitialized = True

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

        color = self.renderGetWellColor(self.wellQ)
        wellBufferCircle = Circle((self.wellCoords[0], self.extentY - self.wellCoords[1]), (self.wellRadius),
          edgecolor=color, facecolor='none', fill=False, zorder=zorder, alpha=1.0, lw=1.0)
        self.ax2.add_artist(wellBufferCircle)
        alpha = self.renderGetWellCircleAlpha(self.wellQ, self.minQ)
        wellBufferCircleFilled = Circle((self.wellCoords[0], self.extentY - self.wellCoords[1]), (self.wellRadius),
          edgecolor='none', facecolor=color, fill=True, zorder=zorder, alpha=alpha)
        self.ax2.add_artist(wellBufferCircleFilled)

        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            wellCoords = []
            for i in range(self.nHelperWells):
                w = str(i+1)
                wellCoords.append(self.helperWells['wellCoords'+w])
            for i, c in enumerate(wellCoords):
                color = self.renderGetWellColor(self.helperWells['wellQ'+str(i+1)])
                wellBufferCircle = Circle((c[0], self.extentY - c[1]), (self.helperWellRadius),
                  edgecolor=color, facecolor='none', fill=False, zorder=zorder, alpha=1.0, lw=1.0)
                self.ax2.add_artist(wellBufferCircle)
                alpha = self.renderGetWellCircleAlpha(self.helperWells['wellQ'+str(i+1)], self.minQhelper)
                wellBufferCircleFilled = Circle((c[0], self.extentY - c[1]), (self.helperWellRadius),
                  edgecolor='none', facecolor=color, fill=True, zorder=zorder, alpha=alpha)
                self.ax2.add_artist(wellBufferCircleFilled)

    def renderTextOnCanvasPumpingRate(self, zorder=10):
        """Plot pumping rate on figure."""
        color = self.renderGetWellColor(self.wellQ)
        label_nextToCircle = str(int(self.wellQ)) + '\n' + r'm$^\mathrm{\mathsf{3}}$ d$^\mathrm{\mathsf{-1}}$'
        # self.ax2.text((self.wellX + 3.0), (self.extentY - self.wellY), label_nextToCircle,
        #     fontsize=12, color=color, zorder=zorder, alpha=0.5)

        class CurvedText(mtext.Text):
            """
            A text object that follows an arbitrary curve.
            https://stackoverflow.com/questions/19353576/curved-text-rendering-in-matplotlib
            """
            def __init__(self, x, y, text, axes, delimiter, **kwargs):
                super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

                axes.add_artist(self)

                # saving the curve:
                self.__x = x
                self.__y = y
                self.__zorder = self.get_zorder()
                self.delimiter = delimiter

                # creating the text objects
                self.__Characters = []
                texts = text.split(self.delimiter)
                # print(text, self.delimiter)
                # print(texts)
                for c in texts:
                    # print(c)
                    if c == ' ':
                        # making this an invisible 'a':
                        t = mtext.Text(0,0,'a')
                        t.set_alpha(0.0)
                    else:
                        t = mtext.Text(0,0,c, **kwargs)

                    self.__Characters.append((c,t))
                    axes.add_artist(t)


            # overloading some member functions, to assure correct functionality on update
            def set_zorder(self, zorder):
                super(CurvedText, self).set_zorder(zorder)
                self.__zorder = self.get_zorder()
                for c,t in self.__Characters:
                    t.set_zorder(self.__zorder+1)

            def draw(self, renderer, *args, **kwargs):
                """
                Overload of the Text.draw() function. Do not do
                do any drawing, but update the positions and rotation
                angles of self.__Characters.
                """
                self.update_positions(renderer)

            def update_positions(self,renderer):
                """
                Update positions and rotations of the individual text elements.
                """

                # determining the aspect ratio:
                # from https://stackoverflow.com/a/42014041/2454357

                # data limits
                xlim = self.axes.get_xlim()
                ylim = self.axes.get_ylim()
                # axis size on figure
                figW, figH = self.axes.get_figure().get_size_inches()
                # ratio of display units
                _, _, w, h = self.axes.get_position().bounds
                # final aspect ratio
                aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

                # points of the curve in figure coordinates:
                x_fig,y_fig = (
                    array(l) for l in zip(*self.axes.transData.transform([
                    (i,j) for i,j in zip(self.__x,self.__y)
                    ]))
                )

                import numpy as np
                # point distances in figure coordinates
                x_fig_dist = (x_fig[1:]-x_fig[:-1])
                y_fig_dist = (y_fig[1:]-y_fig[:-1])
                r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

                # arc length in figure coordinates
                l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

                # angles in figure coordinates
                rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
                degs = np.rad2deg(rads)

                rel_pos = 10
                for c,t in self.__Characters:
                    # finding the width of c:
                    t.set_rotation(0)
                    t.set_va('center')
                    bbox1  = t.get_window_extent(renderer=renderer)
                    w = bbox1.width
                    h = bbox1.height

                    # ignoring all letters that don't fit:
                    if rel_pos+w/2 > l_fig[-1]:
                        t.set_alpha(0.0)
                        rel_pos += w
                        continue

                    elif c != ' ':
                        t.set_alpha(1.0)

                    # finding the two data points between which the horizontal
                    # center point of the character will be situated
                    # left and right indices:
                    il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
                    ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

                    # if we exactly hit a data point:
                    if ir == il:
                        ir += 1

                    # how much of the letter width was needed to find il:
                    used = l_fig[il]-rel_pos
                    rel_pos = l_fig[il]

                    # relative distance between il and ir where the center
                    # of the character will be
                    fraction = (w/2-used)/r_fig_dist[il]

                    # setting the character position in data coordinates:
                    # interpolating between the two points:
                    x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
                    y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

                    # getting the offset when setting correct vertical alignment
                    # in data coordinates
                    t.set_va(self.get_va())
                    bbox2  = t.get_window_extent(renderer=renderer)

                    bbox1d = self.axes.transData.inverted().transform(bbox1)
                    bbox2d = self.axes.transData.inverted().transform(bbox2)
                    dr = np.array(bbox2d[0]-bbox1d[0])

                    from math import cos as mathCos
                    from math import sin as mathSin

                    # the rotation/stretch matrix
                    rad = rads[il]
                    rot_mat = np.array([
                        [mathCos(rad), mathSin(rad)*aspect],
                        [-mathSin(rad)/aspect, mathCos(rad)]
                    ])

                    # computing the offset vector of the rotated character
                    drp = np.dot(dr,rot_mat)

                    # setting final position and rotation:
                    t.set_position(np.array([x,y])+drp)
                    t.set_rotation(degs[il])

                    t.set_va('center')
                    t.set_ha('center')

                    # updating rel_pos to right edge of character
                    rel_pos += w-used

        from numpy import cos as numpyCos
        from numpy import sin as numpySin
        from numpy import pi as numpyPi
        # import math
        # import numpy as np
        N = 500
        self.wellSafetyZone_outer = [-1.0*self.wellRadius*numpyCos(linspace(0, 2*numpyPi, N)),
                                      1.0*self.wellRadius*numpySin(linspace(0, 2*numpyPi, N))]

        label_onCircle = str(int(abs(self.wellQ))).replace("", ";")[1: -1] + r'   ;m;$^\mathrm{\mathsf{3}}$;   ;d;$^\mathrm{\mathsf{-1}}$'
        text = CurvedText(
            x = self.wellSafetyZone_outer[0] + self.wellCoords[0],
            y = self.wellSafetyZone_outer[1] + self.extentY - self.wellCoords[1],
            text=label_onCircle, va = 'bottom', axes = self.ax2, delimiter = ';', fontsize=8, color=color)

        if self.ENVTYPE in ['4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            wellCoords = []
            for i in range(self.nHelperWells):
                w = str(i+1)
                wellCoords.append(self.helperWells['wellCoords'+w])
            for i, c in enumerate(wellCoords):
                Q = self.helperWells['wellQ'+str(i+1)]
                color = self.renderGetWellColor(self.helperWells['wellQ'+str(i+1)])
                helperWellSafetyZone_outer = [-1.0*self.helperWellRadius*numpyCos(linspace(0, 2*numpyPi, N)),
                                              1.0*self.helperWellRadius*numpySin(linspace(0, 2*numpyPi, N))]

                label_onCircle = str(int(abs(Q))).replace("", ";")[1: -1] # + r' ;m;$^\mathrm{\mathsf{3}}$; ;d;$^\mathrm{\mathsf{-1}}$'
                text = CurvedText(
                    x = helperWellSafetyZone_outer[0] + c[0],
                    y = helperWellSafetyZone_outer[1] + self.extentY - c[1],
                    text=label_onCircle, va = 'bottom', axes = self.ax2, delimiter = ';', fontsize=5, color=color)

    def renderGetWellColor(self, Q):
        """Return color symbolizing pumping or injection well regime."""
        if Q < 0.:
            color = 'red'
        else:
            color = 'blue'

        return color

    def renderGetWellCircleAlpha(self, Q, maxQ):
        """Return transparency level representing pumping or injection magnitude."""

        alphaMax = 0.9
        Q = abs(Q)
        fraction = Q/abs(maxQ)
        alpha = fraction*alphaMax

        return alpha

    def renderTextOnCanvasGameOutcome(self, zorder=10):
        """Plot final game outcome on figure."""
        gameResult = ''
        if self.done:
            if self.success:
                gameResult = 'Success'
            elif self.success == False:
                gameResult = 'Failure'
            self.text = self.ax2.text(1.25, 65., gameResult, fontsize=500,
                color='red', zorder=zorder, alpha=0.25,
                bbox=dict(facecolor='none', edgecolor='none', pad=0.0))

            self.textSpanAcrossAxis(self.text, 100., 80., fig=self.fig, ax=self.ax2)

    def textSpanAcrossAxis(self, text, width, height, fig=None, ax=None):
        """Auto-decrease and re-expand the fontsize of a text object to match the axis extent.
        https://stackoverflow.com/questions/5320205/matplotlib-text-dimensions
        Args:
            text (matplotlib.text.Text)
            width (float): allowed width in data coordinates
            height (float): allowed height in data coordinates
        """

        size = fig.get_size_inches()
        whratio = size[0]/size[1]

        # get text bounding box in figure coordinates
        # bbox_text = text.get_window_extent().inverse_transformed(self.fig.gca().transData)
        # https://stackoverflow.com/questions/23032847/matplotlib-image-get-window-extentrenderer-produces-all-zeros
        renderer = fig.canvas.renderer
        text.draw(renderer)
        bbox_text = text.get_window_extent().transformed(ax.transData.inverted())
        # transformed(transform.inverted())

        # evaluate fit and recursively decrease fontsize until text fits
        fits_width = bbox_text.width*whratio < width if width else True
        fits_height = bbox_text.height/whratio < height if height else True
        dFontDecrease = 5.
        if not all((fits_width, fits_height)):
            text.set_fontsize(text.get_fontsize()-dFontDecrease)
            self.textSpanAcrossAxis(text, width, height, fig, ax)

        # re-expanding (in finer increments)
        expandFinished = False
        while not expandFinished:
            text.draw(renderer)
            bbox_text = text.get_window_extent().transformed(ax.transData.inverted())
            fits_width = bbox_text.width*whratio >= width if width else True
            fits_height = bbox_text.height/whratio < height if height else True
            if not all((fits_width, fits_height)):
                dFontIncrease = 0.1
                text.set_fontsize(text.get_fontsize()+dFontIncrease)
            else:
                expandFinished = True

    def renderTextOnCanvasTimeAndScore(self, zorder=10):
        """Plot final game outcome on figure."""
        timeString = '%.0f' % (float(self.timeStep) * self.periodLength)
        if self.ENVTYPE in ['5s-c-cost']:
            self.ax2.text(5, 92, ('score: ' + str('%.2f' % self.rewardCurrent) + ' €' + '\ntime: ' + timeString + ' d'),
              fontsize=12,
              zorder=zorder)
        else:
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
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
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
        left, width = self.minX, numpyAbs(self.minX + self.extentX)
        bottom, height = self.minY, numpyAbs(self.minY + self.extentY)
        top = bottom + height
        textRight = 'Destination:   ' + str('%.2f' % self.headSpecEast) + ' m'
        textLeft = 'Start:   ' + str('%.2f' % self.headSpecWest) + ' m'
        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c']:
            textTop = str('%.2f' % self.actionValueNorth) + ' m'
            textBottom = str('%.2f' % self.actionValueSouth) + ' m'
        elif self.ENVTYPE in ['2s-d', '2s-c', '2r-d', '2r-c']:
            textTop = ''
            textBottom = str('%.2f' % self.actionValue) + ' m'
        elif self.ENVTYPE in ['3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
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
            print(self.wrkspc, 'runs', self.ANIMATIONFOLDER)
            self.plotspth = join(self.wrkspc, 'runs', self.ANIMATIONFOLDER)
            if not exists(self.plotsfolderpth):
                makedirs(self.plotsfolderpth)
            if not exists(self.plotspth):
                makedirs(self.plotspth)
        plotfile = join(self.plotspth, self.MODELNAME + '_' + str(self.timeStep).zfill(len(str(numpyAbs(self.NAGENTSTEPS))) + 1) + '.png')

        s = self.fig.get_size_inches()

        self.fig.gca().set_axis_off()
        margins(0,0)
        self.fig.gca().xaxis.set_major_locator(NullLocator())
        self.fig.gca().yaxis.set_major_locator(NullLocator())
        self.fig.tight_layout(pad=0.)
        self.fig.set_size_inches(7, 7)
        imarray = self.figToArray(self.fig)
        self.fig.set_size_inches(s)
        self.plotArrays.append(imarray)

    def figToArray(self, fig):
        fig.canvas.draw()
        data = fig.canvas.tostring_rgb()
        rows, cols = fig.canvas.get_width_height()
        imarray = frombuffer(data, dtype=uint8).reshape(cols, rows, 3)
        
        return imarray

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

        if self.ENVTYPE in ['1s-d', '1s-c', '1r-d', '1r-c', '3s-d', '3s-c', '3r-d', '3r-c', '4s-d', '4s-c', '4r-d', '4r-c', '5s-d', '5s-c', '5s-c-cost', '5r-d', '5r-c', '6s-d', '6s-c', '6r-d', '6r-c']:
            try:
                self.ax3.cla()
                self.ax3.clear()
            except:
                pass

    def writeGIFtodisk(self, pathGIF, ims, fps=2, optimizeSize=False):
        with get_writer(pathGIF, mode='I', fps=fps) as writer:
            for image in ims:
                writer.append_data(image)

        if optimizeSize:
            try:
                from pygifsicle import optimize as optimizeGIF
                imported_pygifsicle = True
            except:
                imported_pygifsicle = False
                print('Module pygifsicle not properly installed. Returning uncompressed animation.')
            if imported_pygifsicle:
                    optimizeGIF(pathGIF)

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
        for i in range(n):
            if i > 0:
                lv.append(sqrt((x[i] - x[(i - 1)]) ** 2 + (y[i] - y[(i - 1)]) ** 2))

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
        elif self.ENVTYPE in ['4s-c', '4r-c', '5s-c', '5s-c-cost', '5r-c', '6s-c', '6r-c']:
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
                if self.ENVTYPE in ['5s-c', '5s-c-cost', '5r-c', '6s-c', '6r-c']:
                    if self.ENVTYPE in ['5s-c', '5s-c-cost', '5r-c']:
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
        if self.OBSPREP == 'perceptron':
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
        elif self.OBSPREP == 'convolution':
            valsExtra = []
            if 'particleCoords' in observationsDict.keys():
                for obs in observationsDict['particleCoords']:
                    valsExtra.append(obs)
            if 'heads' in observationsDict.keys():
                observationsVector = array(observationsDict['heads'])
            if 'wellQ' in observationsDict.keys():
                valsExtra.append(observationsDict['wellQ'])
            if 'wellCoords' in observationsDict.keys():
                for obs in observationsDict['wellCoords']:
                    valsExtra.append(obs)
            if self.nHelperWells > 0:
                for i in range(self.nHelperWells):
                    iStr = str(i+1)
                    if 'wellQ'+iStr in observationsDict.keys():
                        valsExtra.append(observationsDict['wellQ'+iStr])
                    if 'wellCoords'+iStr in observationsDict.keys():
                        for obs in observationsDict['wellCoords'+iStr]:
                            valsExtra.append(obs)
            for val in valsExtra:
                add_ = multiply(observationsDict['heads'][0, :, :], 0.)
                add_ = add(add_, val)
                add_ = expand_dims(add_, axis=0)
                observationsVector = concatenate((observationsVector, add_), axis=0)

        return observationsVector

    def observationsVectorToDict(self, observationsVector):
        """Convert list of observations to dictionary."""
        observationsDict = {}
        if self.OBSPREP == 'perceptron':
            # offset to capture helper well data
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
        elif self.OBSPREP == 'convolution':
            observationsDict['particleCoords'] = [
                observationsVector[0, 0, 0],
                observationsVector[1, 0, 0],
                observationsVector[2, 0, 0]]
            observationsDict['heads'] = observationsVector[3, :, :]
            observationsDict['wellQ'] = observationsVector[4, 0, 0]
            observationsDict['wellCoords'] = [
                observationsVector[5, 0, 0],
                observationsVector[6, 0, 0],
                observationsVector[7, 0, 0]]
            if self.nHelperWells > 0:
                for i in range(self.nHelperWells):
                    # print('debug i', i)
                    iStr = str(i+1)
                    offset = i*4
                    observationsDict['wellQ'+iStr] = observationsVector[8+offset, 0, 0]
                    observationsDict['wellCoords'+iStr]= [
                        observationsVector[9+offset, 0, 0],
                        observationsVector[10+offset, 0, 0],
                        observationsVector[11+offset, 0, 0]]

        return observationsDict

    def normalize(self, array, zerooffset, diff):

        array = divide(subtract(array, zerooffset), diff)

        return array

    def unnormalize(self, data):
        # from numpy import multiply

        keys = data.keys()
        if 'particleCoords' in keys:
            data['particleCoords'] = multiply(data['particleCoords'],
                numpyAbs(self.minX + self.extentX))
        if 'heads' in keys:
            data['heads'] = multiply(data['heads'],
                self.maxH)
        if 'wellQ' in keys:
            data['wellQ'] = multiply(data['wellQ'], self.minQ)
        if 'wellCoords' in keys:
            data['wellCoords'] = multiply(data['wellCoords'], numpyAbs(self.minX + self.extentX))
        if 'rewards' in keys:
            data['rewards'] = multiply(data['rewards'], self.rewardMax)
        return data

class FloPyArcade():
    """Instance of a FloPy arcade game.

    Initializes a game agent and environment. Then allows to play the game.
    """

    def __init__(self, agent=None, modelNameLoad=None, modelName='FloPyArcade',
        animationFolder=None, NAGENTSTEPS=200, PATHMF2005=None, PATHMP6=None,
        surrogateSimulator=None, flagSavePlot=False,
        flagManualControl=False, actions=None, flagRender=False,
        keepTimeSeries=False, ENVTYPE='1s-d', nLay=1, nRow=100, nCol=100,
        OBSPREP='perceptron'):
        """Constructor."""

        self.ENVTYPE = str(ENVTYPE)
        self.PATHMF2005 = PATHMF2005
        self.PATHMP6 = PATHMP6
        self.SURROGATESIMULATOR = surrogateSimulator
        self.NAGENTSTEPS = NAGENTSTEPS
        self.SAVEPLOT = flagSavePlot
        self.MANUALCONTROL = flagManualControl
        self.RENDER = flagRender
        self.MODELNAME = modelName if modelName is not None else (modelNameLoad if modelNameLoad is not None else "untitled")
        self.ANIMATIONFOLDER = animationFolder if modelName is not None else (modelNameLoad if modelNameLoad is not None else "untitled")
        self.agent = agent
        self.MODELNAMELOAD = modelNameLoad
        self.done = False
        self.keepTimeSeries = keepTimeSeries
        self.actions = actions
        self.nLay, self.nRow, self.nCol = nLay, nRow, nCol
        self.OBSPREP = OBSPREP
        self.ENVTYPES = ['1s-d', '1s-c', '1r-d', '1r-c',
                         '2s-d', '2s-c', '2r-d', '2r-c',
                         '3s-d', '3s-c', '3r-d', '3r-c',
                         '4s-c', '4r-c', # '4s-d', '4r-d'
                         '5s-c', '5s-c-cost', '5r-c', # '5s-d', '5r-d'
                         '6s-c', '6r-c'  # '6s-d', '6r-d'
                         ]
        self.wrkspc = FloPyAgent().wrkspc
        # if this is called, the rest instantiation in play will not take arguments
        # self.wrkspc = FloPyEnv(initWithSolution=False).wrkspc

    def play(self, env=None, ENVTYPE='1s-d', seed=None, returnReward=False, verbose=False):
        """Play an instance of the Flopy arcade game."""

        t0 = time()

        # print(self.MANUALCONTROL)
        # print(env, self.ENVTYPE)

        # creating the environment
        if env is None:
            if self.SURROGATESIMULATOR is None:
                # print('created', type(self.ENVTYPE))
                self.env = FloPyEnv(
                    ENVTYPE=self.ENVTYPE,
                    PATHMF2005=self.PATHMF2005, PATHMP6=self.PATHMP6,
                    _seed=seed,
                    MODELNAME=self.MODELNAME if not None else 'FloPyArcade',
                    ANIMATIONFOLDER=self.ANIMATIONFOLDER if not None else 'FloPyArcade',
                    flagSavePlot=self.SAVEPLOT,
                    flagManualControl=self.MANUALCONTROL,
                    flagRender=self.RENDER,
                    NAGENTSTEPS=self.NAGENTSTEPS,
                    nLay=self.nLay,
                    nRow=self.nRow,
                    nCol=self.nCol,
                    OBSPREP=self.OBSPREP,
                    initWithSolution=True)
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

                verbose = True

                if self.actions is not None and self.actions != []:
                    action = self.actions[self.timeSteps]

                t0step = time()
                observations, reward, self.done, _ = self.env.step(action)

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
                    # self.rewardTotal = 0.0

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
                          str(self.timeSteps+1) +
                          ' timesteps with a reward of ' +
                          str(int(self.rewardTotal)) +
                          ' points.')
                close('all')

                if self.env.ENVTYPE in ['0s-c']:
                    self.env.teardown()

                break

        self.runtime = (time() - t0) / 60.

        if returnReward:
            return self.rewardTotal