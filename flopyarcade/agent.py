#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade
# provides: simulated groundwater flow environments to test reinforcement learning algorithms
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


# imports for environments
from builtins import min as py_min, max as py_max
from copy import copy, deepcopy
from ctypes import sizeof, c_voidp
import gc
import hashlib
from matplotlib import use as matplotlibBackend
from matplotlib.ticker import NullLocator
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
from flopy.utils import HeadFile, PathlineFile
from flopy.mf6.utils.binarygrid_util import MfGrdFile
from glob import glob
import warnings
warnings.filterwarnings('ignore', 'SelectableGroups dict interface')
warnings.filterwarnings("ignore", category=ResourceWarning)
# warnings.filterwarnings("ignore", category=ResourceWarning, message="Implicitly cleaning up TemporaryDirectory")

import gymnasium as gym
from gymnasium import Env as gymEnv
from gymnasium import spaces
# from gymnasium.wrappers import EnvCompatibility
from imageio import get_writer, imread
from itertools import chain, product
from joblib import dump as joblibDump
from joblib import load as joblibLoad
from math import ceil, floor
from math import cos as mathCos
from math import sin as mathSin
from matplotlib import colormaps
# from matplotlib.colormaps import get_cmap
# from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import Circle, close, figure, pause, show
from matplotlib.pyplot import imsave
from matplotlib.pyplot import get_current_fig_manager
from matplotlib.pyplot import margins, NullLocator
from matplotlib.pyplot import waitforbuttonpress
from matplotlib import text as mtext
from matplotlib.transforms import Bbox
import numpy as np
from numpy import cos as numpyCos
from numpy import sin as numpySin
from numpy import pi as numpyPi
# from numpy import chararray
from numpy import add, arange, argmax, argsort, array, ceil, copy, concatenate, divide, empty
from numpy import expand_dims, extract, float32, frombuffer, int32, linspace, max, maximum, min, minimum
from numpy import mean, multiply, ones, prod, reshape, shape, sqrt, subtract, uint8, unique, zeros
from numpy import sum as numpySum
from numpy import abs as numpyAbs
from numpy.random import randint, random, randn, uniform
from numpy.random import seed as numpySeed
from os import chdir, chmod, environ, listdir, makedirs, pathsep, remove, rmdir, unlink
from os.path import abspath, dirname, exists, join, sep
from platform import system as platformSystem
from shutil import rmtree
from shutil import copy2
from sys import modules
if 'ipykernel' in modules:
    from IPython import display
from time import sleep, time
from xmipy import XmiWrapper

import flopy.utils.binaryfile as bf


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
from tensorflow.keras.random import SeedGenerator
from random import sample as randomSample, seed as randomSeed
from random import shuffle
from tensorflow.compat.v1 import ConfigProto, set_random_seed
from tensorflow.compat.v1 import Session as TFSession
from tensorflow.keras import backend as K
# from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import load_model as TFload_model
from tqdm import tqdm
from uuid import uuid4

from tempfile import TemporaryDirectory
from tempfile import NamedTemporaryFile

# avoiding freeze issues on Linux when loading Tensorflow model
# https://github.com/keras-team/keras/issues/9964
# https://stackoverflow.com/questions/40615795/pathos-enforce-spawning-on-linux
pathosHelpers.mp.context._force_start_method('spawn')

# to avoid thread erros
# https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
# matplotlibBackend('Agg')




from .flopyarcade import FloPyEnv
from .flopyarcade import FloPyEnv as FloPyEnvSurrogate
from .arcade import FloPyArcade


class FloPyAgent:
    def __init__(self, observationsVector=None, actionSpace=['keep'],
                 hyParams=None, envSettings=None, mode='random',
                 maxTasksPerWorker=1000, maxTasksPerWorkerMutate=1000,
                 maxTasksPerWorkerNoveltySearch=10000, zFill=6):
        self.wrkspc = self._get_workspace()
        self.observationsVector = observationsVector
        self.hyParams, self.envSettings = hyParams, envSettings
        self.actionSpace = actionSpace
        self.actionSpaceSize = len(actionSpace)
        self.agentMode = mode
        self.maxTasksPerWorker = maxTasksPerWorker
        self.maxTasksPerWorkerMutate = maxTasksPerWorkerMutate
        self.maxTasksPerWorkerNoveltySearch = maxTasksPerWorkerNoveltySearch
        self.zFill = zFill

        # print(envSettings)

        if envSettings:
            self.actionType = self.getActionType(envSettings['ENVTYPE'])
            self.setSeeds()

        self._create_folder(join(self.wrkspc, 'models'), assign_attr='modelpth')

        if self.agentMode == 'DQN':
            self.initializeDQNAgent()
        elif self.agentMode == 'genetic':
            self._setup_genetic_agent()

    def _get_workspace(self):
        wrkspc = dirname(abspath(__file__))
        if 'library.zip' in wrkspc:
            wrkspc = dirname(dirname(wrkspc))
        return wrkspc

    def _create_folder(self, path, assign_attr=None):
        if not exists(path):
            makedirs(path)
        if assign_attr:
            setattr(self, assign_attr, path)

    def _setup_genetic_agent(self):
        if self.envSettings['NAGENTSPARALLEL'] is None:
            self.envSettings['NAGENTSPARALLEL'] = cpu_count()

        self.tempModelpth = join(self.wrkspc, 'temp', self.envSettings['MODELNAME'])

        if not self.envSettings['RESUME']:
            self._cleanup_old_models()

        self._create_folder(self.tempModelpth)

        if self.envSettings['BESTAGENTANIMATION']:
            self._create_folder(join(self.wrkspc, 'runs', self.envSettings['MODELNAME']))

        if self.hyParams and self.hyParams.get('NOVELTYSEARCH', False):
            self._setup_novelty_path()

        self._initialize_seeds()

        if not self.envSettings['RESUME']:
            self.geneticGeneration = 0
            print('Initializing genetic agents.')
            self.initializeGeneticAgents()
            self.pickleDump(join(self.tempModelpth, f"{self.envSettings['MODELNAME']}_hyParams.p"), self.hyParams)
            self.pickleDump(join(self.tempModelpth, f"{self.envSettings['MODELNAME']}_envSettings.p"), self.envSettings)
        else:
            self.hyParams = self.pickleLoad(join(self.tempModelpth, f"{self.envSettings['MODELNAME']}_hyParams.p"))

    def _cleanup_old_models(self):
        pattern = join(self.modelpth, f"{self.envSettings['MODELNAME']}_*")
        for item in glob(pattern):
            try:
                remove(item)
            except:
                for f in glob(join(item, '*')):
                    remove(f)
        try:
            rmtree(self.tempModelpth)
        except:
            pass

    def _setup_novelty_path(self):
        self.tempNoveltypth = join(self.tempModelpth, 'novelties')
        if not exists(self.tempNoveltypth):
            makedirs(self.tempNoveltypth)
        elif not self.envSettings['RESUME']:
            for f in listdir(self.tempNoveltypth):
                remove(join(self.tempNoveltypth, f))

    def _initialize_seeds(self):
        # self.setSeeds()
        self.mutationSeeds = [list(np.random.randint(1, 10_000_000, self.hyParams['NAGENTS']))
                              for _ in range(self.hyParams['NGENERATIONS'])]

        if self.envSettings['SEEDSRANDOM']:
            self.gamesSeeds = [list(np.random.randint(1, 10_000_000, self.hyParams['NGAMESAVERAGED']))
                               for _ in range(self.hyParams['NGENERATIONS'])]
            self.bestRewards = []
            # self.setSeeds()

    def initializeDQNAgent(self):
        self._init_seeds()
        actionType = FloPyEnv(initWithSolution=False).getActionType(self.envSettings['ENVTYPE'])
        self.mainModel = self.createNNModel(actionType)
        self.targetModel = self.createNNModel(actionType)
        self.targetModel.set_weights(self.mainModel.get_weights())
        self.replayMemory = deque(maxlen=self.hyParams['REPLAYMEMORYSIZE'])
        self.epsilon = self.hyParams['EPSILONINITIAL']
        self.targetUpdateCount = 0

    def initializeGeneticAgents(self):
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
                self.mutationHistory = self.updateMutationHistory(self.mutationHistory, agentNumber, creationSeed, mutationSeeds=[], mutationSeed=None)

    def setSeeds(self):
        print('Setting SEEDAGENT seed', self.envSettings['SEEDAGENT'])
        self.SEED = self.envSettings['SEEDAGENT']
        numpySeed(self.SEED)
        randomSeed(self.SEED)
        set_random_seed(self.SEED)

    def getActionType(self, ENVTYPE):
        if '-d' in ENVTYPE:
            actionType = 'discrete'
        elif '-c' in ENVTYPE:
            actionType = 'continuous'
        else:
            print('Environment name is unknown.')
            quit()

        return actionType

    def runDQN(self, env):
        """Run main Deep Q-Learning optimization pipeline."""
        self.actionType = env.actionType
        # self._init_seeds()
        self.gameSeeds = [randint(low=0, high=np.iinfo(np.int32).max) for iGame in range(self.hyParams['NGAMES'] + 1)]
        self.seedsCV = randint(low=0, high=np.iinfo(np.int32).max, size=self.hyParams['NGAMESCROSSVALIDATED'])
        # print('DEBUG self.gameSeeds', self.gameSeeds)
        gameRewards = []
        for iGame in tqdm(range(1, self.hyParams['NGAMES'] + 1), ascii=True, unit='games'):
            env.reset(seed=self.gameSeeds[iGame], MODELNAME=env.MODELNAME) # _seed=
            self._play_one_game(env)
            gameRewards.append(self.gameReward)

            if iGame == 1 or iGame % self.hyParams['CROSSVALIDATEEVERY'] == 0:
                self._crossvalidate_and_save(env, iGame)

            self._decay_epsilon()

    def runGenetic(self, env, searchNovelty=False):
        """Run main pipeline for genetic agent optimization."""
        self._validate_hyperparams()
        self._initialize_environment(env, searchNovelty)
        self._initialize_parallel_settings()
        self._generate_unique_pid()

        # Further genetic algorithm steps would follow here...
        # e.g. population initialization, evaluation, selection, mutation, etc.

        n = self.n

        agentCounts = list(range(self.hyParams['NAGENTS']))
        self.rereturnChildrenGenetic = False

        for self.geneticGeneration in range(self.hyParams['NGENERATIONS']):
            self.flagSkipGeneration = False
            self.generatePathPrefixes()

            if self.envSettings['RESUME']:
                if self.searchNovelty and self.geneticGeneration > 0:
                    self._load_novelty_archive()
                sortedParentIdxs, continueFlag, breakFlag = self.resumeGenetic()
                if continueFlag:
                    continue
                if breakFlag:
                    break
                if self.searchNovelty:
                    self._regenerate_unique_duplicate_lists(env)

            print(f'########## started generation {str(self.geneticGeneration+1).zfill(self.zFill)} ##########')

            self.rewards = self.runAgentsRepeatedlyGenetic(agentCounts, self.n, env)
            self._log_generation_rewards()

            sortedParentIdxs = self._sort_and_save_top_agents()
            self.bestAgentReward = self.rewards[sortedParentIdxs[0]]

            self.perform_novelty_search(env)

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
            #             str(agentIdx + 1).zfill(self.zFill) + '.keras'))

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

            print('########## finished generation ' + str(self.geneticGeneration+1).zfill(self.zFill) + ' ##########')

    def _init_seeds(self):
        """Initialize seeds for reproducible cross-validation."""
        self.SEED = self.envSettings['SEEDAGENT']
        numpySeed(self.envSettings['SEEDAGENT'])
        randomSeed(self.envSettings['SEEDAGENT'])
        set_random_seed(self.envSettings['SEEDAGENT'])

    def _play_one_game(self, env):
        """Simulate one game, update replay memory, and train."""
        self.takeActionsUpdateAndTrainDQN(env)
        if not env.success:
            self.updateReplayMemoryZeroReward(self.gameStep)

    def _crossvalidate_and_save(self, env, iGame):
        """Run cross-validation and save model if performance threshold met."""
        self.crossvalidateDQN(env)
        MODELNAME = self.envSettings['MODELNAME']
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = (
            f"{MODELNAME}{iGame:7d}ep"
            f"{self.max_rewardCV:7.1f}max"
            f"{self.average_rewardCV:7.1f}avg"
            f"{self.min_rewardCV:7.1f}min"
            f"{timestamp}datetime.keras"
        )
        if self.average_rewardCV >= self.envSettings['REWARDMINTOSAVE']:
            self.mainModel.save(join(self.wrkspc, 'models', filename))

        print('DQN results', self.max_rewardCV, self.average_rewardCV, self.min_rewardCV)

    def _decay_epsilon(self):
        """Decay exploration rate epsilon."""
        if self.epsilon > self.hyParams['EPSILONMIN']:
            self.epsilon = py_max(self.hyParams['EPSILONMIN'], self.epsilon * self.hyParams['EPSILONDECAY'])

    def _validate_hyperparams(self):
        if self.hyParams['NAGENTS'] <= self.hyParams['NNOVELTYELITES']:
            raise ValueError(
                'The number of novelty elites must be less than the number of agents.'
            )

    def _initialize_environment(self, env, searchNovelty):
        self.env = env
        self.actionType = env.actionType
        self.n = self.hyParams['NGAMESAVERAGED']

        if searchNovelty:
            self.searchNovelty = True
            self.noveltyArchive = {}
            self.noveltyItemCount = 0
            self.agentsUnique = []
            self.agentsUniqueIDs = []
            self.agentsDuplicate = []
            self.agentsDuplicateID = []
            if env.actionType == 'discrete':
                self.actionsUniqueIDMapping = defaultdict(count().__next__)

    def _initialize_parallel_settings(self):
        self.cores = self.envSettings.get('NAGENTSPARALLEL', 1)

    def _generate_unique_pid(self):
        pid = str(uuid4())
        pid_list = list(pid)
        shuffle(pid_list)
        self.pid = ''.join(pid_list)

    def _load_novelty_archive(self):
        self.noveltyArchive = self.pickleLoad(join(self.tempPrevModelPrefix + '_noveltyArchive.p'))
        if not self.envSettings['KEEPMODELHISTORY']:
            self.mutationHistory = self.pickleLoad(join(self.tempPrevModelPrefix + '_mutationHistory.p'))
        self.noveltyItemCount = len(self.noveltyArchive)

    def _regenerate_unique_duplicate_lists(self, env):
        self.agentsUnique, self.agentsUniqueIDs = [], []
        self.agentsDuplicate = []
        for iAgent in range(self.noveltyItemCount):
            agentStr = f'agent{iAgent+1}'
            tempAgentPrefix = self.noveltyArchive[agentStr]['modelFile'].replace('.keras', '')
            resultsPath = join(tempAgentPrefix + '_results.p')
            actions = self.pickleLoad(resultsPath)['actions']
            if env.actionType == 'discrete':
                actionsAll = [a for sublist in actions for a in sublist]
                actionsUniqueID = self.actionsUniqueIDMapping[tuple(actionsAll)]
                self.noveltyArchive[agentStr]['actionsUniqueID'] = actionsUniqueID
                if actionsUniqueID not in self.agentsUniqueIDs:
                    self.agentsUnique.append(iAgent)
                    self.agentsUniqueIDs.append(actionsUniqueID)
                else:
                    self.agentsDuplicate.append(iAgent)

    def _log_generation_rewards(self):
        print('lowest reward', min(self.rewards))
        print('average reward', mean(self.rewards))
        print('highest reward', max(self.rewards))
        if self.envSettings['SEEDSRANDOM']:
            self.bestRewards.append(max(self.rewards))
            print('highest sliding reward (n=50)', mean(self.bestRewards[-50:]))

    def _sort_and_save_top_agents(self):
        sortedParentIdxs = argsort(self.rewards)[::-1][:self.hyParams['NAGENTELITES']]
        self.pickleDump(join(self.tempModelPrefix + '_agentsSortedParentIndexes.p'), sortedParentIdxs)
        return sortedParentIdxs

    def perform_novelty_search(self, env):
        if not self.searchNovelty:
            return

        print('Performing novelty search')
        t0 = time()
        self._prepare_novelty_archive(env)
        print('Preparing novelty search took', round(time() - t0, 2), 's')

        self._print_novelty_stats()

        t0 = time()
        args, update_flags = self._build_novelty_args()
        print('Calculated novelty search arguments, took', round(time() - t0, 2), 's')

        print('Started novelty search ...')
        t0 = time()
        novelties = self._run_novelty_search(args)
        print('Finished novelty search, took', round(time() - t0, 2), 's')

        self._update_agent_novelties(novelties, update_flags)
        self._save_novelties()

    def _prepare_novelty_archive(self, env):
        for i in range(self.hyParams['NAGENTS']):
            agent_str = f'agent{self.noveltyItemCount+1}'
            temp_prefix = join(self.tempModelPrefix + f'_agent{str(i+1).zfill(self.zFill)}')
            model_file, results_file = temp_prefix + '.keras', temp_prefix + '_results.p'

            self.noveltyArchive[agent_str] = {
                'itemID': self.noveltyItemCount,
                'modelFile': model_file,
                'resultsFile': results_file
            }

            if env.actionType == 'discrete':
                actions = self.pickleLoad(results_file)['actions']
                flat_actions = [a for sub in actions for a in sub]
                unique_id = self.actionsUniqueIDMapping[tuple(flat_actions)]
                self.noveltyArchive[agent_str]['actionsUniqueID'] = unique_id
                self._update_discrete_uniques(unique_id)
            else:
                self._reset_continuous_uniques()

            self.noveltyItemCount += 1

    def _update_discrete_uniques(self, unique_id):
        k = self.noveltyItemCount
        if unique_id not in self.agentsUniqueIDs:
            self.agentsUnique.append(k)
            self.agentsUniqueIDs.append(unique_id)
        else:
            self.agentsDuplicate.append(k)

    def _reset_continuous_uniques(self):
        self.agentsUnique, self.agentsUniqueIDs, self.agentsDuplicate = [], [], []
        for i in range(self.noveltyItemCount + 1):
            self.agentsUnique.append(i)
            self.noveltyArchive[f'agent{i+1}']['actionsUniqueID'] = i

    def _print_novelty_stats(self):
        unique, duplicate = len(self.agentsUnique), len(self.agentsDuplicate)
        n_neighbors = self.hyParams['NNOVELTYNEIGHBORS']
        msg = 'Novelty search'
        if self.noveltyItemCount > n_neighbors:
            msg += ' (neighbor level reached):'
        print(f'{msg} {unique} unique agents, {duplicate} duplicate agents')

    def _build_novelty_args(self):
        args, update_flags = [], []
        self.neighborLimitReached = len(self.agentsUnique) > self.hyParams['NNOVELTYNEIGHBORS']
        for i, _ in enumerate(self.agentsUnique):
            if self.neighborLimitReached:
                bounds = self.calculateNoveltyNeighborBounds(i)
                needs_update = bounds[-1]
                range_lower, range_higher, idx = bounds[:3]
            else:
                needs_update = True
                range_lower, range_higher, idx = 0, len(self.agentsUnique), i
            if needs_update:
                args.append([i, None, range_lower, range_higher, idx])
            update_flags.append(needs_update)
        return args, update_flags

    def _run_novelty_search(self, args):
        novelties = []
        for chunk in self.yieldChunks(args, self.cores * self.maxTasksPerWorkerNoveltySearch):
            novelties += self.multiprocessChunks(self.calculateNoveltyPerAgent, chunk)
        return novelties

    def _update_agent_novelties(self, novelties, update_flags):
        count = 0
        for i, agent_idx in enumerate(self.agentsUnique):
            agent_str = f'agent{agent_idx+1}'
            if update_flags[i]:
                self.noveltyArchive[agent_str]['novelty'] = novelties[count]
                count += 1
        for agent_idx in self.agentsDuplicate:
            agent_str = f'agent{agent_idx+1}'
            unique_id = self.noveltyArchive[agent_str]['actionsUniqueID']
            self.noveltyArchive[agent_str]['novelty'] = novelties[unique_id]

    def _save_novelties(self):
        self.pickleDump(join(self.tempModelPrefix + '_noveltyArchive.p'), self.noveltyArchive)
        self.novelties, self.noveltyFilenames = [], []
        for k in range(self.noveltyItemCount):
            agent_str = f'agent{k+1}'
            self.novelties.append(self.noveltyArchive[agent_str]['novelty'])
            self.noveltyFilenames.append(self.noveltyArchive[agent_str]['modelFile'])

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
                    dimShape = len(shape(actions_))
                    for iDim in range(dimShape):
                        if shape(actions_)[iDim] > maxShape[iDim]:
                            maxShape[iDim] = shape(actions_)[iDim]

        if self.hyParams['NGAMESAVERAGED'] != 1:
            maxShape = [self.hyParams['NGAMESAVERAGED']] + maxShape
        maxShape = tuple(maxShape)

        # replace this with specific agents?
        for iAgent in agents:
        # for iAgent in range(len(agents)):
            agentStr = 'agent' + str(iAgent+1)
            pth = self.noveltyArchive[agentStr]['resultsFile']
            actions = self.pickleLoad(pth)['actions']
            if self.actionType == 'discrete':
                # sharedListActions_ = chararray(shape=maxShape, itemsize=10)
                sharedListActions_ = empty(maxShape, dtype='S10')  # Byte strings, max length 10
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
            sharedListActions = reshape(sharedListActions, tuple([len(agents)] + list(maxShape)))
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
        nMaxUpdates = self.hyParams['NAGENTS'] + self.hyParams['NNOVELTYNEIGHBORS']
        if (nAgentsUnique - (iAgent+1)) >= nMaxUpdates:
            needsUpdate = False
        else:
            needsUpdate = True

        return rangeLower, rangeHigher, iAgentInCroppedArray, needsUpdate

    def createNNModel(self, actionType, seed=None):
        """Create neural network."""
        if seed is None:
            seed = self.SEED
        seed_orig = deepcopy(seed)
        model = Sequential()
        seed = int(seed)  # force Python int
        seed = SeedGenerator(seed=seed)
        if actionType == 'discrete':
            initializer = random_uniform(minval=-2.2, maxval=2.2, seed=seed)
        elif actionType == 'continuous':
            initializer = random_uniform(minval=-2.5, maxval=2.5, seed=seed)
        else:
            print('Chosen normal glorot initializer, as not specified.')
            initializer = glorot_normal(seed=seed)

        nHiddenNodes = copy(self.hyParams['NHIDDENNODES'])
        # resetting numpy seeds to generate reproducible architecture
        numpySeed(seed_orig)

        if self.hyParams['NNTYPE'] == 'perceptron':
            # fully-connected feed-forward multi-layer neural network
            # applying architecture (variable number of nodes per hidden layer)
            if self.agentMode == 'genetic' and self.hyParams['ARCHITECTUREVARY']:
                for layerIdx in range(len(nHiddenNodes)):
                    nHiddenNodes[layerIdx] = randint(2, self.hyParams['NHIDDENNODES'][layerIdx]+1)
            for layerIdx in range(len(nHiddenNodes)):
                inputShape = shape(self.observationsVector) if layerIdx == 0 else []
                model.add(Dense(units=int(nHiddenNodes[layerIdx]),
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
        self.gameReward, self.gameStep, done = 0, 1, False
        for game in range(self.hyParams['NAGENTSTEPS']):
            rnd = random()
            # epsilon defines the fraction of random to queried actions
            if rnd > self.epsilon:
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

            new_state, reward, done, truncated, info = env.step(action)
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

            # print('DEBUG env._seed', env._seed)

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
            env.reset(seed=seedCV, initWithSolution=env.initWithSolution)

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
                new_state, reward, done, truncated, info = env.step(action)
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

        tempAgentPrefix = join(self.tempModelPrefix + '_agent'
            + str(agentCount + 1).zfill(self.zFill))

        # loading specific agent and weights with given ID
        t0load_model = time()

        if self.envSettings['KEEPMODELHISTORY']:
            agent = load_model(join(tempAgentPrefix + '.keras'), compile=False)
        
        else:
            t0RecreateFromMutationHistory = time()
            agentOffset = self.hyParams['NAGENTS'] * (self.geneticGeneration)
            agentNumber = agentCount + 1 + agentOffset
            creationSeed = self.mutationHistory['agent' + str(agentNumber)]['creationSeed']
            mutationSeeds = self.mutationHistory['agent' + str(agentNumber)]['mutationSeeds']
            agent = self.loadModelFromMutationHistory(creationSeed, mutationSeeds)
            tRecreateFromMutationHistory = (self.hyParams['NAGENTS']/self.envSettings['NAGENTSPARALLEL']) * (t0RecreateFromMutationHistory - time())
            if agentCount == 1:
                print('model recreation took about', tRecreateFromMutationHistory, 's')

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
            # resetting to unique temporary folder to enable parallelism
            # Note: This will resimulate the initial environment state
            env.reset(MODELNAME=MODELNAMETEMP, seed=SEEDTEMP)
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

        r = 0
        for step in range(self.hyParams['NAGENTSTEPS']):
            if env.actionType == 'discrete':
                actionIdx = argmax(self.getqsGivenAgentModel(agent,
                    env.observationsVectorNormalized))
                action = self.actionSpace[actionIdx]
            if env.actionType == 'continuous':
                action = self.getqsGivenAgentModel(
                    agent, env.observationsVectorNormalized)
            
            # note: need to feed normalized observations
            new_observation, reward, done, truncated, info = env.step(action)
            actions[self.currentGame-1].append(action)
            rewards[self.currentGame-1].append(reward)
            if not env.ENVTYPE in ['0s-c']:
                wellCoords[self.currentGame-1].append(env.wellCoords)
            r += reward
            if self.envSettings['RENDER']:
                env.render()

            if done or (step == self.hyParams['NAGENTSTEPS']-1): # or if reached end
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
            str(agentIdx + 1).zfill(self.zFill) + '.keras'))

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
                str(sortedParentIdxs[0] + 1).zfill(self.zFill) + '.keras'),
                compile=False)
        else:
            agentOffsetBest = self.hyParams['NAGENTS'] * (self.geneticGeneration)
            agentNumberBest = sortedParentIdxs[0] + 1 + agentOffsetBest
            creationSeedBest = self.mutationHistory['agent' + str(agentNumberBest)]['creationSeed']
            mutationSeedsBest = self.mutationHistory['agent' + str(agentNumberBest)]['mutationSeeds']
            bestAgent = self.loadModelFromMutationHistory(creationSeedBest, mutationSeedsBest)

        if not self.rereturnChildrenGenetic:
            bestAgent.save(join(self.tempModelPrefix + '_agentBest.keras'))
        if generation < self.hyParams['NGENERATIONS']:
            if self.envSettings['KEEPMODELHISTORY']:
                bestAgent.save(join(tempNextModelPrefix + '_agent' +
                    str(self.hyParams['NAGENTS']).zfill(self.zFill) + '.keras'))
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
            str(selected_agent_index + 1).zfill(self.zFill) + '.keras')

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
        # altering parent agent to create child agent
        childrenAgent = self.mutateGenetic(agent, childIdx)

        if self.envSettings['KEEPMODELHISTORY']:
            childrenAgent.save(join(self.tempNextModelPrefix +
                '_agent' + str(childIdx + 1).zfill(self.zFill) + '.keras'))
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
        """ Query given model for Q values given observations of state.
        predict_on_batch robust in parallel operation?
        """

        state = array(state).reshape(-1, *shape(state))
        # state = array(state).reshape(-1, (*shape(state)))
        prediction = agentModel.predict_on_batch(state)[0]
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
            agentModel.load_weights(modelPrefix + '.weights.h5')
        if not exists(modelPrefix + '.json'):
            try: agentModel = load_model(modelPrefix + '.model', compile=compiled)
            except: agentModel = load_model(modelPrefix + '.keras', compile=compiled)
            json_config = agentModel.to_json()
            with open(modelPrefix + '.json', 'w') as json_file:
                json_file.write(json_config)
            agentModel.save_weights(modelPrefix + '.weights.h5')

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
        bestAgentpth = join(self.tempModelPrefix + '_agentBest.keras')
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
            diffsCount = len(shorterObj) - sum(1 for x, y in zip(longerObj, shorterObj) if x == y)

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

        return novelty

    def calculateNoveltyPerPair(self, args):

        actions = args[0]
        actions2 = args[1]

        # loading per agent becomes a real bottleneck if done repeatedly
        # better to know which needed beforehand
        # tempAgentPrefix = self.noveltyArchive[agentStr]['modelFile'].replace('.keras', '')
        # tempAgentPrefix2 = self.noveltyArchive[agentStr2]['modelFile'].replace('.keras', '')
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

        agentStr = 'agent' + str(iAgentOriginal+1)
        iAgent = iAgentInCroppedArray

        # loading noveties for specific agent from disk
        noveltyFile = join(self.tempNoveltypth, agentStr + '_novelties.p')
        if exists(noveltyFile):
            agentNovelties = self.pickleLoad(noveltyFile, compressed=None)
        else:
            agentNovelties = {}

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

        novelties = []
        if not self.neighborLimitReached:
            for iAgent2 in range(len(arr)):
                if iAgent != iAgent2:
                    agentStr2 = 'agent' + str(iAgent2+1)
                    actions2 = arr[iAgent2]
                    try:
                        # calculate novelties only if unavailable
                        # as keys mostly exists, try/except check should be performant here
                        novelty = agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+rangeLower+1)]
                    except:
                        novelty = self.calculateNoveltyPerPair([actions, actions2])
                        agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+rangeLower+1)] = novelty
                    novelties.append(novelty)

        elif self.neighborLimitReached:
            iArr = 0
            for iAgent2 in range(rangeLower, rangeHigher):
                if iAgentOriginal != iAgent2:
                    agentStr2 = 'agent' + str(iAgent2+1)
                    # actionsUniqueID2 = self.noveltyArchive[agentStr2]['actionsUniqueID']
                    actions2 = arr[iArr]
                    try:
                        # calculate novelties only if unavailable
                        # as keys mostly exists, try/except check should be performant here
                        novelty = agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+1)]
                    except:
                        novelty = self.calculateNoveltyPerPair([actions, actions2])
                        agentNovelties[str(iAgentOriginal+1) + '_' + str(iAgent2+1)] = novelty
                    novelties.append(novelty)
                iArr += 1

        dumpPath = join(self.tempNoveltypth, agentStr + '_novelties.p')
        self.pickleDump(dumpPath, agentNovelties, compress=None)

        novelty = mean(novelties)

        return novelty

    def saveBestAgent(self, MODELNAME):
        # saving best agent of the current generation
        bestAgent = load_model(join(self.tempModelPrefix + '_agentBest.keras'),
                compile=False)
        self.bestAgentFileName = (f'{MODELNAME}' + '_gen' +
            str(self.geneticGeneration+1).zfill(self.zFill) + '_avg' +
            f'{self.bestAgentReward:_>7.1f}')
        bestAgent.save(join(self.modelpth, self.bestAgentFileName + '.keras'))

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

    def multiprocessChunks(self, function, chunk, parallelProcesses=None, wait=False, async_=True, sharedArr=None, sharedDict=None):
        """Process function in parallel given a chunk of arguments."""

        # Pool object from pathos instead of multiprocessing library necessary
        # as tensor.keras models are currently not pickleable
        # https://github.com/tensorflow/tensorflow/issues/32159
        if parallelProcesses == None:
            parallelProcesses = self.envSettings['NAGENTSPARALLEL']
        # print('debug parallelProcesses', parallelProcesses)

        if sharedDict == None and sharedArr is None:
            p = Pool(processes=parallelProcesses)
            if async_:
                # imap to use map with a generator and reduce memory impact
                # chunksize = int(max([int(len(chunk)/parallelProcesses), 1]))
                # pasync = p.imap(function, chunk, chunksize=chunksize)
                # pasync = p.imap(function, chunk)
                pasync = p.map_async(function, chunk)
                pasync = pasync.get()
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
            with Pool(processes=parallelProcesses, initializer=self.init_arr, initargs=(arr, arr_lock, shape_)) as p:
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
                p = Pool(processes=parallelProcesses)
            else:
                p = ThreadPool(processes=parallelProcesses)
            if async_:
                pasync = p.map_async(function, chunk) # , chunksize=10)
                pasync = pasync.get()
            else:
                pasync = p.map(function, chunk)
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
            with Pool(processes=parallelProcesses, initializer=self.init_arr, initargs=(arr, arr_lock, shape_)) as p:
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

    # def GPUAllowMemoryGrowth(self):
    #     """Allow GPU memory to grow to enable parallelism on a GPU."""
    #     config = ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = TFSession(config=config)
    #     K.set_session(sess)

    def GPUAllowMemoryGrowth(self):
        """Allow GPU memory to grow to enable parallelism on a GPU."""
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Enabled memory growth for {len(gpus)} GPU(s).")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"Error setting memory growth: {e}")

    def suppressTensorFlowWarnings(self):
        # suppressing TensorFlow output on import, except fatal errors
        # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
        from logging import getLogger, FATAL
        environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        getLogger('tensorflow').setLevel(FATAL)
