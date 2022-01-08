#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from argparse import ArgumentParser
from flopyarcade import FloPyAgent
from flopyarcade import FloPyEnv

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = ArgumentParser(description='FloPyArcade optimization using deep Q networks')
parser.add_argument('--envtype', default='3s-d', type=str,
    help='string defining environment')
parser.add_argument('--seedenv', default=1, type=int,
    help='integer enabling reproducibility of the environments')
parser.add_argument('--seedagent', default=1, type=int,
    help='integer enabling reproducibility of the agents')
parser.add_argument('--seedsrandom', default=True, type=str_to_bool,
    help='boolean defining whether training occurs on random seeds per generation')
parser.add_argument('--nlay', default=1, type=int,
    help='integer defining numbers of model layers')
parser.add_argument('--nrow', default=100, type=int,
    help='integer defining grid rows')
parser.add_argument('--ncol', default=100, type=int,
    help='integer defining grid columns')
parser.add_argument('--render', default=False, type=str_to_bool,
    help='boolean defining if displaying runs')
parser.add_argument('--renderevery', default=100, type=int,
    help='integer defining runs displayed')
parser.add_argument('--bestagentanimation', default=True, type=str_to_bool,
    help='boolean defining whether to save animation of best agent per generation')
parser.add_argument('--modelname', default='FloPyArcadeDQN', type=str,
    help='string defining model basename')
parser.add_argument('--pathmf2005', default=None,
    help='string defining local path to MODFLOW 2005 executable')
parser.add_argument('--pathmp6', default=None,
    help='string defining local path to MODPATH 6 executable')
parser.add_argument('--surrogatesimulator', default=None,
    help='currently unavailable')
parser.add_argument('--nagentsparallel', default=None,
    help='integer defining parallelized agent runs')
parser.add_argument('--rewardmintosave', default=0.0, type=float,
    help='float defining minimal reward to save a model')
parser.add_argument('--keepmodelhistory', default=True, type=str_to_bool,
    help='boolean defining whether to keep all agent evolution')
parser.add_argument('--resume', default=False, type=str_to_bool,
    help='boolean defining whether to keep all agents throughout evolution')
parser.add_argument('--nagents', default=300, type=int,
    help='integer defining number of agents')
parser.add_argument('--nagentelites', default=100, type=int,
    help='integer defining number of agents considered as parents')
parser.add_argument('--ngenerations', default=5000, type=int,
    help='integer defining number of generations for evolution')
parser.add_argument('--ngamesaveraged', default=3, type=int,
    help='integer defining number of games played for averaging')
parser.add_argument('--nagentsteps', default=200, type=int,
    help='integer defining the number of steps taken per game')
parser.add_argument('--mutationprobability', default=1.0, type=float,
    help='float defining fraction of mutated parameters')
parser.add_argument('--mutationpower', default=0.02, type=float,
    help='float defining mutation, 0.02 after https://arxiv.org/pdf/1712.06567.pdf')
parser.add_argument('--nntype', default='perceptron', type=str,
    help='string defining neural network type ["perceptron", "convolution"]')
parser.add_argument('--nhiddennodes', default=[50, 50, 50], nargs='+', type=int,
    help='list of integers defining nodes per hidden layer to define architecture')
parser.add_argument('--architecturevary', default=False, type=str_to_bool,
    help='boolean defining to allow architecture variation')
parser.add_argument('--hiddenactivations', default=['linear', 'linear', 'linear'],
    help='list of strings defining hidden nodal activations')
parser.add_argument('--dropouts', default=[0.0, 0.0, 0.0], nargs='+', type=float,
    help='list of floats defining hidden layer dropouts')
parser.add_argument('--nconvfilters', default=[32, 64, 64], nargs='+', type=int,
    help='list of integers defining numbers of convolutional filters')
parser.add_argument('--convkernelsizes', default=[8, 4, 3], nargs='+', type=int,
    help='list of integers defining convolutional kernel sizes')
parser.add_argument('--convstrides', default=[4, 2, 1], nargs='+', type=int,
    help='list of integers defining convolutional stride sizes ')
parser.add_argument('--convactivations', default=['linear', 'linear', 'linear'], nargs='+', type=str,
    help='list of strings defining convolutional activations')
parser.add_argument('--convpooling', default=None,
    help='string defining type of pooling, options: [None, "mean", "max"]')
parser.add_argument('--convpoolsizes', default=[1, 1, 1], nargs='+', type=int,
    help='list of integers defining sizes of convolutional pool')
parser.add_argument('--convpoolstrides', default=[1, 1, 1], nargs='+', type=int,
    help='list of integers defining strides of convolutional pool')
parser.add_argument('--batchnormalization', default=False, type=str_to_bool,
    help='boolean defining batch normalization')
parser.add_argument('--noveltysearch', default=True, type=str_to_bool,
    help='boolean defining novelty search')
parser.add_argument('--addnoveltyevery', default=1, type=int,
    help='integer defining generations between novelty update')
parser.add_argument('--nnoveltyagents', default=1000, type=int,
    help='integer defining number of novelty agents')
parser.add_argument('--nnoveltyelites', default=100, type=int,
    help='integer defining number of novelty elites')
parser.add_argument('--nnoveltyneighbors', default=200, type=int,
    help='integer defining number of novelty neighbors considered')
args = parser.parse_args()


# environment settings
envSettings = {
    'ENVTYPE': args.envtype,                            # string defining environment
    'SEEDENV': args.seedenv,                            # integer enabling reproducibility of the environments
    'SEEDAGENT': args.seedagent,                        # integer enabling reproducibility of the agents
    'SEEDSRANDOM': args.seedsrandom,                    # boolean defining whether training occurs on random seeds per generation
    'NLAY': args.nlay,                                  # integer defining numbers of model layers
    'NROW': args.nrow,                                  # integer defining grid rows
    'NCOL': args.ncol,                                  # integer defining grid columns
    'RENDER': args.render,                              # boolean defining if displaying runs
    'RENDEREVERY': args.renderevery,                    # integer defining runs displayed
    'BESTAGENTANIMATION': args.bestagentanimation,      # boolean defining whether to save animation of best agent per generation
    'MODELNAME': args.modelname,                        # string defining model basename
    # 'MODELNAME': 'ge_5s-c-cost_s1n2e3e1e2g500av3st200mpr1e-0mpo2e-3_convar200x4v0relubn1_res100_ns-ev1n1e3e1e2nn2e2_randomSeed',    # string defining model basename
    'PATHMF2005': args.pathmf2005,                      # string defining local path to MODFLOW 2005 executable
    'PATHMP6': args.pathmp6,                            # string defining local path to MODPATH 6 executable
    'SURROGATESIMULATOR': args.surrogatesimulator,      # currently unavailable
    'NAGENTSPARALLEL': args.nagentsparallel,            # integer defining parallelized agent runs
    'REWARDMINTOSAVE': args.rewardmintosave,            # float defining minimal reward to save a model
    'KEEPMODELHISTORY': args.keepmodelhistory,          # boolean defining whether to keep all agent evolution
    'RESUME': args.resume,                              # boolean defining whether to keep all agents throughout evolution
}

# hyperparameters
hyParams = {
    'NAGENTS': args.nagents,                            # integer defining number of agents
    'NAGENTELITES': args.nagentelites,                  # integer defining number of agents considered as parents
    'NGENERATIONS': args.ngenerations,                  # integer defining number of generations for evolution
    'NGAMESAVERAGED': args.ngamesaveraged,              # integer defining number of games played for averaging
    'NAGENTSTEPS': args.nagentsteps,                    # integer defining the number of steps taken per game
    'MUTATIONPROBABILITY': args.mutationprobability,    # float defining fraction of mutated parameters
    'MUTATIONPOWER': args.mutationpower,                # float defining mutation, 0.02 after https://arxiv.org/pdf/1712.06567.pdf
    'NNTYPE': args.nntype,                              # string defining neural network type ['perceptron', 'convolution']
    'NHIDDENNODES': args.nhiddennodes,                  # list of integers defining nodes per hidden layer to define architecture
    'ARCHITECTUREVARY': args.architecturevary,          # boolean defining to allow architecture variation
    'HIDDENACTIVATIONS': args.hiddenactivations,        # list of strings defining hidden nodal activations
    'DROPOUTS': args.dropouts,                          # list of floats defining hidden layer dropouts
    'NCONVFILTERS': args.nconvfilters,                  # list of integers defining numbers of convolutional filters
    'CONVKERNELSIZES': args.convkernelsizes,            # list of integers defining convolutional kernel sizes
    'CONVSTRIDES': args.convstrides,                    # list of integers defining convolutional stride sizes 
    'CONVACTIVATIONS': args.convactivations,            # list of strings defining convolutional activations
    'CONVPOOLING': args.convpooling,                    # string defining type of pooling, options: [None, 'mean', 'max']
    'CONVPOOLSIZES': args.convpoolsizes,                # list of integers defining sizes of convolutional pool
    'CONVPOOLSTRIDES': args.convpoolstrides,            # list of integers defining strides of convolutional pool
    'BATCHNORMALIZATION': args.batchnormalization,      # boolean defining batch normalization
    'NOVELTYSEARCH': args.noveltysearch,                # boolean defining novelty search
    'ADDNOVELTYEVERY': args.addnoveltyevery,            # integer defining generations between novelty update
    'NNOVELTYAGENTS': args.nnoveltyagents,              # integer defining number of novelty agents
    'NNOVELTYELITES': args.nnoveltyelites,              # integer defining number of novelty elites
    'NNOVELTYNEIGHBORS': args.nnoveltyneighbors         # integer defining number of novelty neighbors considered
}


def main(envSettings, hyParams):
    # initializing environment

    env = FloPyEnv(
        envSettings['ENVTYPE'],
        envSettings['PATHMF2005'],
        envSettings['PATHMP6'],
        MODELNAME=envSettings['MODELNAME'],
        flagRender=envSettings['RENDER'],
        NAGENTSTEPS=hyParams['NAGENTSTEPS'],
        nLay=envSettings['NLAY'],
        nRow=envSettings['NROW'],
        nCol=envSettings['NCOL'],
        OBSPREP=hyParams['NNTYPE'])

    # initializing agent
    agent = FloPyAgent(
        env.observationsVectorNormalized,
        env.actionSpace,
        hyParams,
        envSettings,
        'genetic')

    # running genetic search algorithm
    agent.runGenetic(env, hyParams['NOVELTYSEARCH'])


if __name__ == '__main__':

    main(envSettings, hyParams)

    # unfinished = True
    # while unfinished:
    #     try:
    #         main(envSettings, hyParams)
    #         unfinished = False
    #     except Exception as e:
    #         # enabling resume in case for random error
    #         print('resumed with error', e)
    #         envSettings['RESUME'] = True
