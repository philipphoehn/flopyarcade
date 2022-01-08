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
parser.add_argument('--envtype', default='6s-c', type=str,
    help='string defining environment')
parser.add_argument('--seedenv', default=1, type=int,
    help='integer enabling reproducibility of the environments')
parser.add_argument('--seedagent', default=1, type=int,
    help='integer enabling reproducibility of the agents')
parser.add_argument('--nlay', default=1, type=int,
    help='integer defining numbers of model layers')
parser.add_argument('--nrow', default=100, type=int,
    help='integer defining grid rows')
parser.add_argument('--ncol', default=100, type=int,
    help='integer defining grid columns')
parser.add_argument('--render', default=False, type=str_to_bool,
    help='boolean to define if displaying runs')
parser.add_argument('--renderevery', default=100, type=int,
    help='integer defining runs displayed')
parser.add_argument('--rewardmintosave', default=0., type=float,
    help='float defining minimal reward to save a model')
parser.add_argument('--modelname', default='FloPyArcadeDQN', type=str,
    help='string defining model basename')
parser.add_argument('--pathmf2005', default=None,
    help='string of local path to MODFLOW 2005 executable')
parser.add_argument('--pathmp6', default=None,
    help='string of local path to MODPATH 6 executable')
parser.add_argument('--surrogatesimulator', default=None,
    help='currently unavailable')
parser.add_argument('--ngames', default=20000, type=int,
    help='integer defining total number of games to play')
parser.add_argument('--nagentsteps', default=200, type=int,
    help='integer defining the number of steps taken per game')
parser.add_argument('--discount', default=0.99, type=float,
    help='float defining the discount policy')
parser.add_argument('--replaymemorysize', default=100000, type=int,
    help='integer defining the number of steps kept for training')
parser.add_argument('--replaymemorysizemin', default=512, type=int,
    help='integer defining the minimum number of steps required for training')
parser.add_argument('--minibatchsize', default=512, type=int,
    help='integer defining the number of sampled steps used for training')
parser.add_argument('--updatepredictivemodelevery', default=5, type=int,
    help='integer defining after how many games the predictive model is updated')
parser.add_argument('--nntype', default='perceptron', type=str,
    help='string defining neural network type, options: ["perceptron", "convolution"]')
parser.add_argument('--nhiddennodes', default=[50, 50, 50], nargs='+', type=int,
    help='list of integers of nodes per hidden layer to define architecture')
parser.add_argument('--hiddenactivations', default=['relu', 'relu', 'relu'],
    help='list of strings defining hidden nodal activations')
parser.add_argument('--dropouts', default=[0.0, 0.0, 0.0], nargs='+', type=float,
    help='list of floats defining hidden layer dropouts')
parser.add_argument('--nconvfilters', default=[32, 64, 64], nargs='+', type=int,
    help='list of integers defining numbers of convolutional filters')
parser.add_argument('--convkernelsizes', default=[8, 4, 3], nargs='+', type=int,
    help='list of integers defining convolutional kernel sizes')
parser.add_argument('--convstrides', default=[4, 2, 1], nargs='+', type=int,
    help='list of integers defining convolutional stride sizes')
parser.add_argument('--convactivations', default=['relu', 'relu', 'relu'],
    help='list of strings defining convolutional activations')
parser.add_argument('--convpooling', default=None,
    help='string defining type of pooling, options: [None, "mean", "max"]')
parser.add_argument('--convpoolsizes', default=[1, 1, 1], nargs='+', type=int,
    help='list of integers defining sizes of convolutional pool')
parser.add_argument('--convpoolstrides', default=[1, 1, 1], nargs='+', type=int,
    help='list of integers defining strides of convolutional pool')
parser.add_argument('--batchnormalization', default=True, type=str_to_bool,
    help='boolean defining batch normalization')
parser.add_argument('--learningrate', default=0.0001, type=float,
    help='float defining the learning rate')
parser.add_argument('--epsiloninitial', default=1.0, type=float,
    help='float defining the exploration fraction')
parser.add_argument('--epsilondecay', default=0.99995, type=float,
    help='float defining the factor to decay the exploration fraction')
parser.add_argument('--epsilonmin', default=0.005, type=float,
    help='float defining the lowest exploration fraction following decay')
parser.add_argument('--crossvalidateevery', default=50, type=int,
    help='integer defining the number of games between cross-validation')
parser.add_argument('--ngamescrossvalidated', default=10, type=int,
    help='integer defining the number of games during cross-validation')
args = parser.parse_args()


# environment settings
envSettings = {
    'ENVTYPE': args.envtype,				            # string defining environment
    'SEEDENV': args.seedenv,                            # integer enabling reproducibility of the environments
    'SEEDAGENT': args.seedagent,                        # integer enabling reproducibility of the agents
    'NLAY': args.nlay,                                  # integer defining numbers of model layers
    'NROW': args.nrow,                                  # integer defining grid rows
    'NCOL': args.ncol,                                  # integer defining grid columns
    'RENDER': args.render,                              # boolean to define if displaying runs
    'RENDEREVERY': args.renderevery,                    # integer defining runs displayed
    'REWARDMINTOSAVE': args.rewardmintosave,            # float defining minimal reward to save a model
    'MODELNAME': args.modelname,                        # string defining model basename
    'PATHMF2005': args.pathmf2005,                      # string of local path to MODFLOW 2005 executable
    'PATHMP6': args.pathmp6,                            # string of local path to MODPATH 6 executable
    'SURROGATESIMULATOR': args.surrogatesimulator,      # currently unavailable
}

# hyperparameters
hyParams = {
    'NGAMES': args.ngames,					            # integer defining total number of games to play
    'NAGENTSTEPS': args.nagentsteps,					# integer defining the number of steps taken per game
    'DISCOUNT': args.discount,                          # float defining the discount policy
    'REPLAYMEMORYSIZE': args.replaymemorysize,          # integer defining the number of steps kept for training
    'REPLAYMEMORYSIZEMIN': args.replaymemorysizemin,    # integer defining the minimum number of steps required for training
    'MINIBATCHSIZE': args.minibatchsize,                # integer defining the number of sampled steps used for training
    'UPDATEPREDICTIVEMODELEVERY': args.updatepredictivemodelevery,  # integer defining after how many games the predictive model is updated
    'NNTYPE': args.nntype,                              # string defining neural network type, options: ['perceptron', 'convolution']
    'NHIDDENNODES': args.nhiddennodes,                  # list of integers of nodes per hidden layer to define architecture
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
    'LEARNINGRATE': args.learningrate,                  # float defining the learning rate
    'EPSILONINITIAL': args.epsiloninitial,              # float defining the exploration fraction
    'EPSILONDECAY': args.epsilondecay,                  # float defining the factor to decay the exploration fraction
    'EPSILONMIN': args.epsilonmin,				        # float defining the lowest exploration fraction following decay
    'CROSSVALIDATEEVERY': args.crossvalidateevery,      # integer defining the number of games between cross-validation
    'NGAMESCROSSVALIDATED': args.ngamescrossvalidated   # integer defining the number of games during cross-validation
}


def main(envSettings, hyParams):
    # initializing environment
    env = FloPyEnv(
        envSettings['ENVTYPE'],
        envSettings['PATHMF2005'],
        envSettings['PATHMP6'],
        envSettings['MODELNAME'],
        flagRender=envSettings['RENDER'],
        NAGENTSTEPS=hyParams['NAGENTSTEPS'],
        OBSPREP=hyParams['NNTYPE'])

    # initializing agent
    agent = FloPyAgent(
        env.observationsVectorNormalized,
        env.actionSpace,
        hyParams,
        envSettings,
        'DQN')
    agent.GPUAllowMemoryGrowth()

    # running training of DQN agent
    agent.runDQN(env)


if __name__ == '__main__':
    main(envSettings, hyParams)
