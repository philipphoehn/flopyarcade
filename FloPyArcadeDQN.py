#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyAgent
from FloPyArcade import FloPyEnv

# environment settings
envSettings = {
    'ENVTYPE': '3s-d',						        # string defining environment
    'MODELNAME': 'test',	                    # string defining model basename
    'PATHMF2005': None,					        # string of local path to MODFLOW 2005 executable
    'PATHMP6': None,					        # string of local path to MODPATH 6 executable
    'SURROGATESIMULATOR': None,
    'SEEDAGENT': 2,						        # integer enabling reproducibility
    'SEEDENV': 2,			                    # integer enabling reproducibility of environments during cross-validation
    'RENDER': False,				     	        # boolean to define if displaying runs
    'RENDEREVERY': 1000,				        # integer defining runs displayed
    'REWARDMINTOSAVE': 0  				        # float defining minimal reward to save a model
}

# hyperparameters
hyParams = {
    'NGAMES': 20000,					        # integer defining total number of games to play
    'NAGENTSTEPS': 200,					        # integer defining the number of steps taken per game
    'DISCOUNT': 0.99,					        # float defining the discount policy
    'REPLAYMEMORYSIZE': 1000000,  		        # integer defining the number of steps kept for training
    'REPLAYMEMORYSIZEMIN': 512,			        # integer defining the minimum number of steps required for training
    'MINIBATCHSIZE': 512,				        # integer defining the number of sampled steps used for training
    'UPDATEPREDICTIVEMODELEVERY': 5, 	        # integer defining after how many games the predictive model is updated
    'NNTYPE': 'convolution',                    # string defining neural network type, options: ['perceptron', 'convolution']
    'NHIDDENNODES': [40] * 5,			        # list of integers of nodes per hidden layer to define architecture
    'HIDDENACTIVATIONS': ['relu'] * 5,	        # list of strings defining hidden nodal activations
    'DROPOUTS': [0.0] * 5,				        # list of floats defining hidden layer dropouts
    'NCONVFILTERS': [32, 64, 64],               # list of integers defining numbers of convolutional filters
    'CONVKERNELSIZES': [8, 4, 3],               # list of integers defining convolutional kernel sizes
    'CONVSTRIDES': [4, 2, 1],                   # list of integers defining convolutional stride sizes 
    'CONVACTIVATIONS': ['relu'] * 3,            # list of strings defining convolutional activations
    'CONVPOOLING': None,                        # string defining type of pooling, options: [None, 'mean', 'max']
    'CONVPOOLSIZES': [1, 1, 1],                 # list of integers defining sizes of convolutional pool
    'CONVPOOLSTRIDES': [1, 1, 1],               # list of integers defining strides of convolutional pool
    'BATCHNORMALIZATION': True,			        # boolean defining batch normalization
    'LEARNINGRATE': 0.000025,			        # float defining the learning rate
    'EPSILONINITIAL': 1.0,				        # float defining the exploration fraction
    'EPSILONDECAY': 0.99995,			        # float defining the factor to decay the exploration fraction
    'EPSILONMIN': 0.005,				        # float defining the lowest exploration fraction following decay
    'CROSSVALIDATEEVERY': 250,			        # integer defining the number of games between cross-validation
    'NGAMESCROSSVALIDATED': 200			        # integer defining the number of games during cross-validation
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
        env.observationsVector,
        env.actionSpace,
        hyParams,
        envSettings,
        'DQN')
    agent.GPUAllowMemoryGrowth()

    # running training of DQN agent
    agent.runDQN(env)


if __name__ == '__main__':
    main(envSettings, hyParams)
