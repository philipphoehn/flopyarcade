#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyAgent
from FloPyArcade import FloPyEnv
from FloPyArcade import FloPyEnvSurrogate

# environment settings
envSettings = {
    'ENVTYPE': '3',                             # string defining environment
    'MODELNAME': 'ConvTest', # string defining model basename
    'PATHMF2005': None,                         # string of local path to MODFLOW 2005 executable
    'PATHMP6': None,                            # string of local path to MODPATH 6 executable
    'SURROGATESIMULATOR': None,
    # 'SURROGATESIMULATOR': ['bestModelUnweightedInitial', 'bestModelUnweighted'],   # current best fit 0.000619
    'SEEDAGENT': 1,                                  # integer enabling reproducibility of the agents
    'SEEDENV': 1,                               # integer enabling reproducibility of the environments
    'NAGENTSPARALLEL': 4,                      # integer defining parallelized agent runs
    'REWARDMINTOSAVE': 0.0,                     # float defining minimal reward to save a model
    'RENDER': False,                            # boolean to define if displaying runs
    'RENDEREVERY': 1000,                        # integer defining runs displayed
    'BESTAGENTANIMATION': True,                 # boolean defining whether to save animation of best agent per generation
    'KEEPMODELHISTORY': True,                   # boolean defining whether to keep all agents throughout evolution
    'RESUME': False,                              # boolean defining whether to keep all agents throughout evolution
    'NLAY': 1,
    'NROW': 100,
    'NCOL': 100
}

# hyperparameters
hyParams = {
    'NAGENTS': 32,                              # integer defining number of agents
    'NAGENTELITES': 25,                         # integer defining number of agents considered as parents
    'NGENERATIONS': 10000,                          # integer defining number of generations for evolution
    'NGAMESAVERAGED': 1,                       # integer defining number of games played for averaging
    'NAGENTSTEPS': 200,                         # integer defining number of episodes per agent
    'MUTATIONPROBABILITY': 1.0,                # float defining fraction of mutated parameters
    'MUTATIONPOWER': 0.005,                     # float defining mutation, 0.02 after https://arxiv.org/pdf/1712.06567.pdf
    'MODELTYPE': 'mlp',
    'NHIDDENNODES': [250] * 5,                   # list of integers of nodes per hidden layer to define architecture
    'ARCHITECTUREVARY': True,                   # boolean defining to allow architecture variation
    'HIDDENACTIVATIONS': ['relu'] * 5,          # list of strings defining hidden nodal activations
    'BATCHNORMALIZATION': True,                  # boolean defining batch normalization
    'NOVELTYSEARCH': True,                       # boolean
    'ADDNOVELTYEVERY': 1,
    'NNOVELTYELITES': 100,
    'NNOVELTYNEIGHBORS': 100
}


def main(envSettings, hyParams):
    # initializing environment

    if envSettings['SURROGATESIMULATOR'] is None:
        env = FloPyEnv(
            envSettings['ENVTYPE'],
            envSettings['PATHMF2005'],
            envSettings['PATHMP6'],
            MODELNAME=envSettings['MODELNAME'],
            flagRender=envSettings['RENDER'],
            NAGENTSTEPS=hyParams['NAGENTSTEPS'],
            nLay=envSettings['NLAY'],
            nRow=envSettings['NROW'],
            nCol=envSettings['NCOL'])

    elif envSettings['SURROGATESIMULATOR'] is not None:
        env = FloPyEnvSurrogate(
            SURROGATESIMULATOR=envSettings['SURROGATESIMULATOR'],
            ENVTYPE=envSettings['ENVTYPE'],
            MODELNAME=envSettings['MODELNAME'],
            NAGENTSTEPS=hyParams['NAGENTSTEPS'])

    # initializing agent
    agent = FloPyAgent(
        env.observationsVectorNormalizedHeads if hyParams['MODELTYPE']=='conv' else env.observationsVector,
        env.actionSpace,
        hyParams,
        envSettings,
        'genetic')

    # running genetic search algorithm
    agent.runGenetic(env, hyParams['NOVELTYSEARCH'])


if __name__ == '__main__':
    main(envSettings, hyParams)