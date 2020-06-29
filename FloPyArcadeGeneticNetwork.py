#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyAgent
from FloPyArcade import FloPyEnv

# environment settings
envSettings = {
    'ENVTYPE': '3',                             # string defining environment
    'MODELNAME': 'test',                        # string defining model basename
    'PATHMF2005': None,                         # string of local path to MODFLOW 2005 executable
    'PATHMP6': None,                            # string of local path to MODPATH 6 executable
    'SURROGATESIMULATOR': None,
    'SEEDAGENT': 1,                             # integer enabling reproducibility of the agents
    'SEEDENV': 3,                               # integer enabling reproducibility of the environments
    'NAGENTSPARALLEL': 16,                      # integer defining parallelized agent runs
    'REWARDMINTOSAVE': 0.0,                     # float defining minimal reward to save a model
    'RENDER': False,                            # boolean to define if displaying runs
    'RENDEREVERY': 1000,                        # integer defining runs displayed
    'BESTAGENTANIMATION': True,                 # boolean defining whether to save animation of best agent per generation
    'KEEPMODELHISTORY': True,                   # boolean defining whether to keep all agents throughout evolution
    'RESUME': True,                             # boolean defining whether to keep all agents throughout evolution
    'NLAY': 1,                                  # integer defining numbers of model layers
    'NROW': 100,                                # integer defining grid rows
    'NCOL': 100                                 # integer defining grid columns
}

# hyperparameters
hyParams = {
    'NAGENTS': 2000,                            # integer defining number of agents
    'NAGENTELITES': 100,                        # integer defining number of agents considered as parents
    'NGENERATIONS': 10000,                      # integer defining number of generations for evolution
    'NGAMESAVERAGED': 50,                       # integer defining number of games played for averaging
    'NAGENTSTEPS': 200,                         # integer defining number of episodes per agent
    'MUTATIONPROBABILITY': 1.0,                 # float defining fraction of mutated parameters
    'MUTATIONPOWER': 0.003,                     # float defining mutation, 0.02 after https://arxiv.org/pdf/1712.06567.pdf
    'NHIDDENNODES': [250] * 5,                  # list of integers of nodes per hidden layer to define architecture
    'ARCHITECTUREVARY': True,                   # boolean defining to allow architecture variation
    'HIDDENACTIVATIONS': ['relu'] * 5,          # list of strings defining hidden nodal activations
    'BATCHNORMALIZATION': True,                 # boolean defining batch normalization
    'NOVELTYSEARCH': True,                      # boolean defining novelty search
    'ADDNOVELTYEVERY': 1,                       # generational interval to add novelty agents
    'NNOVELTYELITES': 1000,                     # number of novelty elites
    'NNOVELTYNEIGHBORS': 100                    # number of novelty neighbors considered
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
        nCol=envSettings['NCOL'])

    # initializing agent
    agent = FloPyAgent(
        env.observationsVector,
        env.actionSpace,
        hyParams,
        envSettings,
        'genetic')

    # running genetic search algorithm
    agent.runGenetic(env, hyParams['NOVELTYSEARCH'])


if __name__ == '__main__':
    main(envSettings, hyParams)