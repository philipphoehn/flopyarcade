#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyAgent
from FloPyArcade import FloPyEnv

# environment settings
envSettings = {
    'ENVTYPE': '3s-d',                          # string defining environment
    'MODELNAME': 'ge_3s-d_s1n1000e100g500av1st200mpr1e-0mpo3e-3_conv_res100_ns-ev1n500e50nn1e4',                       # string defining model basename
    # 'MODELNAME': 'ge_3s-d_s1n1000e100g500av1st200mpr1e-0mpo3e-3_percar1000x4v0relubn1_res100_ns-ev1n500e50nn1e4',    # string defining model basename
    'PATHMF2005': None,                         # string defining local path to MODFLOW 2005 executable
    'PATHMP6': None,                            # string defining local path to MODPATH 6 executable
    'SURROGATESIMULATOR': None,                 # currently unavailable
    'SEEDAGENT': 1,                             # integer enabling reproducibility of the agents
    'SEEDENV': 1,                               # integer enabling reproducibility of the environments
    'NAGENTSPARALLEL': 16,                      # integer defining parallelized agent runs
    'REWARDMINTOSAVE': 0.0,                     # float defining minimal reward to save a model
    'RENDER': False,                            # boolean defining if displaying runs
    'RENDEREVERY': 1000,                        # integer defining runs displayed
    'BESTAGENTANIMATION': True,                 # boolean defining whether to save animation of best agent per generation
    'KEEPMODELHISTORY': True,                   # boolean defining whether to keep all agentut evolution
    'RESUME': False,                            # boolean defining whether to keep all agents throughs throughoout evolution
    'NLAY': 1,                                  # integer defining numbers of model layers
    'NROW': 100,                                # integer defining grid rows
    'NCOL': 100                                 # integer defining grid columns
}

# hyperparameters
hyParams = {
    'NAGENTS': 1000,                            # integer defining number of agents
    'NAGENTELITES': 100,                        # integer defining number of agents considered as parents
    'NGENERATIONS': 500,                        # integer defining number of generations for evolution
    'NGAMESAVERAGED': 1,                        # integer defining number of games played for averaging
    'NAGENTSTEPS': 200,                         # integer defining number of episodes per agent
    'MUTATIONPROBABILITY': 1.0,                 # float defining fraction of mutated parameters
    'MUTATIONPOWER': 0.003,                     # float defining mutation, 0.02 after https://arxiv.org/pdf/1712.06567.pdf
    'NNTYPE': 'convolution',                    # string defining neural network type ['perceptron', 'convolution']
    'NHIDDENNODES': [1000] * 4,                 # list of integers defining nodes per hidden layer to define architecture
    'ARCHITECTUREVARY': False,                  # boolean defining to allow architecture variation
    'HIDDENACTIVATIONS': ['relu'] * 4,          # list of strings defining hidden nodal activations
    'BATCHNORMALIZATION': False,                # boolean defining batch normalization
    'NOVELTYSEARCH': True,                      # boolean defining novelty search
    'ADDNOVELTYEVERY': 1,                       # integer defining generations between novelty update
    'NNOVELTYAGENTS': 500,                      # integer defining number of novelty agents
    'NNOVELTYELITES': 50,                       # integer defining number of novelty elites
    'NNOVELTYNEIGHBORS': 10000                  # integer defining number of novelty neighbors considered
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
        env.observationsVector,
        env.actionSpace,
        hyParams,
        envSettings,
        'genetic')

    # running genetic search algorithm
    agent.runGenetic(env, hyParams['NOVELTYSEARCH'])


if __name__ == '__main__':
    main(envSettings, hyParams)
