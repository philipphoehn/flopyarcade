#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyAgent
from FloPyArcade import FloPyEnv

# environment settings
envSettings = {
    'ENVTYPE': '3',								# string defining environment
    'MODELNAME': 'agentGenetic3av1mp5e-3n1024',	# string defining model basename
    'PATHMF2005': None,							# string of local path to MODFLOW 2005 executable
    'PATHMP6': None,							# string of local path to MODPATH 6 executable
    'SEED': 4,									# integer enabling reproducibility
    'NAGENTSPARALLEL': 8,						# integer defining parallelized agent runs
    'REWARDMINTOSAVE': 0.0,  					# float defining minimal reward to save a model
    'RENDER': False,							# boolean to define if displaying runs
    'RENDEREVERY': 1000,						# integer defining runs displayed
    'BESTAGENTANIMATION': True,					# boolean defining whether to save animation of best agent per generation
    'KEEPMODELHISTORY': True					# boolean defining whether to keep all agents throughout evolution
}

# hyperparameters
hyParams = {
    'NAGENTS': 1024,							# integer defining number of agents
    'NAGENTELITES': 600,						# integer defining number of agents considered as parents
    'NGENERATIONS': 100,						# integer defining number of generations for evolution
    'NGAMESAVERAGED': 1,						# integer defining number of games played for averaging
    'NAGENTSTEPS': 200,							# integer defining number of episodes per agent
    'MUTATIONPROBABILITY': 0.5,					# float defining fraction of mutated parameters
	'MUTATIONPOWER': 0.005,						# float defining mutation, 0.02 after https://arxiv.org/pdf/1712.06567.pdf
	'NHIDDENNODES': [40] * 4,					# list of integers of nodes per hidden layer to define architecture
	'ARCHITECTUREVARY': True,					# boolean defining to allow architecture variation
	'HIDDENACTIVATIONS': ['relu'] * 4,			# list of strings defining hidden nodal activations
    'BATCHNORMALIZATION': True					# boolean defining batch normalization
}


def main(envSettings, hyParams):
    # initializing environment
    env = FloPyEnv(
        envSettings['ENVTYPE'],
        envSettings['PATHMF2005'],
        envSettings['PATHMP6'],
        MODELNAME=envSettings['MODELNAME'],
        flagRender=envSettings['RENDER'],
        NAGENTSTEPS=hyParams['NAGENTSTEPS'])

    # initializing agent
    agent = FloPyAgent(
        env.observationsVector,
        env.actionSpace,
        hyParams,
        envSettings,
        'genetic')
    # agent.GPUAllowMemoryGrowth()

    # running genetic search algorithm
    agent.runGenetic(env)


if __name__ == '__main__':
    main(envSettings, hyParams)