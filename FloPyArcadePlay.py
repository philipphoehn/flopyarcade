#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyArcade, FloPyAgent
FloPyAgent.suppressTensorFlowWarnings(FloPyAgent)
from matplotlib.pyplot import switch_backend
try: switch_backend('TkAgg')
except: print('Could not import TkAgg as a backend. Visualization may not show.')
from numpy.random import randint

# environment settings
envSettings = {
    'ENVTYPE': '3',
    'MODELNAMELOAD': 'test',
    'MODELNAME': None,
    'PATHMF2005': None,
    'PATHMP6': None,
    'SURROGATESIMULATOR': None,
    'SAVEPLOT': False,
    'MANUALCONTROL': True,
    'RENDER': True,
    'ENVSEED': 1,
    'NLAY': 1,
    'NROW': 100,
    'NCOL': 100
}

# game settings
gameSettings = {
    'NGAMES': 1000000000,
    'NAGENTSTEPS': 200
}


def main(envSettings, gameSettings):

    game = FloPyArcade(
        modelNameLoad=envSettings['MODELNAMELOAD'],
        modelName=envSettings['MODELNAME'],
        NAGENTSTEPS=gameSettings['NAGENTSTEPS'],
        PATHMF2005=envSettings['PATHMF2005'],
        PATHMP6=envSettings['PATHMP6'],
        surrogateSimulator=None,
        flagSavePlot=envSettings['SAVEPLOT'],
        flagManualControl=envSettings['MANUALCONTROL'],
        flagRender=envSettings['RENDER'],
        nLay=envSettings['NLAY'],
        nRow=envSettings['NROW'],
        nCol=envSettings['NCOL']
        )

    for run in range(gameSettings['NGAMES']):
        print('game #:', run + 1)
        import time
        t0 = time.time()
        randomInteger = randint(100000000, size=1)[0]
        game.playApp(
            ENVTYPE=envSettings['ENVTYPE'],
            seed=randomInteger
            )
        print('time taken', time.time() - t0)
        # if game.success:
        #     break


if __name__ == '__main__':
    main(envSettings, gameSettings)
