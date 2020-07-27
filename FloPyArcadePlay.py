#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyArcade, FloPyAgent
FloPyAgent.suppressTensorFlowWarnings(FloPyAgent)
from tensorflow.keras.models import load_model as TFload_model
from matplotlib.pyplot import switch_backend
try: switch_backend('TkAgg')
except: print('Could not import TkAgg as a backend. Visualization may not show.')

# environment settings
envSettings = {
    'ENVTYPE': '4',
    'MODELNAMELOAD': None,
    'MODELNAME': None,
    'PATHMF2005': None,
    'PATHMP6': None,
    'SURROGATESIMULATOR': None,
    'SAVEPLOT': False,
    'MANUALCONTROL': False,
    'RENDER': True,
    'ENVSEED': 1,
    'NLAY': 1,
    'NROW': 100,
    'NCOL': 100
}

# game settings
gameSettings = {
    'NGAMES': 10,
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
        game.play(
            ENVTYPE=envSettings['ENVTYPE'],
            seed=envSettings['ENVSEED']+run
            )
        print('time taken', time.time() - t0)
        # if game.success:
        #     break

if __name__ == '__main__':
    main(envSettings, gameSettings)
