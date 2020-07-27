#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyArcade, FloPyAgent
FloPyAgent.suppressTensorFlowWarnings(FloPyAgent)
from tensorflow.keras.models import load_model as TFload_model
from matplotlib.pyplot import switch_backend
switch_backend('TkAgg')

# environment settings
envSettings = {
    'ENVTYPE': '3',
    'MODELNAMELOAD': 'gs-e3s1n20000e100g10000av1st200mpr1e-0mpo3e-3ar250x5v1relubn1_res100_ns-ev1e10000_gen000001_avg__734.5',
    # 'MODELNAMELOAD': None,
    'MODELNAME': None,
    'PATHMF2005': None,
    'PATHMP6': None,
    'SURROGATESIMULATOR': None,
    # 'SURROGATESIMULATOR': 'env3_surrogateSimulator_ar3x500lr1e-04ep10000pat10',
    'SAVEPLOT': False,
    'MANUALCONTROL': False,
    'RENDER': True,
    'ENVSEED': 3,
    'NLAY': 1,
    'NROW': 100,
    'NCOL': 100
}

# game settings
gameSettings = {
    'NGAMES': 5,
    'NAGENTSTEPS': 200
}


def main(envSettings, gameSettings):

    # modelSteady = TFload_model('C:\\FloPyArcade\\dev\\surrogateModelInitialState_ar3x100lr6e-04bs32784ep10000pat10.h5')
    # modelTransient = TFload_model('C:\\FloPyArcade\\dev\\surrogateModelNewwith-1States_ar3x500lr6e-04bs32784ep10000pat10.h5')

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
    from vispy import app
    # app.use_app('glfw')  # for testing specific backends
    app.set_interactive()
    main(envSettings, gameSettings)