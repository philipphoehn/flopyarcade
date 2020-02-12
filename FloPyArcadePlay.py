#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from FloPyArcade import FloPyArcade

# environment settings
envSettings = {
    'ENVTYPE': '3',
    'MODELNAMELOAD': 'unittest',
    'MODELNAME': None,
    'PATHMF2005': None,
    'PATHMP6': None,
    'SAVEPLOT': True,
    'MANUALCONTROL': True,
    'RENDER': False
}

# game settings
gameSettings = {
    'NGAMES': 10,
    'NAGENTSTEPS': 1000
}


def main(envSettings, gameSettings):
    game = FloPyArcade(
    	modelNameLoad=envSettings['MODELNAMELOAD'],
        modelName=envSettings['MODELNAME'],
        NAGENTSTEPS=gameSettings['NAGENTSTEPS'],
        PATHMF2005=envSettings['PATHMF2005'],
        PATHMP6=envSettings['PATHMP6'],
        flagSavePlot=envSettings['SAVEPLOT'],
        flagManualControl=envSettings['MANUALCONTROL'],
        flagRender=envSettings['RENDER'])

    for run in range(gameSettings['NGAMES']):
        print('game #:', run + 1)
        game.play(
            ENVTYPE=envSettings['ENVTYPE'])
        if game.done:
            if game.success:
                break


if __name__ == '__main__':
    main(envSettings, gameSettings)