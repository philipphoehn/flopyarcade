#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


from argparse import ArgumentParser
from flopyarcade import FloPyArcade, FloPyAgent
FloPyAgent.suppressTensorFlowWarnings(FloPyAgent)
from matplotlib.pyplot import switch_backend
try: switch_backend('TkAgg')
except: print('Could not import TkAgg as a backend. Visualization may not show.')
from numpy.random import randint

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = ArgumentParser(description='FloPyArcade game')
parser.add_argument('--envtype', default='3s-d', type=str,
    help='string defining environment')
parser.add_argument('--seedenv', default=1, type=int,
    help='integer enabling reproducibility of the environments')
parser.add_argument('--nlay', default=1, type=int,
    help='integer defining numbers of model layers')
parser.add_argument('--nrow', default=100, type=int,
    help='integer defining grid rows')
parser.add_argument('--ncol', default=100, type=int,
    help='integer defining grid columns')
parser.add_argument('--saveplot', default=False, type=str_to_bool,
    help='boolean defining whether to save game animation')
parser.add_argument('--manualcontrol', default=False, type=str_to_bool,
    help='boolean defining manual control model')
parser.add_argument('--render', default=True, type=str_to_bool,
    help='boolean defining if displaying runs')
parser.add_argument('--modelnameload', default=None, type=str,
    help='string defining model to load')
parser.add_argument('--modelname', default='FloPyArcade', type=str,
    help='string defining model basename')
parser.add_argument('--pathmf2005', default=None, type=str,
    help='string defining local path to MODFLOW 2005 executable')
parser.add_argument('--pathmp6', default=None, type=str,
    help='string defining local path to MODPATH 6 executable')
parser.add_argument('--surrogatesimulator', default=None,
    help='currently unavailable')
parser.add_argument('--ngames', default=100, type=int,
    help='integer defining number of games played in a row')
parser.add_argument('--nagentsteps', default=200, type=int,
    help='integer defining the number of steps taken per game')
args = parser.parse_args()


# environment settings
envSettings = {
    'ENVTYPE': args.envtype,                            # string defining environment
    'SEEDENV': args.seedenv,                            # integer enabling reproducibility of the environments
    'NLAY': args.nlay,                                  # integer defining numbers of model layers
    'NROW': args.nrow,                                  # integer defining grid rows
    'NCOL': args.ncol,                                  # integer defining grid columns
    'SAVEPLOT': args.saveplot,                          # boolean defining whether to save game animation
    'MANUALCONTROL': args.manualcontrol,                # boolean defining manual control model
    'RENDER': args.render,                              # boolean defining if displaying runs
    'MODELNAMELOAD': args.modelnameload,                # string defining model to load
    'MODELNAME': args.modelname,                        # string defining model basename
    'PATHMF2005': args.pathmf2005,                      # string defining local path to MODFLOW 2005 executable
    'PATHMP6': args.pathmp6,                            # string defining local path to MODPATH 6 executable
    'SURROGATESIMULATOR': args.surrogatesimulator       # currently unavailable
}

# game settings
gameSettings = {
    'NGAMES': args.ngames,                              # integer defining number of games played in a row
    'NAGENTSTEPS': args.nagentsteps                     # integer defining the number of steps taken per game
}


def main(envSettings, gameSettings):

    game = FloPyArcade(
        ENVTYPE=envSettings['ENVTYPE'],
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
        game.play(
            ENVTYPE=envSettings['ENVTYPE'],
            seed=randomInteger
            )
        print('time taken', time.time() - t0)
        # if game.success:
        #     break


if __name__ == '__main__':
    main(envSettings, gameSettings)