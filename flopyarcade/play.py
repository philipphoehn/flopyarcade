#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPy Arcade game
# author: Philipp Hoehn
# philipp.hoehn@yahoo.com


import argparse
from argparse import ArgumentParser
from flopyarcade import FloPyArcade, FloPyAgent
FloPyAgent.suppressTensorFlowWarnings(FloPyAgent)
from matplotlib.pyplot import switch_backend
from numpy.random import randint
import time
try: switch_backend('TkAgg')
except: print('Could not import TkAgg as a backend. Visualization may not show.')

from flopyarcade import FloPyAgent
from flopyarcade import FloPyEnv

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


def has_arguments(parser, args):
    # Compare each argument value to its default from the parser
    for action in parser._actions:
        # Skip help action
        if isinstance(action, argparse._HelpAction):
            continue
        arg_name = action.dest
        default = action.default
        value = getattr(args, arg_name, None)
        # If any argument value differs from default, arguments were provided
        if value != default:
            return True
    return False


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main(
    envSettings={
    'ENVTYPE': '3s-d',
    'SEEDENV': 1,
    'NLAY': 1,
    'NROW': 100,
    'NCOL': 100,
    'SAVEPLOT': False,
    'MANUALCONTROL': False,
    'RENDER': True,
    'BLIT': True,
    'SMOOTH_FRAMES': 4,
    'SMOOTH_SUBDIV': 10,
    'MODELNAMELOAD': None,
    'MODELNAME': 'FloPyArcade',
    'PATHMF2005': None,
    'PATHMP6': None,
    'SURROGATESIMULATOR': None
    },
    gameSettings={
    'NGAMES': 100,
    'NAGENTSTEPS': 200
    }):

    parser = ArgumentParser(description='FloPyArcade game', allow_abbrev=False)
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
    parser.add_argument('--blit', default=True, type=str_to_bool,
        help='boolean enabling matplotlib blitting for 60+ FPS (default: True)')
    parser.add_argument('--smooth-frames', default=4, type=int,
        help='number of sub-frames for smooth particle transition (default: 4, set 1 to disable)')
    parser.add_argument('--smooth-subdiv', default=10, type=int,
        help='PCHIP subdivision factor for trajectory smoothing (default: 10)')
    parser.add_argument('--split-screen', default=False, type=str_to_bool,
        help='enable 2-player split-screen mode (Left: WASD, Right: arrows)')
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
    args, _ = parser.parse_known_args()

    # print(has_arguments(parser, args))

    if has_arguments(parser, args):
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
            'BLIT': args.blit,                              # boolean enabling matplotlib blitting
            'SMOOTH_FRAMES': args.smooth_frames,            # int sub-frames for smooth particle
            'SMOOTH_SUBDIV': args.smooth_subdiv,            # int PCHIP subdivision factor
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

    # print('envSettings[ENVTYPE]', envSettings['ENVTYPE'])

    # B2: split-screen mode dispatch
    if getattr(args, 'split_screen', False) if has_arguments(parser, args) else False:
        from flopyarcade.play_split import SplitScreenGame
        game_split = SplitScreenGame(
            envtype=args.envtype,
            seed=args.seedenv,
            nlay=args.nlay,
            nrow=args.nrow,
            ncol=args.ncol,
            nagentsteps=args.nagentsteps,
            smooth_frames=args.smooth_frames,
            smooth_subdiv=args.smooth_subdiv,
            pathmf2005=args.pathmf2005,
            pathmp6=args.pathmp6,
            modelname=args.modelname
        )
        game_split.play(ngames=args.ngames)
        return None

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
        flagBlit=envSettings['BLIT'],
        smoothFrames=envSettings['SMOOTH_FRAMES'],
        smoothSubdiv=envSettings['SMOOTH_SUBDIV'],
        nLay=envSettings['NLAY'],
        nRow=envSettings['NROW'],
        nCol=envSettings['NCOL'],
        )

    for run in range(gameSettings['NGAMES']):
        print('game #:', run + 1)
        t0 = time.time()
        randomInteger = randint(100000000, size=1)[0] if envSettings['SEEDENV'] == None else envSettings['SEEDENV'] # randint(100000000, size=1)[0]
        game.play(
            ENVTYPE=envSettings['ENVTYPE'],
            seed=randomInteger,
            verbose=True
            )
        print('time taken', time.time() - t0)
        # if game.success:
        #     break

    return game.env.envhash


if __name__ == '__main__':
    main()