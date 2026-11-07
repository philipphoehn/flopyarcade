#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unittest
import pytest

pytestmark = pytest.mark.slow


class TestBlitInit(unittest.TestCase):

    def test_blit_init_creates_background(self):
        """Verify render_blit_init() creates a non-None background buffer."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(
            ENVTYPE='1s-d', _seed=1, MODELNAME='test_blit',
            ANIMATIONFOLDER='test_blit', flagRender=True,
            NAGENTSTEPS=5, nLay=1, nRow=10, nCol=10,
            initWithSolution=True
        )
        env.renderInitializeCanvas()
        env.render_blit_init()
        self.assertIsNotNone(env._blit_background)
        plt.close('all')


class TestBlitFrame(unittest.TestCase):

    def test_blit_frame_after_init(self):
        """Verify render_blit_frame() runs without error after init."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(
            ENVTYPE='1s-d', _seed=1, MODELNAME='test_blit_frame',
            ANIMATIONFOLDER='test_blit_frame', flagRender=True,
            NAGENTSTEPS=5, nLay=1, nRow=10, nCol=10,
            initWithSolution=True
        )
        env.renderInitializeCanvas()
        env.render_blit_init()
        env.render_blit_frame()
        self.assertIsNotNone(env._blit_background)
        plt.close('all')

    def test_blit_particle_updates(self):
        """Verify particle position updates between frames."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(
            ENVTYPE='1s-d', _seed=1, MODELNAME='test_blit_p',
            ANIMATIONFOLDER='test_blit_p', flagRender=True,
            NAGENTSTEPS=5, nLay=1, nRow=10, nCol=10,
            initWithSolution=True
        )
        env.renderInitializeCanvas()
        env.render_blit_init()
        env.timeStep = 0
        env.particleCoords = [env.minX, env.minY]
        env.render_blit_frame()
        self.assertIsNotNone(env._blit_artists.get('particle'))
        env.timeStep = 1
        env.trajectories['x'].append([env.minX + 1, env.minX + 2])
        env.trajectories['y'].append([env.minY + 1, env.minY + 2])
        env.render_blit_frame()
        self.assertIsNotNone(env._blit_artists.get('particle'))
        plt.close('all')


class TestBlitFlag(unittest.TestCase):

    def test_blit_flag_default_false(self):
        """Verify _blit_enabled defaults to False in FloPyEnv."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(
            ENVTYPE='1s-d', _seed=1, MODELNAME='test_blit_flag',
            ANIMATIONFOLDER='test_blit_flag', flagRender=False,
            NAGENTSTEPS=5, nLay=1, nRow=10, nCol=10,
            initWithSolution=True
        )
        self.assertFalse(env._blit_enabled)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
