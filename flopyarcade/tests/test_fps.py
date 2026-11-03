#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unittest
from collections import deque
import time


class TestRenderFPS(unittest.TestCase):

    def test_renderFPS_creates_text_artist(self):
        """Verify renderFPS() creates a text artist on the canvas."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(
            ENVTYPE='1s-d', _seed=1, MODELNAME='test_fps',
            ANIMATIONFOLDER='test_fps', flagRender=True,
            NAGENTSTEPS=5, nLay=1, nRow=10, nCol=10,
            initWithSolution=True
        )
        env.renderInitializeCanvas()
        now = time.time()
        for i in range(60):
            env._frame_times.append(now - (60 - i) * (1.0/60.0))
        env.renderFPS()
        self.assertIsNotNone(getattr(env, '_fps_artist', None))
        plt.close('all')

    def test_fps_calculation_correct(self):
        """Verify FPS is calculated correctly from known frame times."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(
            ENVTYPE='1s-d', _seed=1, MODELNAME='test_fps_calc',
            ANIMATIONFOLDER='test_fps_calc', flagRender=True,
            NAGENTSTEPS=5, nLay=1, nRow=10, nCol=10,
            initWithSolution=True
        )
        env.renderInitializeCanvas()
        now = time.time()
        env._frame_times.clear()
        for i in range(60):
            env._frame_times.append(now - (60 - i) * (1.0/60.0))
        env.renderFPS()
        self.assertAlmostEqual(env._fps, 59.0, delta=2.0)
        plt.close('all')


class TestFPSAttributes(unittest.TestCase):

    def test_fps_attrs_exist(self):
        """Verify _fps and _frame_times attributes exist on FloPyEnv."""
        from flopyarcade import FloPyEnv
        env = FloPyEnv(
            ENVTYPE='1s-d', _seed=1, MODELNAME='test_fps_attr',
            ANIMATIONFOLDER='test_fps_attr', flagRender=False,
            NAGENTSTEPS=5, nLay=1, nRow=10, nCol=10,
            initWithSolution=True
        )
        self.assertEqual(env._fps, 0.0)
        self.assertIsInstance(env._frame_times, deque)
        self.assertEqual(env._frame_times.maxlen, 60)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
