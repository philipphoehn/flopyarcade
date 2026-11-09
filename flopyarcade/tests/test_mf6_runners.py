"""Tests for mf6_mp7_runner.py and mf6_prt_runner.py"""
import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch, PropertyMock


class TestMF6MP7Runner:
    def make_env(self, **kwargs):
        s = MagicMock()
        s.MODELNAME = 'test_model'
        s.model_ws = '/tmp/test_mf6_mp7'
        s.exe_name = 'mf6'
        s.periodSteadiness = True
        s.periods = 5
        s.periodLength = 10
        s.periodSteps = 2
        s.nLay = 2
        s.nRow = 5
        s.nCol = 5
        s.dRow = 10.0
        s.dCol = 10.0
        s.zTop = 100.0
        s.botM = np.array([100.0, 50.0, 0.0])
        s.strt = 90.0
        s.ENVTYPE = '1s-d'
        s.headSpecWest = 95.0
        s.headSpecEast = 95.0
        s.actionValueSouth = 90.0
        s.actionValueNorth = 90.0
        s.actionValue = 90.0
        s.headSpecSouth = 90.0
        s.headSpecNorth = 90.0
        s.wellCellLayer = 1
        s.wellCellColumn = 3
        s.wellCellRow = 3
        s.wellQ = -10.0
        s.nHelperWells = 0
        s.rechTimeSeries = None
        s.timeStep = 1
        s.particleCoords = [2.5, 3.5, 0.5]
        s.dVer = 50.0
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    def test_init(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env()
        runner = MF6MP7Runner(s)
        assert runner._s is s

    def test_build_model(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env()
        with patch('flopyarcade.mf6_mp7_runner.MFSimulation') as MockSim, \
             patch('flopyarcade.mf6_mp7_runner.ModflowTdis'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowIms'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwf'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfdis'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfic'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfnpf'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfchd'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfwel'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfoc'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfrcha'), \
             patch('os.makedirs'):
            runner = MF6MP7Runner(s)
            runner.build_model()
            assert s.sim.write_sim.called

    def test_build_model_non_steady(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env(periodSteadiness=False)
        with patch('flopyarcade.mf6_mp7_runner.MFSimulation'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowTdis'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowIms'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwf'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfdis'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfic'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfnpf'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfchd'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfwel'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfoc'), \
             patch('os.makedirs'):
            runner = MF6MP7Runner(s)
            runner.build_model()

    def test_build_model_with_recharge(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env(rechTimeSeries=[np.ones((5,5))])
        with patch('flopyarcade.mf6_mp7_runner.MFSimulation'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowTdis'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowIms'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwf'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfdis'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfic'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfnpf'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfchd'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfwel'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfoc'), \
             patch('flopyarcade.mf6_mp7_runner.ModflowGwfrcha'), \
             patch('os.makedirs'):
            runner = MF6MP7Runner(s)
            runner.build_model()
            assert s.gwf_rcha is not None

    def test_add_chd_1s(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env(ENVTYPE='1s-d')
        s.gwf = MagicMock()
        with patch('flopyarcade.mf6_mp7_runner.ModflowGwfchd') as MockChd:
            runner = MF6MP7Runner(s)
            runner._add_chd()
            MockChd.assert_called_once()

    def test_add_chd_2s(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env(ENVTYPE='2s-c')
        s.gwf = MagicMock()
        with patch('flopyarcade.mf6_mp7_runner.ModflowGwfchd'):
            runner = MF6MP7Runner(s)
            runner._add_chd()

    def test_add_chd_other(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env(ENVTYPE='3s-d')
        s.gwf = MagicMock()
        with patch('flopyarcade.mf6_mp7_runner.ModflowGwfchd'):
            runner = MF6MP7Runner(s)
            runner._add_chd()

    def test_add_wel_no_helpers(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env(nHelperWells=0)
        s.gwf = MagicMock()
        with patch('flopyarcade.mf6_mp7_runner.ModflowGwfwel'):
            runner = MF6MP7Runner(s)
            runner._add_wel()

    def test_add_wel_with_helpers(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env(nHelperWells=2)
        s.helperWells = {'l1': 1, 'c1': 2, 'r1': 3, 'wellQ1': -5.0,
                         'l2': 1, 'c2': 4, 'r2': 2, 'wellQ2': -3.0}
        s.gwf = MagicMock()
        with patch('flopyarcade.mf6_mp7_runner.ModflowGwfwel'):
            runner = MF6MP7Runner(s)
            runner._add_wel()

    def test_run_modflow(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env()
        s.sim = MagicMock()
        head_data = np.ones((2, 5, 5))
        s.gwf = MagicMock()
        s.gwf.output.return_value.head.get_data.return_value = head_data
        runner = MF6MP7Runner(s)
        runner.run_modflow()
        assert s.sim.run_sim.called
        np.testing.assert_array_equal(s.heads, head_data)

    def test_run_modpath7(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env(timeStep=1)
        s._transform_particle_coords = MagicMock()
        with patch.object(MF6MP7Runner, '_init_mp7'), \
             patch.object(MF6MP7Runner, '_write_mp7_particle'):
            s.mp7 = MagicMock()
            runner = MF6MP7Runner(s)
            runner.run_modpath7()
            s.mp7.run_model.assert_called_with(silent=True)

    def test_run_modpath7_not_first_step(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env(timeStep=3)
        s._transform_particle_coords = MagicMock()
        with patch.object(MF6MP7Runner, '_init_mp7') as m_init, \
             patch.object(MF6MP7Runner, '_write_mp7_particle'):
            s.mp7 = MagicMock()
            runner = MF6MP7Runner(s)
            runner.run_modpath7()
            m_init.assert_not_called()

    def test_init_mp7(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env()
        with patch('flopyarcade.mf6_mp7_runner.exists', return_value=True), \
             patch('flopyarcade.mf6_mp7_runner.Modpath7') as MockMP7, \
             patch('flopyarcade.mf6_mp7_runner.Modpath7Bas'):
            MockMP7.return_value.create_mp7sim.return_value = 'sim'
            runner = MF6MP7Runner(s)
            runner._init_mp7()
            MockMP7.assert_called_once()

    def test_write_mp7_particle(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env()
        s.cellInfoFromCoordinates.return_value = (0, 2, 2)
        import tempfile
        s.model_ws = tempfile.mkdtemp()
        runner = MF6MP7Runner(s)
        runner._write_mp7_particle()
        mploc = os.path.join(s.model_ws, 'test_model.mploc')
        assert os.path.exists(mploc)
        os.remove(mploc)

    def test_load_particle_data(self):
        from flopyarcade.mf6_mp7_runner import MF6MP7Runner
        s = self.make_env()
        import tempfile
        s.model_ws = tempfile.mkdtemp()
        pth = os.path.join(s.model_ws, 'test_model.pth')
        with open(pth, 'w') as f:
            f.write('1,2,3\n4,5,6\n')
        with patch('flopyarcade.mf6_mp7_runner.exists', return_value=True), \
             patch('pandas.read_csv') as mock_read:
            mock_read.return_value = 'df'
            runner = MF6MP7Runner(s)
            result = runner.load_particle_data()
            assert result == 'df'


class TestMF6PRTRunner:
    def make_env(self, **kwargs):
        s = MagicMock()
        s.MODELNAME = 'prt_test'
        s.model_ws = '/tmp/test_mf6_prt'
        s.exe_name = 'mf6'
        s.nLay = 2
        s.nRow = 5
        s.nCol = 5
        s.dRow = 10.0
        s.dCol = 10.0
        s.zTop = 100.0
        s.botM = np.array([100.0, 50.0, 0.0])
        s.wellCellLayer = 1
        s.wellCellColumn = 3
        s.wellCellRow = 3
        s.wellQ = -10.0
        s.nHelperWells = 0
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    def test_get_exe(self):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        s = self.make_env()
        s.simulators_dir = '/tmp/nonexistent'
        result = MF6PRTMixin._mf6prt_get_exe(s)
        assert isinstance(result, str)

    def test_cell_from_coords(self):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        s = self.make_env()
        s.dCol = 10.0
        s.dRow = 10.0
        s.nCol = 5
        s.nRow = 5
        cell = MF6PRTMixin._mf6prt_cell_from_coords(s, 15.0, 25.0)
        assert cell[0] == 2  # row (5-1-2=2)
        assert cell[1] == 1  # col (15/10=1)

    def test_get_wells(self):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        s = self.make_env(nHelperWells=0)
        s.helperWells = {}
        wells = MF6PRTMixin._mf6prt_get_wells(s)
        assert len(wells) == 1
        assert wells[0][1] == -10.0

    def test_get_wells_with_helpers(self):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        s = self.make_env(nHelperWells=2)
        s.helperWells = {'l1': 1, 'c1': 2, 'r1': 3, 'wellQ1': -5.0,
                         'l2': 1, 'c2': 4, 'r2': 2, 'wellQ2': -3.0}
        wells = MF6PRTMixin._mf6prt_get_wells(s)
        assert len(wells) == 3

    def test_get_hydraulics(self):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        s = self.make_env()
        result = MF6PRTMixin._mf6prt_get_hydraulics(s)
        assert result is not None

    def test_parse_track_csv(self):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        s = self.make_env()
        import tempfile
        csv_path = tempfile.mktemp(suffix='.csv')
        with open(csv_path, 'w') as f:
            f.write('x,y,z,t\n1.0,2.0,3.0,0.0\n')
        result = MF6PRTMixin._mf6prt_parse_track_csv(s, csv_path)
        os.unlink(csv_path)

