"""Tests covering missed lines in play_againstai.py and mf6_prt_runner.py."""
import os
import sys
import csv
import tempfile
import shutil
import importlib
import numpy as np
import pandas as pd
import pytest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch, PropertyMock, call
from pathlib import Path


def apply_patches(specs):
    """Apply patches. specs: list of (patch_obj, name_or_None). Returns (stack, mocks)."""
    stack = ExitStack()
    mocks = {}
    for p, name in specs:
        stack.enter_context(p)
        if name:
            mocks[name] = p.new
    return stack, mocks


# ---------------------------------------------------------------------------
# PLAY_AGAINSTAI TESTS
# ---------------------------------------------------------------------------

class TestPlayAgainstAIImport:
    """Test module-level code in play_againstai.py (lines 65-66)."""

    @patch('sys.argv', ['play_againstai.py', '--envtype', '1s-d'])
    def test_import_default(self):
        for mod in list(sys.modules.keys()):
            if 'play_againstai' in mod:
                del sys.modules[mod]
        mod = importlib.import_module('flopyarcade.play_againstai')
        assert mod.args.playbenchmark is False
        assert mod.args.external is True
        assert mod.ENVTYPE == '1s-d'

    @patch('sys.argv', ['play_againstai.py', '--envtype', '2s-c', '--playbenchmark', 'True', '--external', 'True'])
    def test_import_playbenchmark_external(self, capsys):
        for mod in list(sys.modules.keys()):
            if 'play_againstai' in mod:
                del sys.modules[mod]
        mod = importlib.import_module('flopyarcade.play_againstai')
        captured = capsys.readouterr()
        assert 'Starting benchmark visualization' in captured.out
        assert mod.args.playbenchmark is True
        assert mod.args.external is True

    @patch('sys.argv', ['play_againstai.py', '--envtype', '1s-d', '--playbenchmark', 'True', '--external', 'False'])
    def test_import_playbenchmark_not_external(self):
        for mod in list(sys.modules.keys()):
            if 'play_againstai' in mod:
                del sys.modules[mod]
        mod = importlib.import_module('flopyarcade.play_againstai')
        assert mod.args.playbenchmark is True
        assert mod.args.external is False

    def test_str_to_bool(self):
        for mod in list(sys.modules.keys()):
            if 'play_againstai' in mod:
                del sys.modules[mod]
        with patch('sys.argv', ['x']):
            mod = importlib.import_module('flopyarcade.play_againstai')
        assert mod.str_to_bool('true') is True
        assert mod.str_to_bool('false') is False
        assert mod.str_to_bool('1') is True
        assert mod.str_to_bool('0') is False
        assert mod.str_to_bool(True) is True
        with pytest.raises(ValueError):
            mod.str_to_bool('invalid')


class TestPlayAgainstAIFunctions:
    """Test the test() function and env_creator."""

    def _get_module(self):
        for mod in list(sys.modules.keys()):
            if 'play_againstai' in mod:
                del sys.modules[mod]
        with patch('sys.argv', ['play_againstai.py']):
            return importlib.import_module('flopyarcade.play_againstai')

    def test_env_creator(self):
        mod = self._get_module()
        with patch.object(mod, 'FloPyEnv', return_value=MagicMock()) as mock_env:
            env = mod.env_creator({'anything': 1})
            mock_env.assert_called_once()

    def test_test_function_single_step(self):
        mod = self._get_module()

        agent = MagicMock()
        agent.compute_single_action.return_value = 0

        env = MagicMock()
        env.done = False
        env.observationsVectorNormalized = np.zeros(84)
        env.MANUALCONTROLTIME = 0.01
        env.actionSpace = ['keep', 'up', 'down']
        env.render.return_value = np.zeros((100, 200, 3))

        # Make step set env.done=True so the while loop exits after one iteration
        def step_side_effect(action):
            env.done = True
            return (np.zeros(84), 1.0, True, {})
        env.step.side_effect = step_side_effect

        fig = MagicMock()
        fig.add_gridspec.return_value = MagicMock()
        fig.add_subplot.return_value = MagicMock()
        fig.canvas.mpl_connect.return_value = None
        fig_manager = MagicMock()
        fig_manager.window.state.side_effect = Exception("no zoom")
        fig_manager.full_screen_toggle.side_effect = Exception("no fullscreen")

        import matplotlib.pyplot as plt
        import matplotlib.image as mpl_image

        specs = [
            (patch('copy.deepcopy', new=env), None),
            (patch.object(mod, 'FloPyAgent', new=MagicMock()), None),
            (patch.object(plt, 'show', new=MagicMock()), None),
            (patch.object(plt, 'waitforbuttonpress', new=MagicMock()), None),
            (patch.object(plt, 'pause', new=MagicMock()), None),
            (patch.object(plt, 'figure', new=fig), None),
            (patch.object(plt, 'cla', new=MagicMock()), None),
            (patch.object(plt, 'clf', new=MagicMock()), None),
            (patch.object(plt, 'close', new=MagicMock()), None),
            (patch.object(plt, 'tight_layout', new=MagicMock()), None),
            (patch.object(plt, 'switch_backend', new=MagicMock()), None),
            (patch.object(plt, 'get_current_fig_manager', new=fig_manager), None),
            (patch.object(mpl_image, 'imread', new=MagicMock(return_value=np.zeros((10, 10, 3)))), None),
        ]
        stack, _ = apply_patches(specs)
        try:
            reward = mod.test(agent, env)
            assert reward == pytest.approx(1.0)
        finally:
            stack.close()


class TestPlayAgainstAIMain:
    """Test the __main__ block (lines 296-386)."""

    def _get_dedented_main_block(self, pa):
        src_path = Path(pa.__file__).resolve()
        source = src_path.read_text()
        main_marker = 'if __name__ == "__main__":'
        idx = source.find(main_marker)
        assert idx != -1, "Could not find __main__ block"
        main_block = source[idx + len(main_marker):]
        lines = main_block.split('\n')
        non_empty = [l for l in lines if l.strip()]
        indent = len(non_empty[0]) - len(non_empty[0].lstrip()) if non_empty else 0
        dedented = '\n'.join(l[indent:] if l.strip() else '' for l in lines)
        return dedented, str(src_path)

    def _build_ns(self, pa, args, envtype, **extra_ns):
        src_path = Path(pa.__file__).resolve()
        ns = {
            'args': args,
            'ENVTYPE': envtype,
            'config': pa.config,
            'config_stopCriteria': pa.config_stopCriteria,
            'env_creator': pa.env_creator,
            'test': pa.test,
            'tune': pa.tune,
            'rayInit': MagicMock(name='rayInit'),
            'register_env': MagicMock(name='register_env'),
            'ApexTrainer': MagicMock(name='ApexTrainer'),
            'abspath': pa.abspath,
            'dirname': pa.dirname,
            'exists': lambda x: True,
            'join': pa.join,
            'makedirs': MagicMock(name='makedirs'),
            'FATAL': pa.FATAL,
            'os': os,
            'np': np,
            '__file__': str(src_path),
            'wrkspc': os.path.dirname(os.path.dirname(str(src_path))),
        }
        ns.update(extra_ns)
        return ns

    def test_main_training_path(self):
        import flopyarcade.play_againstai as pa
        mock_tune_result = MagicMock()
        dedented, _ = self._get_dedented_main_block(pa)
        mock_ray_init = MagicMock(name='rayInit')
        mock_register = MagicMock(name='register_env')

        with patch.object(pa.tune, 'run', new=MagicMock(return_value=mock_tune_result)) as mock_run:
            ns = self._build_ns(pa, pa.args, pa.ENVTYPE,
                               rayInit=mock_ray_init,
                               register_env=mock_register)
            exec(compile(dedented, '<main_block>', 'exec'), ns)
            mock_ray_init.assert_called_once()
            mock_run.assert_called_once()

    def test_main_playbenchmark_external_path(self):
        import flopyarcade.play_againstai as pa
        mock_args = MagicMock()
        mock_args.playbenchmark = True
        mock_args.external = True
        mock_args.envtype = '1s-d'
        mock_args.suffix = ''
        mock_args.cpus = 1
        mock_args.gpus = 0

        dedented, _ = self._get_dedented_main_block(pa)
        mock_ray_init = MagicMock(name='rayInit')
        mock_register = MagicMock(name='register_env')
        mock_system = MagicMock(name='os_system')
        mock_os = MagicMock(name='os_mock')
        mock_os.chdir = MagicMock()
        mock_os.system = mock_system
        mock_os.path = type('path', (), {'join': staticmethod(pa.join)})

        ns = self._build_ns(pa, mock_args, '1s-d',
                           rayInit=mock_ray_init,
                           register_env=mock_register,
                           os=mock_os)

        exec(compile(dedented, '<main_block>', 'exec'), ns)
        mock_system.assert_called_once()
        cmd = mock_system.call_args[0][0]
        assert 'play_againstai.py' in cmd
        assert '--playbenchmark' in cmd

    def test_main_playbenchmark_external_with_suffix(self):
        import flopyarcade.play_againstai as pa
        mock_args = MagicMock()
        mock_args.playbenchmark = True
        mock_args.external = True
        mock_args.envtype = '2s-c'
        mock_args.suffix = '_test'
        mock_args.cpus = 2
        mock_args.gpus = 1

        dedented, _ = self._get_dedented_main_block(pa)
        mock_ray_init = MagicMock(name='rayInit')
        mock_register = MagicMock(name='register_env')
        mock_system = MagicMock(name='os_system')
        mock_os = MagicMock(name='os_mock')
        mock_os.chdir = MagicMock()
        mock_os.system = mock_system
        mock_os.path = type('path', (), {'join': staticmethod(pa.join)})

        ns = self._build_ns(pa, mock_args, '2s-c',
                           rayInit=mock_ray_init,
                           register_env=mock_register,
                           os=mock_os)

        exec(compile(dedented, '<main_block>', 'exec'), ns)
        cmd = mock_system.call_args[0][0]
        assert '--suffix' in cmd
        assert '_test' in cmd

    def test_main_playbenchmark_not_external(self):
        import flopyarcade.play_againstai as pa
        mock_args = MagicMock()
        mock_args.playbenchmark = True
        mock_args.external = False
        mock_args.envtype = '1s-d'
        mock_args.suffix = ''
        mock_args.cpus = 1
        mock_args.gpus = 0

        mock_agent = MagicMock(name='agent')
        mock_apex = MagicMock(name='ApexTrainer', return_value=mock_agent)
        mock_env = MagicMock(name='env')

        dedented, _ = self._get_dedented_main_block(pa)
        mock_ray_init = MagicMock(name='rayInit')
        mock_register = MagicMock(name='register_env')

        ns = self._build_ns(pa, mock_args, '1s-d',
                           rayInit=mock_ray_init,
                           register_env=mock_register,
                           ApexTrainer=mock_apex,
                           env_creator=MagicMock(return_value=mock_env),
                           test=MagicMock(side_effect=Exception("break loop")))

        with pytest.raises(Exception, match="break loop"):
            exec(compile(dedented, '<main_block>', 'exec'), ns)
        mock_agent.restore.assert_called_once()


# ---------------------------------------------------------------------------
# MF6_PRT_RUNNER TESTS
# ---------------------------------------------------------------------------

class TestMF6PRTMixin:
    """Test MF6PRTMixin methods."""

    def _make_instance(self, **attrs):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = object.__new__(MF6PRTMixin)
        for k, v in attrs.items():
            setattr(inst, k, v)
        return inst

    def _patch_all_mf6(self, specs_extra=None):
        """Patch all MF6 classes used in _mf6prt_build_and_run with explicit new= mocks."""
        sim_cls = MagicMock(name='MFSimulation')
        base_specs = [
            (patch('flopyarcade.mf6_prt_runner.MFSimulation', new=sim_cls), 'sim_cls'),
            (patch('flopyarcade.mf6_prt_runner.ModflowTdis', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwf', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwfdis', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwfic', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwfnpf', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwfsto', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwfwel', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwfrcha', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwfchd', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwfoc', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowIms', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowPrt', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowPrtdis', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowPrtmip', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowPrtprp', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowPrtoc', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowEms', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.ModflowGwfprt', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.shutil.rmtree', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.os.makedirs', new=MagicMock()), None),
            (patch('flopyarcade.mf6_prt_runner.os.path.exists', new=MagicMock(return_value=True)), None),
        ]
        if specs_extra:
            base_specs.extend(specs_extra)
        return base_specs

    def test_get_exe_fallback(self):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = object.__new__(MF6PRTMixin)
        with patch('flopyarcade.mf6_prt_runner.exists', new=MagicMock(return_value=False)):
            result = inst._mf6prt_get_exe()
            assert 'simulators' in result
            assert 'mf6' in result

    def test_get_exe_found(self):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = object.__new__(MF6PRTMixin)
        with patch('flopyarcade.mf6_prt_runner.exists', new=MagicMock(return_value=True)):
            result = inst._mf6prt_get_exe()
            assert 'simulators' in result

    def test_cell_from_coords(self):
        inst = self._make_instance(dCol=10, dRow=10, nRow=5, nCol=5)
        row, col = inst._mf6prt_cell_from_coords(15, 20)
        assert row == 2
        assert col == 1

    def test_cell_from_coords_clamped(self):
        inst = self._make_instance(dCol=10, dRow=10, nRow=5, nCol=5)
        row, col = inst._mf6prt_cell_from_coords(1000, 20)
        assert col == 4
        row, col = inst._mf6prt_cell_from_coords(-100, 20)
        assert col == 0

    def test_get_wells_basic(self):
        inst = self._make_instance(
            wellCellLayer=1, wellCellRow=2, wellCellColumn=3, wellQ=-100.0
        )
        wells = inst._mf6prt_get_wells()
        assert len(wells) == 1
        # Format: [(layer, row, col), Q]
        assert wells[0] == [(0, 1, 2), -100.0]

    def test_get_wells_with_helpers(self):
        inst = self._make_instance(
            wellCellLayer=1, wellCellRow=2, wellCellColumn=3, wellQ=-100.0,
            nHelperWells=2,
            helperWells={
                'l1': 1, 'r1': 2, 'c1': 3, 'wellQ1': -50.0,
                'l2': 2, 'r2': 4, 'c2': 5, 'wellQ2': -75.0,
            }
        )
        wells = inst._mf6prt_get_wells()
        assert len(wells) == 3

    def test_get_wells_no_wells(self):
        inst = self._make_instance()
        wells = inst._mf6prt_get_wells()
        assert wells == []

    def test_get_hydraulics_defaults(self):
        inst = self._make_instance()
        hk, vka, ss, sy, chd_west, chd_east, rch = inst._mf6prt_get_hydraulics()
        assert hk == 10.0
        assert vka == 10.0
        assert ss == 1e-5
        assert sy == 0.15
        assert chd_west is None
        assert chd_east is None
        assert rch is None

    def test_get_hydraulics_custom(self):
        inst = self._make_instance(
            hk=5.0, chd_west=5.0, chd_east=4.0, rch_rate=0.001
        )
        hk, vka, ss, sy, chd_west, chd_east, rch = inst._mf6prt_get_hydraulics()
        assert hk == 5.0
        assert chd_west == 5.0
        assert chd_east == 4.0
        assert rch == 0.001

    def test_get_hydraulics_headspec(self):
        inst = self._make_instance(headSpecWest=3.0, headSpecEast=2.5)
        hk, vka, ss, sy, chd_west, chd_east, rch = inst._mf6prt_get_hydraulics()
        assert chd_west == 3.0
        assert chd_east == 2.5

    @pytest.mark.skip(reason="Tests BMI/XmiWrapper flow from PR#60 which was reverted from master")
    def test_build_and_run_full(self, tmp_path):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = self._make_instance(
            MODELNAME='test_model',
            nLay=1, nRow=3, nCol=3,
            dRow=10.0, dCol=10.0,
            zTop=100.0,
            periodSteps=10,
            strt=50.0,
            hk=10.0, chd_west=5.0, chd_east=4.0,
            wellCellLayer=1, wellCellRow=2, wellCellColumn=2, wellQ=-10.0,
            particleCoords=[15.0, 25.0, 50.0],
        )
        workdir = str(tmp_path / 'mf6prt_run')
        mock_df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0], 'time': [0.0]})

        mock_bmi = MagicMock(name='XmiWrapper')
        extra = [
            (patch('flopyarcade.mf6_prt_runner.exists', new=MagicMock(return_value=True)), None),
            (patch('xmipy.XmiWrapper', new=mock_bmi), 'xmi'),
            (patch.object(inst, '_mf6prt_parse_track_csv', new=MagicMock(return_value=mock_df)), None),
        ]
        specs = self._patch_all_mf6(extra)
        stack, mocks = apply_patches(specs)
        try:
            result = inst._mf6prt_build_and_run(workdir)
            mock_sim = mocks['sim_cls'].return_value
            mock_sim.write_simulation.assert_called_once()
            mock_bmi.return_value.initialize.assert_called_once()
            mock_bmi.return_value.execute.assert_called_once()
            mock_bmi.return_value.finalize.assert_called_once()
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ['x', 'y', 'z', 'time']
        finally:
            stack.close()

    def test_build_and_run_bmi_fallback_success(self, tmp_path):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = self._make_instance(
            MODELNAME='test_model',
            nLay=1, nRow=2, nCol=2,
            dRow=10.0, dCol=10.0,
            zTop=50.0,
            periodSteps=5,
            strt=25.0,
            hk=10.0,
            chd_west=None, chd_east=None,
            particleCoords=[5.0, 5.0, 25.0],
        )
        workdir = str(tmp_path / 'mf6prt_run2')
        os.makedirs(workdir, exist_ok=True)
        mock_df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0], 'time': [0.0]})

        extra = [
            (patch('flopyarcade.mf6_prt_runner.exists', new=MagicMock(return_value=True)), None),
            (patch('xmipy.XmiWrapper', new=MagicMock(side_effect=Exception("BMI failed"))), None),
            (patch.object(inst, '_mf6prt_parse_track_csv', new=MagicMock(return_value=mock_df)), None),
        ]
        specs = self._patch_all_mf6(extra)
        stack, mocks = apply_patches(specs)
        try:
            mock_sim = mocks['sim_cls'].return_value
            mock_sim.run_simulation.return_value = True
            result = inst._mf6prt_build_and_run(workdir)
            mock_sim.run_simulation.assert_called_once()
        finally:
            stack.close()

    def test_build_and_run_sim_failure(self, tmp_path):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = self._make_instance(
            MODELNAME='test_model',
            nLay=1, nRow=2, nCol=2,
            dRow=10.0, dCol=10.0,
            zTop=50.0,
            periodSteps=5,
            strt=25.0,
            particleCoords=[5.0, 5.0, 25.0],
        )
        workdir = str(tmp_path / 'mf6prt_fail')
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, 'mfsim.lst'), 'w') as f:
            f.write("Line 1: ERROR: something went wrong\n")

        extra = [
            (patch('flopyarcade.mf6_prt_runner.exists', new=MagicMock(return_value=True)), None),
            (patch('xmipy.XmiWrapper', new=MagicMock(side_effect=Exception("BMI fail"))), None),
        ]
        specs = self._patch_all_mf6(extra)
        stack, mocks = apply_patches(specs)
        try:
            mock_sim = mocks['sim_cls'].return_value
            mock_sim.run_simulation.return_value = False
            with pytest.raises(RuntimeError, match=r"MF6\+PRT failed"):
                inst._mf6prt_build_and_run(workdir)
        finally:
            stack.close()

    def test_build_and_run_track_csv_not_found(self, tmp_path):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = self._make_instance(
            MODELNAME='test_model',
            nLay=1, nRow=2, nCol=2,
            dRow=10.0, dCol=10.0,
            zTop=50.0,
            periodSteps=5,
            strt=25.0,
            particleCoords=[5.0, 5.0, 25.0],
        )
        workdir = str(tmp_path / 'mf6prt_notrack')
        os.makedirs(workdir, exist_ok=True)

        extra = [
            (patch('flopyarcade.mf6_prt_runner.exists', new=MagicMock(return_value=False)), None),
            (patch('xmipy.XmiWrapper', new=MagicMock()), None),
            (patch('glob.glob', new=MagicMock(return_value=[])), None),
        ]
        specs = self._patch_all_mf6(extra)
        stack, mocks = apply_patches(specs)
        try:
            with pytest.raises(RuntimeError, match="Track CSV not found"):
                inst._mf6prt_build_and_run(workdir)
        finally:
            stack.close()

    def test_parse_track_csv(self, tmp_path):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = object.__new__(MF6PRTMixin)
        csv_file = str(tmp_path / 'track.csv')
        with open(csv_file, 'w') as f:
            f.write('x,y,z,t\n')
            f.write('1.0,2.0,3.0,0.0\n')
            f.write('4.0,5.0,6.0,1.0\n')
        df = inst._mf6prt_parse_track_csv(csv_file)
        assert len(df) == 2
        assert list(df.columns) == ['x', 'y', 'z', 'time']

    def test_parse_track_csv_empty(self, tmp_path):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = object.__new__(MF6PRTMixin)
        csv_file = str(tmp_path / 'empty.csv')
        with open(csv_file, 'w') as f:
            f.write('x,y,z,t\n')
        with pytest.raises(RuntimeError, match="Track CSV is empty"):
            inst._mf6prt_parse_track_csv(csv_file)

    def test_step_run_simulation_mf6prt(self, tmp_path):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = self._make_instance(
            modelpth=str(tmp_path),
            particleCoords=[10.0, 20.0, 30.0],
            trajectories={'x': [], 'y': [], 'z': []},
            _filterByTime=MagicMock(return_value=np.array([True, True, True])),
        )
        mock_df = pd.DataFrame({
            'x': [10.0, 15.0, 20.0],
            'y': [20.0, 25.0, 30.0],
            'z': [30.0, 35.0, 40.0],
            'time': [0.0, 1.0, 2.0],
        })
        with patch.object(inst, '_mf6prt_build_and_run', new=MagicMock(return_value=mock_df)), \
             patch('shutil.rmtree', new=MagicMock()):
            inst._step_run_simulation_mf6prt()
            assert len(inst.trajectories['x']) == 1
            assert len(inst.trajectories['y']) == 1
            assert len(inst.trajectories['z']) == 1
            # Last position in df
            assert inst.particleCoords == [20.0, 30.0, 40.0]

    def test_step_run_simulation_mf6prt_with_filter(self, tmp_path):
        from flopyarcade.mf6_prt_runner import MF6PRTMixin
        inst = self._make_instance(
            modelpth=str(tmp_path),
            particleCoords=[10.0, 20.0, 30.0],
            trajectories={'x': [], 'y': [], 'z': []},
            _filterByTime=MagicMock(return_value=np.array([True, True, False])),
        )
        mock_df = pd.DataFrame({
            'x': [10.0, 15.0, 20.0],
            'y': [20.0, 25.0, 30.0],
            'z': [30.0, 35.0, 40.0],
            'time': [0.0, 1.0, 2.0],
        })
        with patch.object(inst, '_mf6prt_build_and_run', new=MagicMock(return_value=mock_df)), \
             patch('shutil.rmtree', new=MagicMock()):
            inst._step_run_simulation_mf6prt()
            # Last valid row (index 1)
            assert inst.particleCoords == [15.0, 25.0, 35.0]
