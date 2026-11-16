"""Coverage tests for play_againstai, actions, observations, mf6_prt_runner, play3d, modpath_runner."""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import numpy as np
from copy import deepcopy

from flopyarcade.actions import ActionsMixin
from flopyarcade.observations import ObservationsMixin
from flopyarcade.mf6_prt_runner import MF6PRTMixin
from flopyarcade.modpath_runner import ModpathMixin
import flopyarcade.play3d as play3d_mod


def make_instance(cls, **attrs):
    obj = object.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _defaults(**overrides):
    d = {
        "nHelperWells": 2, "delr": 100.0, "delc": 100.0,
        "extentX": 4000.0, "extentY": 4000.0,
        "wellX": 2000.0, "wellY": 2000.0, "wellQ": -0.005, "wellZ": 50.0,
        "wellCoords": [2000.0, 2000.0, 50.0],
        "wellRadius": 500.0, "helperWellRadius": 125.0,
        "minQ": -0.01, "maxQ": 0.01, "minQhelper": -0.01, "maxQhelper": 0.01,
        "well_dxMax": 100.0, "well_dyMax": 100.0, "well_dQMax": 0.001,
        "actionRange": 2.5, "actionRangeQ": 100.0,
        "actionValueX": 2000.0, "actionValueY": 2000.0,
        "actionValue": 5.0, "actionValueNorth": 55.0, "actionValueSouth": 57.0,
        "headSpecNorth": 58.0, "headSpecSouth": 57.0,
        "dCol": 100.0, "dRow": 100.0, "maxCoordChange": 500.0,
        "ENVTYPE": "4s-d", "actionType": "discrete",
        "actionsDict": {}, "helperWells": {}, "state": {},
        "actionSpaceIndividual": ["up", "keep", "down", "left", "right"],
        "actionSpace": [], "nRow": 40, "nCol": 40, "nLay": 1,
        "zTop": 50.0, "zBot": 0.0, "minX": 0.0, "minY": 0.0,
        "particleCoords": [100.0, 2000.0, 50.0],
        "wellCellLayer": 1, "wellCellRow": 20, "wellCellColumn": 20,
        "MODELNAME": "test_model", "modelpth": "/tmp/test_model",
        "strt": 55.0, "periodSteps": 1,
        "trajectories": {"x": [], "y": [], "z": []},
        "timeStep": 0, "dVer": 50.0, "hk": 10.0,
        "headSpecWest": 60.0, "headSpecEast": 56.0,
        "particleX": 100.0, "particleY": 2000.0, "particleZ": 50.0,
        "particleSpawnBufferY": 20.0, "done": False,
        "exe_mp": "/usr/bin/modpath",
        "heads": np.ones((40, 40)) * 55.0,
        "headsSampledField": np.ones((2, 2)) * 55.0,
        "OBSPREP": "perceptron",
        "minH": 50.0, "maxH": 65.0,
        "observations": {}, "observationsNormalized": {}, "observationsNormalizedHeads": {},
        "observationsVector": [], "observationsVectorNormalized": [],
        "observationsVectorNormalizedHeads": [],
        "observationsSampledField": np.ones((2, 2)),
        "diffQ": 0.005,
    }
    d.update(overrides)
    return d


# === play_againstai.py ===
class TestPlayAgainstAI:

    @pytest.mark.parametrize("val,expected", [
        (True, True), (False, False),
        ("true", True), ("t", True), ("1", True),
        ("yes", True), ("y", True),
        ("false", False), ("f", False), ("0", False),
        ("no", False), ("n", False),
        ("TRUE", True), ("FALSE", False),
    ])
    def test_str_to_bool_valid(self, val, expected):
        from flopyarcade.play_againstai import str_to_bool
        result = str_to_bool(val)
        assert result is expected

    def test_str_to_bool_invalid(self):
        from flopyarcade.play_againstai import str_to_bool
        with pytest.raises(ValueError, match="not a valid boolean value"):
            str_to_bool("maybe")

    def test_env_creator(self):
        import flopyarcade.play_againstai as pa
        with patch.object(pa, "FloPyEnv") as MockEnv:
            MockEnv.return_value = MagicMock()
            env = pa.env_creator({})
            assert env is MockEnv.return_value

    def test_config_structure(self):
        from flopyarcade.play_againstai import config, config_model, config_stopCriteria
        assert config["seed"] == 1
        assert config["num_workers"] >= 0  # depends on args.cpus at import time
        assert config["num_gpus"] == 0
        # last key 'env' in dict wins: it is 'my_env'
        assert config["env"] == "my_env"
        assert config_model["fcnet_hiddens"] == [512, 1024, 512]
        assert config_model["fcnet_activation"] == "relu"
        assert config_stopCriteria["training_iteration"] > 0
        assert config_stopCriteria["timesteps_total"] > 0

    def test_env_creator_envtype(self):
        import flopyarcade.play_againstai as pa
        orig = pa.ENVTYPE
        try:
            pa.ENVTYPE = "6r-c"
            with patch.object(pa, "FloPyEnv") as MockEnv:
                MockEnv.return_value = MagicMock()
                pa.env_creator({})
                call_kwargs = MockEnv.call_args[1]
                assert call_kwargs["env_config"]["ENVTYPE"] == "6r-c"
        finally:
            pa.ENVTYPE = orig


# === actions.py ===
class TestActionsMixin:

    def _m(self, **kw):
        return make_instance(ActionsMixin, **_defaults(**kw))

    def test_getActionType_discrete(self):
        e = self._m()
        assert e.getActionType("1s-d") == "discrete"

    def test_getActionType_continuous(self):
        e = self._m()
        assert e.getActionType("1s-c") == "continuous"

    def test_getActionType_invalid(self):
        e = self._m()
        with pytest.raises(ValueError, match="Unknown environment type"):
            e.getActionType("invalid")

    def test_get_action_value_Q(self):
        e = self._m()
        e.actionsDict = {"well0Q": -0.003}
        assert e.get_action_value(0, "Q") == -0.003

    def test_get_action_value_dx(self):
        e = self._m()
        e.actionsDict = {"well0dx": 0.5}
        assert e.get_action_value(0, "dx") == 50.0

    def test_get_action_value_dy(self):
        e = self._m()
        e.actionsDict = {"well0dy": 0.3}
        assert e.get_action_value(0, "dy") == 30.0

    def test_update_well_Q(self):
        e = self._m()
        e.well = np.zeros((3, 40))
        e.collect_Qs = MagicMock(return_value=np.array([-0.005, -0.003, -0.002]))
        e.update_well_Q()
        assert e.well[0][0] == -0.005
        assert e.well[1][0] == -0.003
        assert e.well[2][0] == -0.002

    def test_update_main_well_random(self):
        e = self._m(ENVTYPE="1r-d")
        e.wellX, e.wellY, e.wellZ = 2000.0, 2000.0, 50.0
        with patch("flopyarcade.actions.uniform", side_effect=[0.5, -0.3]):
            e._update_main_well_location()
        assert e.wellX == 2250.0
        assert e.wellY == 1850.0

    def test_update_main_well_bounds(self):
        e = self._m(ENVTYPE="1r-d", extentX=1000.0, extentY=1000.0)
        e.wellX, e.wellY, e.wellZ = 900.0, 900.0, 50.0
        with patch("flopyarcade.actions.uniform", side_effect=[1.0, 1.0]):
            e._update_main_well_location()
        assert e.wellX == 900.0
        assert e.wellY == 900.0

    def test_update_main_well_action(self):
        e = self._m(ENVTYPE="3s-d")
        e.actionValueX, e.actionValueY, e.wellZ = 1500.0, 2500.0, 40.0
        e._update_main_well_location()
        assert e.wellX == 1500.0
        assert e.wellY == 2500.0

    def test_resolve_overlap_empty(self):
        e = self._m()
        e.helperWells = {}
        e._resolve_well_overlap()

    def test_resolve_overlap_no_overlap(self):
        e = self._m(nHelperWells=2)
        e.helperWells = {"wellX1": 1000.0, "wellY1": 1000.0,
                         "wellX2": 3000.0, "wellY2": 3000.0,
                         "wellZ1": 50.0, "wellZ2": 50.0}
        e.wellX, e.wellY = 2000.0, 2000.0
        e._resolve_well_overlap()
        assert e.helperWells["wellX1"] == 1000.0

    def test_update_helper_wells_location(self):
        e = self._m(ENVTYPE="4s-d", nHelperWells=2)
        e.helperWells = {
            "actionValueX1": 1500.0, "actionValueY1": 2500.0, "wellZ1": 50.0,
            "actionValueX2": 1600.0, "actionValueY2": 2600.0, "wellZ2": 50.0,
        }
        e.wellX, e.wellY = 2000.0, 2000.0
        e._resolve_well_overlap = MagicMock()  # prevent re-sampling
        e._update_helper_wells_location()
        assert e.helperWells["wellX1"] == 1500.0
        assert e.helperWells["wellY1"] == 2500.0
        assert e.helperWells["wellCoords1"] == [1500.0, 2500.0, 50.0]

    def test_update_helper_wells_location_wrong_env(self):
        e = self._m(ENVTYPE="1s-d", nHelperWells=2)
        e.helperWells = {}
        e._update_helper_wells_location()

    def test_update_helper_wells_Q(self):
        e = self._m(ENVTYPE="5s-d", nHelperWells=2)
        e.helperWells = {"actionValueQ1": -0.003, "actionValueQ2": -0.005}
        e._update_helper_wells_Q()
        assert e.helperWells["wellQ1"] == -0.003
        assert e.helperWells["wellQ2"] == -0.005

    def test_update_helper_wells_Q_wrong_env(self):
        e = self._m(ENVTYPE="4s-d", nHelperWells=2)
        e.helperWells = {}
        e._update_helper_wells_Q()

    def test_changeActionDict(self):
        e = self._m(nHelperWells=2)
        action = [0.1] * 12
        d = {}
        result = e.changeActionDict(d, action)
        assert "well0actiondxRight" in result
        assert "well1actiondQDown" in result
        assert result["well0actiondxRight"] == 0.1

    def test_changeActionDict_bad_length(self):
        e = self._m(nHelperWells=2)
        with pytest.raises(ValueError, match="does not match expected"):
            e.changeActionDict({}, [1.0, 2.0])

    def test_getActionSpaceSize(self):
        e = self._m()
        assert e.getActionSpaceSize({"a": 1, "b": 2}) == 2

    def test_update_well_x_clamp_high(self):
        e = self._m()
        d = {"well0x": 3900.0, "well0actiondxRight": 1.0, "well0actiondxLeft": 0.0}
        e._update_well_x(d, 0)
        assert d["well0x"] <= e.extentX - 1.1 * e.delr

    def test_update_well_x_clamp_low(self):
        e = self._m()
        d = {"well0x": 50.0, "well0actiondxRight": 0.0, "well0actiondxLeft": 1.0}
        e._update_well_x(d, 0)
        assert d["well0x"] >= 0.0 + 1.1 * e.delr

    def test_update_well_y_clamp_low(self):
        e = self._m()
        d = {"well0y": 10.0, "well0actiondyUp": 0.0, "well0actiondyDown": 1.0}
        e._update_well_y(d, 0)
        assert d["well0y"] >= 0.0 + 0.1 * e.delc

    def test_update_well_y_clamp_high(self):
        e = self._m()
        d = {"well0y": 3990.0, "well0actiondyUp": 1.0, "well0actiondyDown": 0.0}
        e._update_well_y(d, 0)
        assert d["well0y"] <= e.extentY - 0.1 * e.delc

    def test_update_well_Q_clamp_high(self):
        e = self._m()
        d = {"well0Q": 0.009, "well0actiondQUp": 1.0, "well0actiondQDown": 0.0}
        e._update_well_Q(d, 0)
        assert d["well0Q"] <= e.maxQ

    def test_update_well_Q_clamp_low(self):
        e = self._m()
        d = {"well0Q": -0.009, "well0actiondQUp": 0.0, "well0actiondQDown": 1.0}
        e._update_well_Q(d, 0)
        assert d["well0Q"] >= e.minQ

    def test_update_action_1d(self):
        e = self._m()
        e._get_delta_1d = lambda action: 1.0 if action == "up" else -1.0 if action == "down" else 0.0
        e.actionValueNorth = 55.0
        e.actionValueSouth = 57.0
        e.update_action_1d("up")
        assert e.actionValueNorth == 56.0
        assert e.actionValueSouth == 58.0

    def test_update_action_1d_keep(self):
        e = self._m()
        e._get_delta_1d = lambda action: 1.0 if action == "up" else -1.0 if action == "down" else 0.0
        e.actionValueNorth = 55.0
        e.actionValueSouth = 57.0
        e.update_action_1d("keep")
        assert e.actionValueNorth == 55.0

    def test_update_action_1c(self):
        e = self._m()
        e.actionRange = 2.5
        e.actionValueNorth = 55.0
        e.actionValueSouth = 57.0
        e.update_action_1c([1.0, 0.0])
        assert e.actionValueNorth == 57.5
        assert e.actionValueSouth == 59.5

    def test_update_action_2d(self):
        e = self._m()
        e._get_delta_2d = lambda action: 2.0 if action == "up" else 0.0
        e.actionValue = 5.0
        e.update_action_2d("up")
        assert e.actionValue == 7.0

    def test_update_action_2c(self):
        e = self._m()
        e.actionRange = 2.5
        e.actionValue = 5.0
        e.update_action_2c([1.0, 0.0])
        assert e.actionValue == 7.5

    def test_update_action_3d_up(self):
        e = self._m()
        e.wellX, e.wellY = 2000.0, 2000.0
        e.extentX, e.extentY = 4000.0, 4000.0
        e.dCol, e.dRow = 100.0, 100.0
        e.actionRange = 2.5
        e.actionValueY = 2000.0
        e.update_action_3d("up")
        assert e.actionValueY == 1997.5

    def test_update_action_3d_right(self):
        e = self._m()
        e.wellX, e.wellY = 2000.0, 2000.0
        e.extentX, e.extentY = 4000.0, 4000.0
        e.dCol, e.dRow = 100.0, 100.0
        e.actionRange = 2.5
        e.actionValueX = 2000.0
        e.update_action_3d("right")
        assert e.actionValueX == 2002.5

    def test_update_action_3c(self):
        e = self._m()
        e._calc_delta = lambda up, down: (up - down) * 2.0
        e.wellX, e.wellY = 2000.0, 2000.0
        e.extentX, e.extentY = 4000.0, 4000.0
        e.dCol, e.dRow = 100.0, 100.0
        e.actionValueX, e.actionValueY = 2000.0, 2000.0
        e.update_action_3c([1, 0, 0, 0])
        assert e.actionValueY == 2002.0

    def test_move_helper_well_x(self):
        e = self._m()
        e.helperWells = {"wellX1": 2000.0, "actionValueX1": 2000.0}
        e.dCol, e.extentX = 100.0, 4000.0
        e._move_helper_well_x("1", 10.0)
        assert e.helperWells["actionValueX1"] == 2010.0

    def test_move_helper_well_y(self):
        e = self._m()
        e.helperWells = {"wellY1": 2000.0, "actionValueY1": 2000.0}
        e.dRow, e.extentY = 100.0, 4000.0
        e._move_helper_well_y("1", -10.0)
        assert e.helperWells["actionValueY1"] == 1990.0

    def test_change_helper_well_q(self):
        e = self._m()
        e.helperWells = {"wellQ1": -0.005, "actionValueQ1": -0.005}
        e.minQhelper, e.maxQhelper = -0.01, 0.01
        e._change_helper_well_q("1", 0.002)
        assert e.helperWells["actionValueQ1"] == -0.003

    def test_setActionValue_single(self):
        e = self._m(ENVTYPE="1s-d")
        e._ACTION_SINGLE = {"1s-d": "update_action_1d"}
        e._MULTI_C = set()
        e.update_action_1d = MagicMock()
        e.setActionValue("up")
        e.update_action_1d.assert_called_once_with("up")

    def test_setActionValue_multi_d(self):
        e = self._m(ENVTYPE="6s-d")
        e._ACTION_SINGLE = {}
        e._MULTI_C = set()
        e.update_action_multi_wells_d = MagicMock()
        e.setActionValue([1, 0, 1, 0, 1, 0])
        e.update_action_multi_wells_d.assert_called_once_with([1, 0, 1, 0, 1, 0])

    def test_setActionValue_multi_c(self):
        e = self._m(ENVTYPE="6c-d")
        e._ACTION_SINGLE = {}
        e._MULTI_C = {"6c-d"}
        e.update_action_multi_wells_c_position = MagicMock()
        e.update_action_multi_wells_c_rate = MagicMock()
        e.setActionValue([1, 0, 1, 0, 1, 0])
        e.update_action_multi_wells_c_position.assert_called_once()
        e.update_action_multi_wells_c_rate.assert_called_once()


# === observations.py ===
class TestObservationsMixin:

    def _m(self, **kw):
        return make_instance(ObservationsMixin, **_defaults(**kw))

    def test_normalize_heads(self):
        e = self._m()
        result = e._normalize_heads([50, 57.5, 65], 50, 65)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_normalize_coords(self):
        e = self._m()
        result = e._normalize_coords([0, 2000, 4000], 0, 4000)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_normalize_wellQ(self):
        e = self._m()
        result = e._normalize_wellQ([-0.005, -0.01], -0.01)
        np.testing.assert_allclose(result, [0.5, 1.0])

    def test_normalize(self):
        e = self._m()
        assert e._normalize(5, 0, 10) == 0.5

    def test_normalize_well(self):
        e = self._m()
        result = e._normalize_well(-0.005, 2000, 2000, 50)
        assert len(result) == 4

    def test_normalize_helper_wells(self):
        e = self._m(nHelperWells=2)
        e.helperWells = {
            "wellQ1": -0.005, "wellX1": 2000.0, "wellY1": 2000.0, "wellZ1": 50.0,
            "wellQ2": -0.01, "wellX2": 4000.0, "wellY2": 4000.0, "wellZ2": 50.0,
        }
        result = e._normalize_helper_wells()
        assert len(result) == 8

    def test_get_normalized_stresses_1s(self):
        e = self._m(ENVTYPE="1s-d")
        e.actionValueSouth, e.actionValueNorth = 57.0, 55.0
        e.wellQ, e.wellX, e.wellY, e.wellZ = -0.005, 2000, 2000, 50
        e._normalize_basic_stresses = lambda vals: [0.5, 0.3]
        e._normalize_well = lambda Q, X, Y, Z: [0.5, 0.5, 0.5, 0.5]
        result = e.get_normalized_stresses()
        assert len(result) == 6

    def test_get_normalized_stresses_invalid(self):
        e = self._m(ENVTYPE="invalid")
        assert e.get_normalized_stresses() == []

    def test_example_feature_extractor(self):
        e = self._m()
        obs = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = e.example_feature_extractor(obs)
        assert result.shape == (4,)

    def test_example_feature_extractor_zero(self):
        e = self._m()
        obs = np.zeros((2, 2))
        result = e.example_feature_extractor(obs)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_initializeState(self):
        e = self._m()
        heads = np.ones((10, 10))
        e.initializeState({"heads": heads})
        np.testing.assert_array_equal(e.headsPrev, heads)

    def test_initializeState_missing(self):
        e = self._m()
        with pytest.raises(AssertionError):
            e.initializeState({})

    def test_set_heads_observations_perceptron_1s(self):
        e = self._m(ENVTYPE="1s-d", OBSPREP="perceptron")
        e.actionValueNorth = 55.0
        e.actionValueSouth = 57.0
        e.observations = {}
        e.observations["headsSampledField"] = np.ones((2, 2))
        e._set_heads_observations()
        assert len(e.observations["heads"]) == 6

    def test_set_heads_observations_perceptron_4s(self):
        e = self._m(ENVTYPE="4s-d", OBSPREP="perceptron")
        e.headSpecNorth = 58.0
        e.headSpecSouth = 57.0
        e.observations = {}
        e.observations["headsSampledField"] = np.ones((2, 2))
        e._set_heads_observations()
        assert len(e.observations["heads"]) == 6

    def test_set_well_observations_single(self):
        e = self._m(ENVTYPE="1s-d")
        e.wellQ = -0.005
        e.wellCoords = [2000, 2000, 50]
        e.observations = {}
        e._set_well_observations()
        assert e.observations["wellQ"] == -0.005
        assert e.observations["wellCoords"] == [2000, 2000, 50]

    def test_set_well_observations_multi(self):
        e = self._m(ENVTYPE="4s-d", nHelperWells=2)
        e.wellQ = -0.005
        e.wellCoords = [2000, 2000, 50]
        e.helperWells = {
            "wellQ1": -0.003, "wellCoords1": [1000, 1000, 50],
            "wellQ2": -0.004, "wellCoords2": [2000, 2000, 50],
        }
        e.observations = {}
        e._set_well_observations()
        assert e.observations["wellQ1"] == -0.003
        assert e.observations["wellCoords2"] == [2000, 2000, 50]

    def test_normalize_observations_basic(self):
        e = self._m(ENVTYPE="1s-d")
        e.particleCoords = [100, 2000, 50]
        e.minX = 0.0
        e.extentX = 4000.0
        e.minH, e.maxH = 50.0, 65.0
        e.wellQ = -0.005
        e.minQ = -0.01
        e.wellCoords = [2000, 2000, 50]
        e.observations = {"heads": [55.0, 57.0]}
        e.heads = np.ones((2, 2)) * 55.0
        e.observationsNormalized = {}
        e.observationsNormalizedHeads = {}
        e._normalize_observations()
        assert "particleCoords" in e.observationsNormalized
        assert "heads" in e.observationsNormalized
        assert "wellQ" in e.observationsNormalized

    def test_build_observations_wells(self):
        e = self._m(nHelperWells=2)
        e.actionsDict = {
            "well0x": 1000.0, "well0y": 2000.0, "well0Q": -0.005,
            "well1x": 3000.0, "well1y": 1000.0, "well1Q": -0.003,
        }
        e.normalize = lambda v, lo, hi: (v - lo) / (hi - lo) if hi != lo else 0.0
        e.extentX, e.extentY = 4000.0, 4000.0
        e.minQ, e.diffQ = -0.01, 0.005
        result = e.build_observations_wells()
        assert len(result) == 6


# === mf6_prt_runner.py ===
class TestMF6PRTMixin:

    def _m(self, **kw):
        return make_instance(MF6PRTMixin, **_defaults(**kw))

    def test_cell_from_coords(self):
        e = self._m()
        e.dCol, e.dRow, e.nRow, e.nCol = 100.0, 100.0, 40, 40
        row, col = e._mf6prt_cell_from_coords(2000, 3000)
        assert col == 20
        assert row == 9

    def test_cell_from_coords_clamp(self):
        e = self._m()
        e.dCol, e.dRow, e.nRow, e.nCol = 100.0, 100.0, 40, 40
        row, col = e._mf6prt_cell_from_coords(-10, -10)
        assert row == 39
        assert col == 0

    def test_get_wells_single(self):
        e = self._m()
        e.wellCellLayer, e.wellQ = 1, -0.005
        e.wellCellRow, e.wellCellColumn = 20, 20
        e.nHelperWells = 0
        wells = e._mf6prt_get_wells()
        assert len(wells) == 1
        assert wells[0][0] == (0, 19, 19)

    def test_get_wells_multi(self):
        e = self._m(nHelperWells=2)
        e.wellCellLayer, e.wellQ = 1, -0.005
        e.wellCellRow, e.wellCellColumn = 20, 20
        e.helperWells = {
            "l1": 1, "r1": 10, "c1": 10, "wellQ1": -0.003,
            "l2": 1, "r2": 20, "c2": 20, "wellQ2": -0.004,
        }
        wells = e._mf6prt_get_wells()
        assert len(wells) == 3

    def test_get_hydraulics(self):
        e = self._m()
        e.hk = 10.0
        e.headSpecWest, e.headSpecEast = 60.0, 56.0
        hk, vka, ss, sy, cw, ce, rch = e._mf6prt_get_hydraulics()
        assert hk == 10.0
        assert cw == 60.0
        assert ce == 56.0

    def test_get_hydraulics_defaults(self):
        e = self._m()
        e.hk = None
        e.headSpecWest = None
        e.headSpecEast = None
        hk, vka, ss, sy, cw, ce, rch = e._mf6prt_get_hydraulics()
        assert hk == 10.0
        assert cw is None

    def test_parse_track_csv(self):
        e = self._m()
        import tempfile, csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["x", "y", "z", "t"])
            writer.writeheader()
            writer.writerow({"x": "1.0", "y": "2.0", "z": "3.0", "t": "0.0"})
            writer.writerow({"x": "1.5", "y": "2.5", "z": "3.5", "t": "1.0"})
            csv_path = f.name
        try:
            df = e._mf6prt_parse_track_csv(csv_path)
            assert len(df) == 2
            assert "x" in df.columns
            assert "y" in df.columns
        finally:
            os.unlink(csv_path)

    def test_parse_track_csv_empty(self):
        e = self._m()
        import tempfile, csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["x", "y", "z", "t"])
            writer.writeheader()
            csv_path = f.name
        try:
            with pytest.raises(Exception):
                e._mf6prt_parse_track_csv(csv_path)
        finally:
            os.unlink(csv_path)

    def test_step_run_simulation_mf6prt(self):
        import pandas as pd
        e = self._m()
        e.modelpth = "/tmp/test_model"
        mock_df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]})
        e._mf6prt_build_and_run = MagicMock(return_value=mock_df)
        # _filterByTime must return a boolean Series
        bool_series = pd.Series([True, True])
        e._filterByTime = MagicMock(return_value=bool_series)
        e.trajectories = {"x": [], "y": [], "z": []}
        e._step_run_simulation_mf6prt()
        assert len(e.trajectories["x"]) == 1
        assert len(e.trajectories["y"]) == 1
        assert len(e.trajectories["z"]) == 1
        assert e.particleCoords == [2.0, 4.0, 6.0]


# === play3d.py ===
class TestPlay3D:

    def test_mock_env_init(self):
        env = play3d_mod.MockEnv(nrow=10, ncol=10)
        assert env.nrow == 10
        assert env.ncol == 10
        assert env.heads.shape == (10, 10)

    def test_mock_env_reset(self):
        env = play3d_mod.MockEnv()
        state = env.reset()
        assert "heads" in state
        assert env.done is False

    def test_mock_env_step(self):
        env = play3d_mod.MockEnv(nrow=10, ncol=10)
        env.reset()
        result = env.step()
        assert len(result) == 5
        coords, reward, done, trunc, info = result
        assert done is False

    def test_mock_env_step_done(self):
        env = play3d_mod.MockEnv(nrow=10, ncol=10)
        env.reset()
        env.particleX = env.extentX * 0.94
        _, _, done, _, _ = env.step()
        assert done is True

    def test_mock_env_distance(self):
        env = play3d_mod.MockEnv()
        d = env.distance_to_well([0, 0], [3, 4])
        assert d == 5.0

    def test_main_demo(self):
        with patch("sys.argv", ["play3d.py", "--mode", "demo", "--nsteps", "2", "--nrow", "5", "--ncol", "5"]):
            with patch("flopyarcade.play3d.run_demo") as rd:
                rd.return_value = "out.gif"
                play3d_mod.main()
                rd.assert_called_once()

    def test_main_live(self):
        with patch("sys.argv", ["play3d.py", "--mode", "live", "--envtype", "1s-d"]):
            with patch("flopyarcade.play3d.run_live") as rl:
                rl.return_value = "frame.png"
                play3d_mod.main()
                rl.assert_called_once()


# === modpath_runner.py ===
class TestModpathMixin:

    def _m(self, **kw):
        return make_instance(ModpathMixin, **_defaults(**kw))

    def test_initializeParticle(self):
        e = self._m()
        e.extentX, e.extentY, e.dCol, e.zTop = 4000.0, 4000.0, 100.0, 50.0
        e.initializeParticle()
        assert e.particleX == 4000.0 - 110.0
        assert e.particleZ == 50.0
        assert len(e.particleCoords) == 3

    def test_initializeParticle_invalid(self):
        e = self._m()
        e.extentX = -1
        with pytest.raises(AssertionError):
            e.initializeParticle()

    def test_check_particle_within_well(self):
        e = self._m()
        e.particleCoords = [2000, 2000, 50]
        e.wellCoords = [2000, 2000, 50]
        e.wellRadius = 500.0
        e.distance_to_well = MagicMock(return_value=10.0)
        e.update_reward_on_fail = MagicMock()
        e.done = False
        e.check_particle_within_well()
        assert e.done is True
        e.update_reward_on_fail.assert_called_once()

    def test_check_particle_within_well_outside(self):
        e = self._m()
        e.particleCoords = [100, 100, 50]
        e.wellCoords = [2000, 2000, 50]
        e.wellRadius = 500.0
        e.distance_to_well = MagicMock(return_value=2900.0)
        e.update_reward_on_fail = MagicMock()
        e.done = False
        e.check_particle_within_well()
        assert e.done is False

    def test_check_particle_helper_wells(self):
        e = self._m(ENVTYPE="4s-d", nHelperWells=2)
        e.particleCoords = [1000, 1000, 50]
        e.helperWells = {
            "wellCoords1": [1010, 1010, 50],
            "wellCoords2": [3000, 3000, 50],
        }
        e.helperWellRadius = 125.0
        e.distance_to_well = MagicMock(side_effect=[14.1, 2900.0])
        e.update_reward_on_fail = MagicMock()
        e.done = False
        e.check_particle_within_helper_wells()
        assert e.done is True

    def test_check_particle_helper_wells_wrong_env(self):
        e = self._m(ENVTYPE="1s-d")
        e.done = False
        e.check_particle_within_helper_wells()
        assert e.done is False

    def test_check_particle_state(self):
        e = self._m(ENVTYPE="1s-d")
        e.check_particle_within_well = MagicMock()
        e.check_particle_within_helper_wells = MagicMock()
        e.check_boundaries = MagicMock()
        e.check_max_steps = MagicMock()
        e.check_particle_state()
        e.check_particle_within_well.assert_called_once()
        e.check_particle_within_helper_wells.assert_called_once()
        e.check_boundaries.assert_called_once()
        e.check_max_steps.assert_called_once()

    def test_cleanup_temp_dir(self):
        e = self._m()
        e.tempDir = MagicMock()
        e.cleanup_temp_dir()
        e.tempDir.cleanup.assert_called_once()

    def test_transform_particle_coords(self):
        e = self._m()
        e.extentX = 4000.0
        e.particleCoords = [100.0, 2000.0, 50.0]
        e._transform_particle_coords()
        assert e.particleCoords[0] == 3900.0

    def test_runMODPATH_first_step(self):
        e = self._m()
        e.timeStep = 1
        e._transform_particle_coords = MagicMock()
        e._initialize_modpath_objects = MagicMock()
        e._write_modpath_input = MagicMock()
        e._modify_mpsim_file = MagicMock()
        e._write_particle_location = MagicMock()
        e.mp = MagicMock()
        e.runMODPATH()
        e._initialize_modpath_objects.assert_called_once()
        e.mp.run_model.assert_called_once_with(silent=True)

    def test_runMODPATH_not_first_step(self):
        e = self._m()
        e.timeStep = 5
        e._transform_particle_coords = MagicMock()
        e._initialize_modpath_objects = MagicMock()
        e._write_modpath_input = MagicMock()
        e._modify_mpsim_file = MagicMock()
        e._write_particle_location = MagicMock()
        e.mp = MagicMock()
        e.runMODPATH()
        e._initialize_modpath_objects.assert_not_called()
        e.mp.run_model.assert_called_once_with(silent=True)

    def test_insert_custom_particle_settings(self):
        e = self._m()
        e.MODELNAME = "test_model"
        lines = ["line1\n", "rch file\n", "line3\n"]
        result = e._insert_custom_particle_settings(lines)
        assert any("test_model.mploc" in l for l in result)

    def test_evaluateParticleTracking(self):
        e = self._m()
        mock_data = MagicMock()
        e._loadParticleData = MagicMock(return_value=mock_data)
        e._filterByTime = MagicMock(return_value=None)
        e._appendTrajectories = MagicMock()
        e._updateParticleCoords = MagicMock()
        e.evaluateParticleTracking()
        e._loadParticleData.assert_called_once()
        e._appendTrajectories.assert_called_once()
        e._updateParticleCoords.assert_called_once()

    def test_get_particle_coords_first(self):
        e = self._m()
        e.timeStep = 0
        e.minX = 0.0
        e.particleCoords = [100, 2000, 50]
        x, y = e._get_particle_coords()
        assert x == 0.0
        assert y == 2000

    def test_get_particle_coords_later(self):
        e = self._m()
        e.timeStep = 5
        e.trajectories = {"x": [[1, 2]], "y": [[3, 4]]}
        x, y = e._get_particle_coords()
        assert x == 2
        assert y == 4
