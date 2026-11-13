"""
C7: Action space initialisation tests.

Verifies that:
1. Discrete multi-well envs do NOT materialize infeasible product lists (OOM fix).
2. The action space size is consistent with the step() interface.
3. The _setup_action_space method produces valid gym spaces for all env types.
"""
import pytest
import numpy as np


class TestDiscreteMultiWellNoOOM:
    """C7 Bug 1: product() explosion must not occur."""

    @pytest.mark.parametrize("envtype", ['4s-d', '5s-d', '6s-d', '4r-d', '5r-d', '6r-d'])
    def test_action_space_not_infeasible(self, envtype):
        """The actionSpace list must not be infeasibly large (was 5^20 or 8^20)."""
        from flopyarcade import FloPyEnv

        env = FloPyEnv(
            ENVTYPE=envtype,
            MODELNAME=f"c7_{envtype}",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=42,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        # The old code would OOM here. The fix caps the list at product(..., repeat=min(2, nHelperWells))
        # So max list size is 5^2=25 or 8^2=64, well within memory.
        assert len(env.actionSpace) < 1000, (
            f"{envtype}: actionSpace has {len(env.actionSpace)} entries, expected < 1000"
        )
        # Per-well discrete count should be stored
        assert hasattr(env, '_multiwell_discrete_per_well')
        assert env._multiwell_discrete_per_well >= 3
        env.close()

    def test_action_space_size_is_finite(self):
        """actionSpaceSize must be finite and reasonable (was infeasible 5^20 before C7 fix)."""
        from flopyarcade import FloPyEnv

        env = FloPyEnv(
            ENVTYPE='4s-d',
            MODELNAME="c7_size",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=42,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        # Before C7 fix: 5^20 = 9.5e13 (OOM). After fix: capped to 5^2 = 25
        assert env.actionSpaceSize < 1000, (
            f"actionSpaceSize={env.actionSpaceSize} is too large (C7 OOM not fixed)"
        )
        assert env.actionSpaceSize > 0
        env.close()


class TestActionSpaceConsistency:
    """C7: Verify action space dimensions match what step() expects."""

    @pytest.mark.parametrize("envtype", ['1s-d', '2s-d', '3s-d', '1s-c', '2s-c', '3s-c'])
    def test_single_well_action_space(self, envtype):
        """Single-well envs should have a consistent, small action space."""
        from flopyarcade import FloPyEnv

        env = FloPyEnv(
            ENVTYPE=envtype,
            MODELNAME=f"c7_single_{envtype}",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=42,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        # actionSpaceSize should be a small number (3-6)
        assert 2 <= env.actionSpaceSize <= 10, (
            f"{envtype}: actionSpaceSize={env.actionSpaceSize} seems wrong"
        )
        env.close()

    @pytest.mark.parametrize("envtype", ['4s-c', '5s-c', '6s-c'])
    def test_continuous_multiwell_action_space(self, envtype):
        """Continuous multi-well envs should have actionSpaceSize = nHelperWells * per_well_actions."""
        from flopyarcade import FloPyEnv

        env = FloPyEnv(
            ENVTYPE=envtype,
            MODELNAME=f"c7_cont_{envtype}",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=42,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        # For continuous multi-well, actionSpaceSize = len(actionSpace) = nHelperWells * len(actionSpaceIndividual)
        expected = env.nHelperWells * len(env.actionSpaceIndividual)
        assert env.actionSpaceSize == expected, (
            f"{envtype}: actionSpaceSize={env.actionSpaceSize}, expected {expected}"
        )
        env.close()


class TestInitOrder:
    """C7 Bug 3: initializeAction() must be called after well placement for envs 3-6."""

    @pytest.mark.parametrize("envtype", ['3s-d', '4s-c', '6s-c'])
    def test_action_values_match_well_positions(self, envtype):
        """After init, actionValueX/Y should match wellX/Y for envs that copy them."""
        from flopyarcade import FloPyEnv

        env = FloPyEnv(
            ENVTYPE=envtype,
            MODELNAME=f"c7_order_{envtype}",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=42,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        env.reset()

        # For envs 4/6, actionValueX/Y should be set (copied from well positions)
        if envtype in ('4s-c', '6s-c'):
            hw = env.helperWells
            n = env.nHelperWells
            for i in range(n):
                w = str(i + 1)
                if f'actionValueX{w}' in hw:
                    assert hw[f'actionValueX{w}'] == hw[f'wellX{w}'], (
                        f"Well {w}: actionValueX != wellX"
                    )
        env.close()
