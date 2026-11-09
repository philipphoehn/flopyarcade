"""
A10: Protection-zone overlap prevention tests.

Verifies that _resolve_well_overlap() ensures no two helper wells (or a helper
well and the main well) are closer than the minimum separation distance.
"""
import numpy as np
import pytest


class TestWellOverlapPrevention:
    """Test that helper wells maintain minimum separation after init and step."""

    @pytest.mark.parametrize("envtype", ['4s-c', '6s-c'])
    def test_no_overlap_after_init(self, envtype):
        """After environment init, no two helper wells should be within min separation."""
        from flopyarcade import FloPyEnv

        env = FloPyEnv(
            ENVTYPE=envtype,
            MODELNAME=f"a10_test_{envtype}",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=42,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        env.reset()

        hw = env.helperWells
        n = env.nHelperWells
        hw_radius = env.helperWellRadius
        min_sep_helper = 2.0 * hw_radius
        min_sep_main = env.wellRadius + hw_radius

        # Check helper-to-helper
        for i in range(n):
            wi = str(i + 1)
            xi = hw[f'wellX{wi}']
            yi = hw[f'wellY{wi}']
            for j in range(i + 1, n):
                wj = str(j + 1)
                xj = hw[f'wellX{wj}']
                yj = hw[f'wellY{wj}']
                dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                assert dist >= min_sep_helper - 1e-9, (
                    f"Wells {wi} and {wj} overlap: dist={dist:.4f} < min_sep={min_sep_helper:.4f}"
                )

        # Check helper-to-main
        for i in range(n):
            wi = str(i + 1)
            xi = hw[f'wellX{wi}']
            yi = hw[f'wellY{wi}']
            dist = np.sqrt((xi - env.wellX) ** 2 + (yi - env.wellY) ** 2)
            assert dist >= min_sep_main - 1e-9, (
                f"Well {wi} overlaps main well: dist={dist:.4f} < min_sep={min_sep_main:.4f}"
            )

        env.close()

    @pytest.mark.parametrize("envtype", ['4s-c', '6s-c'])
    def test_no_overlap_after_step(self, envtype):
        """After a step with position-changing actions, no overlap should exist."""
        from flopyarcade import FloPyEnv

        env = FloPyEnv(
            ENVTYPE=envtype,
            MODELNAME=f"a10_step_{envtype}",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=42,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        obs, _ = env.reset()

        # Take a few steps with 0.5 (neutral) actions
        n_steps = 3
        for _ in range(n_steps):
            obs, reward, term, trunc, _ = env.step(np.full((env.actionSpaceSize,), 0.5, dtype=np.float32))
            if term or trunc:
                break

        hw = env.helperWells
        n = env.nHelperWells
        hw_radius = env.helperWellRadius
        min_sep_helper = 2.0 * hw_radius

        for i in range(n):
            wi = str(i + 1)
            xi = hw[f'wellX{wi}']
            yi = hw[f'wellY{wi}']
            for j in range(i + 1, n):
                wj = str(j + 1)
                xj = hw[f'wellX{wj}']
                yj = hw[f'wellY{wj}']
                dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                assert dist >= min_sep_helper - 1e-9, (
                    f"After step: Wells {wi} and {wj} overlap: dist={dist:.4f} < min_sep={min_sep_helper:.4f}"
                )

        env.close()

    def test_resolve_well_overlap_no_op_when_no_overlap(self):
        """_resolve_well_overlap should not consume RNG if no overlap exists."""
        from flopyarcade import FloPyEnv

        env = FloPyEnv(
            ENVTYPE='4s-c',
            MODELNAME="a10_noop",
            flagSavePlot=False,
            flagManualControl=False,
            flagRender=False,
            _seed=123,
            nLay=1,
            nRow=10,
            nCol=10,
        )
        env.reset()

        # Record well positions
        hw = env.helperWells
        n = env.nHelperWells
        positions_before = [(hw[f'wellX{i+1}'], hw[f'wellY{i+1}']) for i in range(n)]

        # Record RNG state
        import numpy as np
        rng_state = np.random.get_state()[1].copy()

        # Call resolve - should be no-op (no overlap with well-separated wells)
        env._resolve_well_overlap()

        # Positions should be unchanged
        for i in range(n):
            assert hw[f'wellX{i+1}'] == positions_before[i][0]
            assert hw[f'wellY{i+1}'] == positions_before[i][1]

        # RNG state should be unchanged (no uniform() calls)
        new_rng_state = np.random.get_state()[1]
        np.testing.assert_array_equal(rng_state, new_rng_state)

        env.close()
