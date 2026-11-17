"""Tests for B14: well safety-zone linewidth scaled by pumping rate.

The scaling formula in render.py _draw_well_circle is:
    lw = 1.0 + 4.0 * (abs(quantity) / max(abs(min_quantity), 1e-6))

This test file verifies:
- Baseline (Q=0 -> lw=1.0)
- Exact formula values
- Monotonicity (higher |Q| -> higher lw)
- Symmetry (positive and negative Q give same lw)
- Helper-well parity (same formula applies)
- Edge cases (min_quantity=0, very small values)
"""

import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, "/a0/usr/projects/flopyarcade_roadmapworker2/flopyarcade")

from flopyarcade.render import RenderMixin


def _make_env(**overrides):
    """Build a minimal mock env with the attributes _draw_well_circle needs."""
    e = MagicMock()
    defaults = {
        'wellCoords': (50, 30),
        'wellRadius': 20.0,
        'wellQ': -1.5,
        'minQ': -5.0,
        'extentY': 100.0,
        'ENVTYPE': '1s-d',
        'nHelperWells': 0,
        'helperWells': {},
        'helperWellRadius': 10.0,
        'minQhelper': -3.0,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(e, k, v)

    # Stub the helper methods so we can capture the lw argument
    e._circle_coords = lambda coords: (coords[0], e.extentY - coords[1])
    e._create_outline_circle = MagicMock(return_value=MagicMock())
    e._create_filled_circle = MagicMock(return_value=MagicMock())
    e.ax2 = MagicMock()
    e.renderGetWellColor = lambda Q: 'red' if Q < 0 else ('blue' if Q > 0 else 'gray')
    e.renderGetWellCircleAlpha = lambda Q, maxQ: abs(Q) / max(abs(maxQ), 1e-6) * 0.9
    return e


class TestLWFormula:
    """Verify the lw = 1.0 + 4.0*(|Q|/|minQ|) formula."""

    def test_baseline_zero_q(self):
        """Q=0 -> lw=1.0 (minimum, no scaling)."""
        e = _make_env()
        RenderMixin._draw_well_circle(e, (50, 30), 20.0, 0.0, -5.0, 3)
        call_kwargs = e._create_outline_circle.call_args
        assert call_kwargs[1]['lw'] == pytest.approx(1.0)

    def test_exact_value_at_min(self):
        """|Q| == |minQ| -> lw = 1.0 + 4.0 = 5.0."""
        e = _make_env()
        RenderMixin._draw_well_circle(e, (50, 30), 20.0, -5.0, -5.0, 3)
        call_kwargs = e._create_outline_circle.call_args
        assert call_kwargs[1]['lw'] == pytest.approx(5.0)

    def test_exact_value_half_min(self):
        """|Q| = |minQ|/2 -> lw = 1.0 + 2.0 = 3.0."""
        e = _make_env()
        RenderMixin._draw_well_circle(e, (50, 30), 20.0, -2.5, -5.0, 3)
        call_kwargs = e._create_outline_circle.call_args
        assert call_kwargs[1]['lw'] == pytest.approx(3.0)

    def test_exact_value_double_min(self):
        """|Q| = 2*|minQ| -> lw = 1.0 + 8.0 = 9.0."""
        e = _make_env()
        RenderMixin._draw_well_circle(e, (50, 30), 20.0, -10.0, -5.0, 3)
        call_kwargs = e._create_outline_circle.call_args
        assert call_kwargs[1]['lw'] == pytest.approx(9.0)

    def test_positive_q_same_as_negative(self):
        """lw depends on |Q|, so +Q and -Q give identical lw."""
        e1 = _make_env()
        RenderMixin._draw_well_circle(e1, (50, 30), 20.0, 3.0, -5.0, 3)
        lw_pos = e1._create_outline_circle.call_args[1]['lw']

        e2 = _make_env()
        RenderMixin._draw_well_circle(e2, (50, 30), 20.0, -3.0, -5.0, 3)
        lw_neg = e2._create_outline_circle.call_args[1]['lw']

        assert lw_pos == pytest.approx(lw_neg)


class TestMonotonicity:
    """Higher pumping rate -> thicker line."""

    def test_increasing_q_increasing_lw(self):
        """lw must be strictly increasing in |Q|."""
        minQ = -5.0
        q_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
        lws = []
        for q in q_values:
            e = _make_env()
            RenderMixin._draw_well_circle(e, (50, 30), 20.0, q, minQ, 3)
            lws.append(e._create_outline_circle.call_args[1]['lw'])

        for i in range(1, len(lws)):
            assert lws[i] > lws[i - 1], f"lw not increasing at index {i}: {lws}"


class TestEdgeCases:
    """Robustness for degenerate inputs."""

    def test_min_quantity_zero(self):
        """minQ=0 -> denominator clamped to 1e-6; lw is large but finite."""
        e = _make_env()
        RenderMixin._draw_well_circle(e, (50, 30), 20.0, -1.0, 0.0, 3)
        lw = e._create_outline_circle.call_args[1]['lw']
        # lw = 1.0 + 4.0 * (1.0 / 1e-6) = 1.0 + 4_000_000.0
        assert lw == pytest.approx(1.0 + 4.0 * (1.0 / 1e-6))
        assert math.isfinite(lw)

    def test_very_small_q(self):
        """Very small Q -> lw ~ 1.0 (barely above baseline)."""
        e = _make_env()
        RenderMixin._draw_well_circle(e, (50, 30), 20.0, -0.001, -5.0, 3)
        lw = e._create_outline_circle.call_args[1]['lw']
        assert lw == pytest.approx(1.0 + 4.0 * 0.001 / 5.0)
        assert lw < 1.01


class TestHelperWellParity:
    """Helper wells use the same lw scaling formula."""

    def test_helper_well_lw_matches_main(self):
        """For same |Q|/|minQ| ratio, helper and main wells get same lw."""
        # Main well: |Q|/|minQ| = 2.5/5.0 = 0.5 -> lw = 3.0
        e = _make_env()
        RenderMixin._draw_well_circle(e, (50, 30), 20.0, -2.5, -5.0, 3)
        lw_main = e._create_outline_circle.call_args[1]['lw']

        # Helper well: |Q|/|minQ| = 1.5/3.0 = 0.5 -> lw = 3.0
        e2 = _make_env(
            ENVTYPE='4s-d',
            nHelperWells=1,
            helperWells={'wellCoords1': (70, 40), 'wellQ1': -1.5},
            minQhelper=-3.0,
        )
        RenderMixin._draw_well_circle(e2, (70, 40), 10.0, -1.5, -3.0, 3)
        lw_helper = e2._create_outline_circle.call_args[1]['lw']

        assert lw_main == pytest.approx(3.0)
        assert lw_helper == pytest.approx(3.0)


class TestWellSafetyZoneIntegration:
    """Full renderWellSafetyZone calls _draw_well_circle with correct args."""

    def test_render_well_safety_zone_calls_draw(self):
        """renderWellSafetyZone delegates to _draw_well_circle."""
        e = _make_env()
        e._draw_well_circle = MagicMock()
        e._render_helper_wells = MagicMock()
        RenderMixin.renderWellSafetyZone(e, zorder=5)
        e._draw_well_circle.assert_called_once_with(
            (50, 30), 20.0, -1.5, -5.0, 5
        )

    def test_render_well_safety_zone_helper_skipped_for_1sd(self):
        """1s-d has no helper wells -> _render_helper_wells exits early."""
        e = _make_env(ENVTYPE='1s-d', nHelperWells=0)
        # _render_helper_wells should not call _create_outline_circle for 1s-d
        RenderMixin._render_helper_wells(e, 3)
        assert e._create_outline_circle.call_count == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
