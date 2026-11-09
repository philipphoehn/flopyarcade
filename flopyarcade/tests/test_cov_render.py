"""Coverage tests for flopyarcade/render.py and flopyarcade/render3d.py.
All heavy external dependencies are mocked via sys.modules before import."""
import sys
import os
import types
import importlib.util
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────
# Mock heavy dependencies BEFORE importing target modules
# ──────────────────────────────────────────────────────────────────────

def _mk(name):
    m = MagicMock()
    sys.modules[name] = m
    return m

for name in [
    'flopy','flopy.modflow','flopy.mf6','flopy.mf6.modflow',
    'flopy.mf6.modflow.mfgwfwel','flopy.mf6.utils','flopy.mf6.utils.binarygrid_util',
    'flopy.modpath','flopy.plot','flopy.utils','flopy.utils.binaryfile',
    'gymnasium','imageio','joblib',
    'pathos','pathos.multiprocessing','pathos.helpers','pathos.pools',
    'xmipy',
    'tensorflow','tensorflow.keras','tensorflow.keras.initializers',
    'tensorflow.keras.layers','tensorflow.keras.models',
    'tensorflow.keras.optimizers','tensorflow.keras.random','tensorflow.keras.backend',
    'tensorflow.compat','tensorflow.compat.v1',
    'tqdm','ipykernel','IPython','IPython.display',
]:
    _mk(name)

sys.modules['pathos.helpers'].mp = MagicMock()
sys.modules['pathos.helpers'].mp.context = MagicMock()
sys.modules['pathos.helpers'].mp.context._force_start_method = MagicMock()

# Fix IPython mock so matplotlib's _fix_ipython_backend2gui doesn't crash
sys.modules['IPython'].version_info = (8, 25, 0)

# ──────────────────────────────────────────────────────────────────────
# Import target modules directly by file path (bypasses package __init__)
# ──────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
RENDER_PATH = os.path.join(BASE, '..', 'render.py')
RENDER3D_PATH = os.path.join(BASE, '..', 'render3d.py')

_pkg = types.ModuleType("flopyarcade")
sys.modules["flopyarcade"] = _pkg
sys.modules["flopyarcade"].__path__ = [os.path.dirname(RENDER_PATH)]
sys.modules["flopyarcade"].__file__ = os.path.join(BASE, "..", "__init__.py")

spec_r = importlib.util.spec_from_file_location('flopyarcade.render', RENDER_PATH)
mod_r = importlib.util.module_from_spec(spec_r)
sys.modules['flopyarcade.render'] = mod_r
spec_r.loader.exec_module(mod_r)
RenderMixin = mod_r.RenderMixin

spec_r3d = importlib.util.spec_from_file_location('flopyarcade.render3d', RENDER3D_PATH)
mod_r3d = importlib.util.module_from_spec(spec_r3d)
sys.modules['flopyarcade.render3d'] = mod_r3d
spec_r3d.loader.exec_module(mod_r3d)
Render3D = mod_r3d.Render3D
render_environment_3d = mod_r3d.render_environment_3d

# ──────────────────────────────────────────────────────────────────────
# Helper: concrete subclass + mock env builder
# ──────────────────────────────────────────────────────────────────────

class _Dummy(RenderMixin):
    pass

# Add methods that live on OTHER mixins (not importable here) as no-ops
_DUMMY_METHODS = [
    '_circle_coords', '_create_outline_circle', '_create_filled_circle',
    '_generate_circle_coords', '_add_curved_text', '_get_particle_coords',
    '_count_trajectory_points', '_generate_fade_colors', 'clear_axis',
    'figToArray', 'textSpanAcrossAxis', '_get_game_result', 'captureKeyPress',
    '_setup_axes', '_detect_ipython', '_update_title', '_load_grid',
    '_load_head_file', '_handle_display_or_return', '_handle_display_mode',
    'format_score', 'format_time',
]
for _m in _DUMMY_METHODS:
    if not hasattr(_Dummy, _m):
        setattr(_Dummy, _m, lambda self, *a, **kw: MagicMock())


def make_env(**overrides):
    e = _Dummy.__new__(_Dummy)
    defaults = {
        'ENVTYPE': '1s-c',
        'nlay': 1, 'nrow': 10, 'ncol': 10,
        'dRow': 10.0, 'dCol': 10.0,
        'heads': np.random.rand(10, 10),
        'head': np.random.rand(1, 10, 10),
        'chd_west': 10.0, 'chd_east': 5.0,
        'wellCoords': (50.0, 50.0), 'wellRadius': 20.0, 'wellQ': -1.5,
        'minQ': -5.0, 'nHelperWells': 0, 'helperWells': {},
        'helperWellRadius': 15.0, 'minQhelper': -3.0,
        'particleCoords': (30.0, 30.0),
        'particleY': 30.0,
        'timeStep': 0, 'nstp': 100, 'NAGENTSTEPS': 100,
        'done': False,
        'trajectories': {'x': [np.array([0.0, 1.0, 2.0])], 'y': [np.array([0.0, 1.0, 2.0])]},
        'stormStarts': [], 'stormCentersX': [], 'stormCentersY': [],
        'modelmap': MagicMock(),
        'ax': MagicMock(), 'ax2': MagicMock(), 'ax3': MagicMock(),
        'fig': MagicMock(),
        'canvasInitialized': False,
        'plotArrays': [], 'plotfilesSaved': [],
        'SAVEPLOT': False,
        'RENDER': False, 'MANUALCONTROL': False, 'MANUALCONTROLTIME': 1.0,
        'flagFromIPythonNotebook': False,
        'wrkspc': '/tmp', 'ANIMATIONFOLDER': 'anim', 'MODELNAMEGENCOUNT': 'test',
        'MODELNAME': 'test',
        'extentX': 100.0, 'extentY': 100.0,
        'minX': 0.0, 'minY': 0.0,
        'headSpecWest': 10.0, 'headSpecEast': 5.0,
        'actionValue': 2.0, 'actionValueNorth': 3.0, 'actionValueSouth': 1.0,
        'actionValueEast': 2.5, 'actionValueWest': 1.5,
        'headSpecNorth': 8.0, 'headSpecSouth': 7.0,
        'gameResult': 'WIN',
        '_frame_times': [], '_fps': 0.0, '_fps_artist': None,
        '_blit_background': None,
        '_blit_artists': {'particle': None, 'trajectory': [], 'score_text': None},
        'mf': MagicMock(),
        'actionsDict': {},
        'plotspth': '/tmp/plots',
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(e, k, v)
    # Override dummy methods with proper return values where needed
    e._circle_coords = lambda coords: (coords[0], coords[1])
    e._create_outline_circle = lambda *a, **kw: MagicMock()
    e._create_filled_circle = lambda *a, **kw: MagicMock()
    e._generate_circle_coords = lambda r: (np.zeros(10), np.zeros(10))
    e._add_curved_text = lambda *a, **kw: None
    e._get_particle_coords = lambda: (30.0, 40.0)
    e._count_trajectory_points = lambda: (3, [3])
    e._generate_fade_colors = lambda n: np.random.rand(n, 4)
    e.clear_axis = lambda ax: None
    e.figToArray = lambda fig: np.zeros((10, 10, 4), dtype=np.uint8)
    e.textSpanAcrossAxis = lambda *a, **kw: None
    e._get_game_result = lambda: 'WIN'
    e.captureKeyPress = lambda *a: None
    e._setup_axes = lambda: None
    e._detect_ipython = lambda: None
    e._update_title = lambda: None
    e._load_grid = lambda: None
    e._load_head_file = lambda: MagicMock()
    e._handle_display_or_return = lambda cb, rf, dpi: None
    e._handle_display_mode = lambda: None
    e.format_score = lambda: '100'
    e.format_time = lambda: '5d'
    return e


# ──────────────────────────────────────────────────────────────────────
# render.py – RenderMixin tests
# ──────────────────────────────────────────────────────────────────────

class TestFormatLabel:
    def test_basic(self):
        e = make_env()
        result = e._format_label(123)
        assert '1' in result and '2' in result and '3' in result

    def test_negative(self):
        e = make_env()
        result = e._format_label(-456)
        assert '4' in result and '5' in result and '6' in result

    def test_custom_params(self):
        e = make_env()
        result = e._format_label(10, unit='L', exponent='s', delimiter='|')
        assert '|' in result and 'L' in result


class TestDrawWellCircle:
    def test_basic(self):
        e = make_env()
        e._draw_well_circle((50, 50), 20.0, -1.5, -5.0, 3)


class TestRenderHelperWells:
    def test_invalid_envtype(self):
        e = make_env(ENVTYPE='1s-c', nHelperWells=2)
        e._render_helper_wells(3)

    def test_valid_envtype_no_wells(self):
        e = make_env(ENVTYPE='4s-c', nHelperWells=0)
        e._render_helper_wells(3)

    def test_with_helper_wells(self):
        e = make_env(ENVTYPE='4s-c', nHelperWells=2)
        e.helperWells = {'wellCoords1': (20.0, 30.0), 'wellQ1': -1.0,
                         'wellCoords2': (60.0, 60.0), 'wellQ2': 2.0}
        with patch.object(RenderMixin, '_draw_well_circle') as dwc:
            e._render_helper_wells(3)
            assert dwc.call_count == 2


class TestRenderTextOnCanvasPumpingRate:
    def test_basic(self):
        e = make_env()
        with patch.object(RenderMixin, 'render_all_well_labels'):
            e.renderTextOnCanvasPumpingRate(zorder=10)


class TestRenderContourLines:
    def test_basic(self):
        e = make_env(heads=np.random.rand(100))  # 1D so min/max work
        e.modelmap = MagicMock()
        e.modelmap.contour_array = MagicMock(return_value=MagicMock())
        e.renderContourLines(n=10, zorder=4)
        assert hasattr(e, 'levels')
        assert hasattr(e, 'contours')


class TestRenderWellSafetyZone:
    def test_basic(self):
        e = make_env()
        with patch.object(RenderMixin, '_draw_well_circle'), \
             patch.object(RenderMixin, '_render_helper_wells'):
            e.renderWellSafetyZone(zorder=3)


class TestPlotBaseMap:
    def test_basic(self):
        e = make_env()
        e.modelmap = MagicMock()
        e._plot_base_map()
        e.modelmap.plot_grid.assert_called_once_with(alpha=0.1)


class TestPlotStormCenters:
    def test_no_storm(self):
        e = make_env(stormStarts=[], timeStep=0)
        e._plot_storm_centers()

    def test_with_storm(self):
        e = make_env(stormStarts=[0, 50], timeStep=0)
        e.stormCentersX = [10.0, 80.0]
        e.stormCentersY = [20.0, 60.0]
        e._plot_storm_centers()
        e.ax.scatter.assert_called()


class TestPlotHeadArray:
    def test_basic(self):
        e = make_env()
        e.modelmap = MagicMock()
        pa = MagicMock()
        e.modelmap.plot_array.return_value = pa
        cb = e._plot_head_array()
        e.fig.colorbar.assert_called_once()


class TestPlotHelperWells:
    def test_no_wells(self):
        e = make_env(nHelperWells=0)
        e._plot_helper_wells()

    def test_with_wells(self):
        e = make_env(nHelperWells=2)
        e.actionsDict = {'well0x': 10, 'well0y': 20, 'well0Q': -2.0,
                         'well1x': 30, 'well1y': 40, 'well1Q': 3.0}
        e._plot_helper_wells()


class TestDisplayPlot:
    def test_basic(self):
        e = make_env(timeStep=0, stormStarts=[])
        cb = MagicMock()
        with patch.object(mod_r, 'show'), \
             patch.object(mod_r, 'waitforbuttonpress'):
            e._display_plot(cb)
        cb.remove.assert_called_once()

    def test_with_storm(self):
        e = make_env(timeStep=0, stormStarts=[0])
        cb = MagicMock()
        with patch.object(mod_r, 'show'), \
             patch.object(mod_r, 'waitforbuttonpress'):
            e._display_plot(cb)


class TestSaveGifIfNeeded:
    def test_not_done(self):
        e = make_env(done=False, timeStep=0, nstp=100)
        e._save_gif_if_needed()

    def test_done(self):
        e = make_env(done=True, timeStep=0, nstp=100)
        with patch.object(RenderMixin, 'writeGIFtodisk') as wg:
            e._save_gif_if_needed()
            wg.assert_called_once()


class TestRenderEnvtype0sC:
    def test_basic(self):
        e = make_env(ENVTYPE='0s-c', timeStep=0)
        with patch.object(mod_r, 'figure', return_value=MagicMock()):
            e.render_envtype_0s_c(False, 'human', 120)


class TestRenderEnvtypeOther:
    def test_no_return(self):
        e = make_env(ENVTYPE='1s-c')
        with patch.object(mod_r, 'figure', return_value=MagicMock()),              patch.object(RenderMixin, 'renderContourLines'):
            e.render_envtype_other(False, 120)

    def test_return_figure(self):
        e = make_env(ENVTYPE='1s-c')
        with patch.object(mod_r, 'figure', return_value=MagicMock()),              patch.object(RenderMixin, 'renderContourLines'),              patch.object(RenderMixin, '_handle_return_figure', return_value=np.zeros((10,10,4), dtype=np.uint8)):
            result = e.render_envtype_other(True, 120)
            assert result is not None


class TestInitCanvasIfNeeded:
    def test_first_step(self):
        e = make_env(timeStep=0, canvasInitialized=False)
        with patch.object(mod_r, 'figure', return_value=MagicMock()):
            e._init_canvas_if_needed()
            assert e.canvasInitialized is True

    def test_already_initialized(self):
        e = make_env(timeStep=5, canvasInitialized=True)
        with patch.object(RenderMixin, 'renderInitializeCanvas') as ric:
            e._init_canvas_if_needed()
            ric.assert_not_called()


class TestPlotEnvironment:
    def test_basic(self):
        e = make_env(ENVTYPE='1s-c')
        e.modelmap = MagicMock()
        e.mf = MagicMock()
        with patch.object(mod_r, 'PlotMapView') as pmm, \
             patch.object(RenderMixin, 'renderWellSafetyZone'), \
             patch.object(RenderMixin, 'renderContourLines'), \
             patch.object(RenderMixin, 'renderIdealParticleTrajectory'), \
             patch.object(RenderMixin, 'renderTextOnCanvasPumpingRate'), \
             patch.object(RenderMixin, 'renderTextOnCanvasGameOutcome'), \
             patch.object(RenderMixin, 'renderParticle'), \
             patch.object(RenderMixin, 'renderParticleTrajectory'), \
             patch.object(RenderMixin, 'renderTextOnCanvasTimeAndScore'), \
             patch.object(RenderMixin, 'renderRemoveAxesTicks'), \
             patch.object(RenderMixin, 'renderSetAxesLimits'), \
             patch.object(RenderMixin, 'renderAddAxesTextLabels'):
            e._plot_environment()


class TestPrepareFigureForReturn:
    def test_basic(self):
        e = make_env()
        e.fig = MagicMock()
        e.fig.get_size_inches.return_value = np.array([8.0, 8.0])
        with patch.object(mod_r, 'margins'), \
             patch.object(mod_r, 'NullLocator'):
            s = e._prepare_figure_for_return(120)
        assert s is not None


class TestRestoreFigureSize:
    def test_basic(self):
        e = make_env()
        e.fig = MagicMock()
        e._restore_figure_size((7.0, 7.0))
        e.fig.set_size_inches.assert_called_once_with((7.0, 7.0))


class TestHandleReturnFigure:
    def test_basic(self):
        e = make_env(SAVEPLOT=False, done=False, timeStep=0, NAGENTSTEPS=100)
        with patch.object(RenderMixin, '_prepare_figure_for_return', return_value=(7,7)), \
             patch.object(RenderMixin, 'fig_to_rgb_array', return_value=np.zeros((10,10,4), dtype=np.uint8)), \
             patch.object(RenderMixin, '_restore_figure_size'), \
             patch.object(RenderMixin, 'renderClearAxes'):
            result = e._handle_return_figure(120)
            assert result.shape == (10, 10, 4)


class TestRenderInitializeCanvas:
    def test_basic(self):
        e = make_env(RENDER=False, MANUALCONTROL=False, flagFromIPythonNotebook=False)
        with patch.object(RenderMixin, '_create_figure'), \
             patch.object(RenderMixin, '_maximize_figure_window'), \
             patch.object(RenderMixin, '_finalize_figure'):
            e.renderInitializeCanvas()
            assert e.canvasInitialized is True


class TestCreateFigure:
    def test_basic(self):
        e = make_env()
        with patch.object(mod_r, 'figure', return_value=MagicMock()) as fig_mock:
            e._create_figure()
            assert e.fig is not None


class TestMaximizeFigureWindow:
    def test_basic(self):
        e = make_env()
        e.fig = MagicMock()
        with patch.object(mod_r, 'get_current_fig_manager', return_value=MagicMock()):
            e._maximize_figure_window()

    def test_fallback(self):
        e = make_env()
        e.fig = MagicMock()
        mgr = MagicMock()
        mgr.window.state.side_effect = Exception('no state')
        with patch.object(mod_r, 'get_current_fig_manager', return_value=mgr):
            e._maximize_figure_window()


class TestFinalizeFigure:
    def test_basic(self):
        e = make_env()
        e.fig = MagicMock()
        e.fig.gca.return_value = MagicMock()
        with patch.object(mod_r, 'NullLocator'):
            e._finalize_figure()


class TestRenderAllWellLabels:
    def test_basic(self):
        e = make_env()
        with patch.object(RenderMixin, '_render_main_well_label'), \
             patch.object(RenderMixin, '_render_helper_wells_labels'):
            e.render_all_well_labels('red')


class TestRenderMainWellLabel:
    def test_basic(self):
        e = make_env()
        e._render_main_well_label('red')


class TestRenderGetWellColor:
    def test_positive(self):
        e = make_env()
        assert e.renderGetWellColor(5.0) == 'blue'

    def test_negative(self):
        e = make_env()
        assert e.renderGetWellColor(-5.0) == 'red'

    def test_zero(self):
        e = make_env()
        assert e.renderGetWellColor(0.0) == 'gray'


class TestRenderGetWellCircleAlpha:
    def test_basic(self):
        e = make_env()
        alpha = e.renderGetWellCircleAlpha(2.0, 4.0)
        assert alpha == 0.5 * 0.9

    def test_zero_max(self):
        e = make_env()
        assert e.renderGetWellCircleAlpha(2.0, 0) == 0.0


class TestRenderTextOnCanvasGameOutcome:
    def test_not_done(self):
        e = make_env(done=False)
        e.renderTextOnCanvasGameOutcome()

    def test_done(self):
        e = make_env(done=True)
        e.renderTextOnCanvasGameOutcome()


class TestRenderGameResultText:
    def test_basic(self):
        e = make_env()
        e.ax2 = MagicMock()
        e._render_game_result_text('WIN', 10)


class TestRenderTextOnCanvasTimeAndScore:
    def test_basic(self):
        e = make_env()
        e.ax2 = MagicMock()
        e.renderTextOnCanvasTimeAndScore()


class TestRenderParticle:
    def test_basic(self):
        e = make_env()
        e.renderParticle(zorder=6)


class TestPlotParticle:
    def test_basic(self):
        e = make_env(timeStep=0)
        e.ax2 = MagicMock()
        e._plot_particle(10.0, 20.0, 6)
        e.ax2.scatter.assert_called_once()


class TestRenderParticleTrajectory:
    def test_zero_step(self):
        e = make_env(timeStep=0)
        e.renderParticleTrajectory()

    def test_with_trajectory(self):
        e = make_env(timeStep=5)
        e.renderParticleTrajectory()


class TestPlotTrajectories:
    def test_basic(self):
        e = make_env(timeStep=5)
        e.trajectories = {'x': [np.array([0.0, 1.0, 2.0])], 'y': [np.array([0.0, 1.0, 2.0])]}
        e.ax2 = MagicMock()
        colors = np.array([[0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, 1.0], [0.7, 0.8, 0.9, 1.0]])
        lengths = [3]
        e._plot_trajectories(lengths, colors, 6)


class TestFigToRgbArray:
    def test_basic(self):
        e = make_env()
        fig = MagicMock()
        w, h = 10, 10
        fig.canvas.get_width_height.return_value = (w, h)
        fig.canvas.tostring_argb.return_value = b'\x00' * (w * h * 4)
        result = e.fig_to_rgb_array(fig)
        assert result.shape == (10, 10, 4)


class TestRender:
    def test_envtype_0s_c(self):
        e = make_env(ENVTYPE='0s-c', timeStep=0, SAVEPLOT=False)
        with patch.object(RenderMixin, 'render_envtype_0s_c') as rec:
            e.render(mode='human')
            rec.assert_called_once()

    def test_envtype_other(self):
        e = make_env(ENVTYPE='1s-c', timeStep=0, SAVEPLOT=False)
        with patch.object(RenderMixin, 'render_envtype_other') as reo:
            e.render(mode='human')
            reo.assert_called_once()

    def test_rgb_array_mode(self):
        e = make_env(ENVTYPE='1s-c', timeStep=0, SAVEPLOT=False)
        with patch.object(RenderMixin, 'render_envtype_other') as reo:
            e.render(mode='rgb_array')
            reo.assert_called_once()


class TestRenderIdealParticleTrajectory:
    def test_basic(self):
        e = make_env(minX=0.0, extentX=100.0, particleY=50.0)
        e.ax2 = MagicMock()
        e.renderIdealParticleTrajectory(zorder=5)


class TestRenderRemoveAxesTicks:
    def test_basic(self):
        e = make_env(ENVTYPE='1s-c')
        e.ax = MagicMock()
        e.ax2 = MagicMock()
        e.ax3 = MagicMock()
        e.renderRemoveAxesTicks()


class TestRenderSetAxesLimits:
    def test_basic(self):
        e = make_env(minX=0.0, minY=0.0, extentX=100.0, extentY=100.0)
        e.ax = MagicMock()
        e.ax2 = MagicMock()
        e.ax3 = MagicMock()
        e.renderSetAxesLimits()


class TestRenderAddAxesTextLabels:
    def test_basic(self):
        e = make_env(minX=0.0, minY=0.0, extentX=100.0, extentY=100.0, dCol=10.0, dRow=10.0)
        e.ax2 = MagicMock()
        with patch.object(RenderMixin, '_add_vertical_labels'), \
             patch.object(RenderMixin, '_add_horizontal_labels'):
            e.renderAddAxesTextLabels()


class TestAddVerticalLabels:
    def test_basic(self):
        e = make_env(minX=0.0, dCol=10.0, extentX=100.0)
        e.ax2 = MagicMock()
        e._add_vertical_labels('left', 'right', 50.0)


class TestAddHorizontalLabels:
    def test_basic(self):
        e = make_env(extentY=100.0, dRow=10.0, minY=0.0)
        e.ax2 = MagicMock()
        e._add_horizontal_labels('top', 'bottom', 50.0)


class TestGetLrLabels:
    def test_basic(self):
        e = make_env(headSpecWest=10.0, headSpecEast=5.0)
        left, right = e._get_lr_labels()
        assert '10.00' in left
        assert '5.00' in right


class TestGetTbLabels:
    def test_group1(self):
        e = make_env(ENVTYPE='1s-c', actionValueNorth=3.0, actionValueSouth=1.0)
        top, bottom = e._get_tb_labels()
        assert '3.00' in top
        assert '1.00' in bottom

    def test_group2(self):
        e = make_env(ENVTYPE='2s-c', actionValue=2.0)
        top, bottom = e._get_tb_labels()
        assert '2.00' in bottom

    def test_group3(self):
        e = make_env(ENVTYPE='4s-c', headSpecNorth=8.0, headSpecSouth=7.0)
        top, bottom = e._get_tb_labels()
        assert '8.00' in top
        assert '7.00' in bottom

    def test_other(self):
        e = make_env(ENVTYPE='99')
        top, bottom = e._get_tb_labels()
        assert top == ''
        assert bottom == ''


class TestRenderClearAxes:
    def test_basic(self):
        e = make_env(ENVTYPE='1s-c')
        e.ax = MagicMock()
        e.ax2 = MagicMock()
        e.ax3 = MagicMock()
        e.renderClearAxes()


class TestRenderSavePlot:
    def test_basic(self):
        e = make_env(timeStep=0, SAVEPLOT=False, NAGENTSTEPS=100)
        e.fig = MagicMock()
        e.fig.get_size_inches.return_value = np.array([7.0, 7.0])
        with patch.object(mod_r, 'makedirs'), \
             patch.object(mod_r, 'imsave'), \
             patch.object(mod_r, 'NullLocator'):
            e.renderSavePlot()


class TestEnsurePlotDirs:
    def test_basic(self):
        e = make_env(timeStep=0, wrkspc='/tmp', ANIMATIONFOLDER='anim')
        with patch.object(mod_r, 'makedirs'):
            e._ensure_plot_dirs()


class TestGetPlotFilepath:
    def test_basic(self):
        e = make_env(timeStep=5, NAGENTSTEPS=100, MODELNAME='test', plotspth='/tmp/plots')
        path = e._get_plot_filepath()
        assert 'test' in path
        assert '005' in path


class TestPrepareFigureForSaving:
    def test_basic(self):
        e = make_env()
        e.fig = MagicMock()
        e.fig.gca.return_value = MagicMock()
        with patch.object(mod_r, 'NullLocator'):
            e._prepare_figure_for_saving()


class TestBlitInit:
    def test_basic(self):
        e = make_env()
        e.fig = MagicMock()
        e.fig.bbox = MagicMock()
        e._blit_background = None
        e.render_blit_init()


class TestBlitFrame:
    def test_basic(self):
        e = make_env(timeStep=5)
        e.fig = MagicMock()
        e.fig.bbox = MagicMock()
        e._blit_background = MagicMock()
        with patch.object(RenderMixin, '_blit_update_particle'), \
             patch.object(RenderMixin, '_blit_update_trajectory'), \
             patch.object(RenderMixin, '_blit_update_score_text'):
            e.render_blit_frame()


class TestBlitUpdateParticle:
    def test_basic(self):
        e = make_env(timeStep=0)
        e.ax2 = MagicMock()
        e._blit_artists = {'particle': None}
        e._blit_update_particle()

    def test_existing(self):
        e = make_env(timeStep=5)
        e.ax2 = MagicMock()
        particle_artist = MagicMock()
        e._blit_artists = {'particle': particle_artist}
        e._blit_update_particle()
        particle_artist.set_offsets.assert_called_once()


class TestBlitUpdateTrajectory:
    def test_zero_step(self):
        e = make_env(timeStep=0)
        e._blit_update_trajectory()

    def test_with_traj(self):
        e = make_env(timeStep=5)
        e.trajectories = {'x': [np.array([0.0, 1.0])], 'y': [np.array([0.0, 1.0])]}
        e.ax2 = MagicMock()
        e.ax2.plot.return_value = [MagicMock()]  # line, = ax2.plot(...)
        e._blit_artists = {'trajectory': []}
        e._count_trajectory_points = lambda: (1, [1])
        e._generate_fade_colors = lambda n: np.random.rand(n, 4)
        e._blit_update_trajectory()


class TestBlitUpdateScoreText:
    def test_new(self):
        e = make_env()
        e.ax2 = MagicMock()
        e._blit_artists = {'score_text': None}
        e._blit_update_score_text()

    def test_existing(self):
        e = make_env()
        e.ax2 = MagicMock()
        text_artist = MagicMock()
        e._blit_artists = {'score_text': text_artist}
        e._blit_update_score_text()
        text_artist.set_text.assert_called_once()


class TestRenderFPS:
    def test_first_frame(self):
        e = make_env()
        e._frame_times = []
        e.ax2 = MagicMock()
        e.renderFPS()

    def test_subsequent(self):
        import time as _time
        e = make_env()
        e._frame_times = [_time.time() - 0.1, _time.time()]
        e.ax2 = MagicMock()
        e._fps_artist = None
        e.renderFPS()


class TestWriteGifToDisk:
    def test_basic(self):
        e = make_env()
        with patch.object(RenderMixin, '_write_gif'), \
             patch.object(RenderMixin, '_optimize_gif'):
            e.writeGIFtodisk('/tmp/test.gif', [np.zeros((10,10,4), dtype=np.uint8)], fps=2, optimizeSize=True)


class TestWriteGif:
    def test_basic(self):
        e = make_env()
        with patch.object(mod_r, 'get_writer') as gw:
            ctx = MagicMock()
            gw.return_value.__enter__ = MagicMock(return_value=ctx)
            gw.return_value.__exit__ = MagicMock(return_value=False)
            e._write_gif('/tmp/test.gif', [np.zeros((10,10,4), dtype=np.uint8)], 2)


class TestOptimizeGif:
    def test_no_module(self):
        e = make_env()
        with patch.dict(sys.modules, {'pygifsicle': None}):
            e._optimize_gif('/tmp/test.gif')


class TestRenderUserInterAction:
    def test_basic(self):
        e = make_env(timeStep=0, MANUALCONTROLTIME=1.0)
        e.fig = MagicMock()
        e.fig.canvas = MagicMock()
        with patch.object(mod_r, 'show'), \
             patch.object(mod_r, 'waitforbuttonpress'):
            e.renderUserInterAction()


class TestRenderHelperWellsLabels:
    def test_invalid_envtype(self):
        e = make_env(ENVTYPE='1s-c')
        e._render_helper_wells_labels()

    def test_valid_envtype(self):
        e = make_env(ENVTYPE='4s-c', nHelperWells=2)
        e.helperWells = {'wellCoords1': (10, 20), 'wellQ1': -1.0,
                         'wellCoords2': (30, 40), 'wellQ2': 2.0}
        e._render_helper_wells_labels()


class TestGetTextLabels:
    def test_basic(self):
        e = make_env(ENVTYPE='1s-c', headSpecWest=10.0, headSpecEast=5.0,
                     actionValueNorth=3.0, actionValueSouth=1.0)
        left, right, top, bottom = e._get_text_labels()
        assert '10.00' in left
        assert '5.00' in right


# ──────────────────────────────────────────────────────────────────────
# render3d.py – Render3D tests
# ──────────────────────────────────────────────────────────────────────

def make_env3d(**overrides):
    e = MagicMock()
    defaults = {
        'nrow': 5, 'ncol': 5,
        'delr': 10.0, 'delc': 10.0,
        'extentX': 50.0, 'extentY': 50.0,
        'zTop': 50.0, 'zBot': 0.0,
        'heads': np.random.rand(5, 5),
        'particleCoords': (25.0, 25.0),
        'wellCoords': (30.0, 30.0),
        'wellRadius': 10.0,
        'headSpecNorth': 8.0, 'headSpecSouth': 7.0,
        'ENVTYPE': '1s-c',
        'rewardCurrent': 50,
        'timeStep': 5,
        'periodLength': 50,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(e, k, v)
    return e


class TestRender3DInit:
    def test_basic(self):
        env = make_env3d()
        r = Render3D(env, headless=True)
        assert r.nrow == 5
        assert r.ncol == 5
        assert r.trajectory == []
        assert r._grid_cache is None


class TestRender3DBuildGrid:
    def test_basic(self):
        env = make_env3d(nrow=3, ncol=3, extentX=30.0, extentY=30.0)
        env.heads = np.random.rand(3, 3)
        r = Render3D(env, headless=True)
        X, Y, Z, head_nodes = r._build_grid()
        assert X.shape == (4, 4)
        assert Y.shape == (4, 4)
        assert Z.shape == (4, 4)

    def test_cached(self):
        env = make_env3d(nrow=3, ncol=3, extentX=30.0, extentY=30.0)
        env.heads = np.random.rand(3, 3)
        r = Render3D(env, headless=True)
        g1 = r._build_grid()
        g2 = r._build_grid()
        assert g1 is g2


class TestRender3DInterpolateHead:
    def test_basic(self):
        env = make_env3d(nrow=3, ncol=3, delr=10.0, delc=10.0)
        env.heads = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])
        r = Render3D(env, headless=True)
        h = r._interpolate_head(15.0, 15.0)
        assert 1.0 <= h <= 9.0


class TestRender3DPos3d:
    def test_basic(self):
        env = make_env3d(nrow=3, ncol=3)
        env.heads = np.ones((3, 3)) * 5.0
        r = Render3D(env, headless=True, elevation_scale=2.0)
        pos = r._pos3d(10.0, 10.0)
        assert pos.shape == (3,)


class TestRender3DGetParticlePosition:
    def test_basic(self):
        env = make_env3d(nrow=3, ncol=3)
        env.heads = np.ones((3, 3))
        r = Render3D(env, headless=True)
        pos = r._get_particle_position()
        assert pos.shape == (3,)


class TestRender3DGetWellPosition:
    def test_basic(self):
        env = make_env3d(nrow=3, ncol=3)
        env.heads = np.ones((3, 3))
        r = Render3D(env, headless=True)
        pos = r._get_well_position()
        assert pos.shape == (3,)


class TestRender3DRender:
    def test_headless(self):
        env = make_env3d(nrow=3, ncol=3, extentX=30.0, extentY=30.0)
        env.heads = np.random.rand(3, 3)
        r = Render3D(env, headless=True)
        with patch.object(mod_r3d, 'plt'):
            r.render(save_path=None)

    def test_save_path(self):
        env = make_env3d(nrow=3, ncol=3, extentX=30.0, extentY=30.0)
        env.heads = np.random.rand(3, 3)
        r = Render3D(env, headless=True)
        import tempfile
        with patch.object(mod_r3d, 'plt'):
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = os.path.join(tmpdir, 'test.png')
                result = r.render(save_path=save_path)
                assert result == save_path


class TestRender3DRecordTrajectory:
    def test_basic(self):
        env = make_env3d(nrow=3, ncol=3)
        env.heads = np.ones((3, 3))
        r = Render3D(env, headless=True)
        r.record_trajectory()
        assert len(r.trajectory) == 1


class TestRenderEnvironment3d:
    def test_basic(self):
        env = make_env3d(nrow=3, ncol=3, extentX=30.0, extentY=30.0)
        env.heads = np.random.rand(3, 3)
        with patch.object(mod_r3d, 'plt'):
            result = render_environment_3d(env, output_path=None)
        assert result is None
