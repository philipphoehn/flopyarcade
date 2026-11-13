"""Maximize coverage: agent pure-math, play main, train_rllib configs, play_againstai config."""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from argparse import ArgumentParser


# ─── 1. __init__.py: full import coverage ────────────────────────────────────

class TestInitFullImport:
    def test_all_importable(self):
        """Every name in __all__ must be importable."""
        import flopyarcade
        for name in flopyarcade.__all__:
            assert hasattr(flopyarcade, name), f"{name} not exported"

    def test_flopyenv_callable(self):
        import flopyarcade
        assert callable(flopyarcade.FloPyEnv)

    def test_flopyarcade_callable(self):
        import flopyarcade
        assert callable(flopyarcade.FloPyArcade)

    def test_flopyagent_callable(self):
        import flopyarcade
        assert callable(flopyarcade.FloPyAgent)

    def test_version_format(self):
        import flopyarcade
        assert isinstance(flopyarcade.__version__, str)
        parts = flopyarcade.__version__.split('.')
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ─── 2. play.py: main() flag branches ─────────────────────────────────────────

class TestPlayMain:
    """Test play.main() with various flag combinations using mocks."""

    @patch('flopyarcade.play.FloPyArcade')
    def test_main_defaults(self, mock_arcade_cls):
        """main() with no CLI args uses defaults."""
        with patch.object(sys, 'argv', ['play.py']):
            from flopyarcade.play import main
            mock_game = MagicMock()
            mock_game.env = MagicMock()
            mock_game.env.envhash = 'abc123'
            mock_arcade_cls.return_value = mock_game
            result = main()
            assert result == 'abc123'
            mock_arcade_cls.assert_called_once()
            # verify ENVTYPE default
            call_kwargs = mock_arcade_cls.call_args[1]
            assert call_kwargs['ENVTYPE'] == '3s-d'
            assert call_kwargs['flagRender'] is True
            assert call_kwargs['flagManualControl'] is False
            assert call_kwargs['flagSavePlot'] is False

    @patch('flopyarcade.play.FloPyArcade')
    def test_main_manualcontrol(self, mock_arcade_cls):
        """main() with --manualcontrol True."""
        with patch.object(sys, 'argv', ['play.py', '--manualcontrol', 'True']):
            from flopyarcade.play import main
            mock_game = MagicMock()
            mock_game.env = MagicMock()
            mock_game.env.envhash = 'hash_mc'
            mock_arcade_cls.return_value = mock_game
            main()
            call_kwargs = mock_arcade_cls.call_args[1]
            assert call_kwargs['flagManualControl'] is True

    @patch('flopyarcade.play.FloPyArcade')
    def test_main_norender(self, mock_arcade_cls):
        """main() with --render False."""
        with patch.object(sys, 'argv', ['play.py', '--render', 'False']):
            from flopyarcade.play import main
            mock_game = MagicMock()
            mock_game.env = MagicMock()
            mock_game.env.envhash = 'hash_nr'
            mock_arcade_cls.return_value = mock_game
            main()
            call_kwargs = mock_arcade_cls.call_args[1]
            assert call_kwargs['flagRender'] is False

    @patch('flopyarcade.play.FloPyArcade')
    def test_main_saveplot(self, mock_arcade_cls):
        """main() with --saveplot True."""
        with patch.object(sys, 'argv', ['play.py', '--saveplot', 'True']):
            from flopyarcade.play import main
            mock_game = MagicMock()
            mock_game.env = MagicMock()
            mock_game.env.envhash = 'hash_sp'
            mock_arcade_cls.return_value = mock_game
            main()
            call_kwargs = mock_arcade_cls.call_args[1]
            assert call_kwargs['flagSavePlot'] is True

    @patch('flopyarcade.play.FloPyArcade')
    def test_main_modelnameload(self, mock_arcade_cls):
        """main() with --modelnameload."""
        with patch.object(sys, 'argv', ['play.py', '--modelnameload', 'my_model.keras']):
            from flopyarcade.play import main
            mock_game = MagicMock()
            mock_game.env = MagicMock()
            mock_game.env.envhash = 'hash_ml'
            mock_arcade_cls.return_value = mock_game
            main()
            call_kwargs = mock_arcade_cls.call_args[1]
            assert call_kwargs['modelNameLoad'] == 'my_model.keras'

    @patch('flopyarcade.play.FloPyArcade')
    def test_main_custom_envtype_and_grid(self, mock_arcade_cls):
        """main() with custom envtype, nlay, nrow, ncol, ngames, nagentsteps."""
        with patch.object(sys, 'argv', [
            'play.py', '--envtype', '2s-d', '--nlay', '2',
            '--nrow', '50', '--ncol', '60', '--ngames', '5', '--nagentsteps', '100'
        ]):
            from flopyarcade.play import main
            mock_game = MagicMock()
            mock_game.env = MagicMock()
            mock_game.env.envhash = 'hash_custom'
            mock_arcade_cls.return_value = mock_game
            result = main()
            call_kwargs = mock_arcade_cls.call_args[1]
            assert call_kwargs['ENVTYPE'] == '2s-d'
            assert call_kwargs['nLay'] == 2
            assert call_kwargs['nRow'] == 50
            assert call_kwargs['nCol'] == 60
            assert call_kwargs['NAGENTSTEPS'] == 100

    @patch('flopyarcade.play.FloPyArcade')
    def test_main_multiple_games(self, mock_arcade_cls):
        """main() runs NGAMES iterations of play()."""
        with patch.object(sys, 'argv', ['play.py', '--ngames', '3']):
            from flopyarcade.play import main
            mock_game = MagicMock()
            mock_game.env = MagicMock()
            mock_game.env.envhash = 'hash_multi'
            mock_arcade_cls.return_value = mock_game
            main()
            assert mock_game.play.call_count == 3


class TestPlayHelpers:
    def test_str_to_bool_true_values(self):
        from flopyarcade.play import str_to_bool
        for v in ['true', 't', '1', 'yes', 'y', 'True', 'TRUE']:
            assert str_to_bool(v) is True

    def test_str_to_bool_false_values(self):
        from flopyarcade.play import str_to_bool
        for v in ['false', 'f', '0', 'no', 'n', 'False', 'FALSE']:
            assert str_to_bool(v) is False

    def test_str_to_bool_passthrough(self):
        from flopyarcade.play import str_to_bool
        assert str_to_bool(True) is True
        assert str_to_bool(False) is False

    def test_str_to_bool_invalid(self):
        from flopyarcade.play import str_to_bool
        with pytest.raises(ValueError, match='not a valid boolean'):
            str_to_bool('maybe')

    def test_has_arguments_no_args(self):
        from flopyarcade.play import has_arguments
        parser = ArgumentParser()
        parser.add_argument('--foo', default='bar')
        parser.add_argument('--num', default=42)
        args = parser.parse_args([])
        assert has_arguments(parser, args) is False

    def test_has_arguments_with_arg(self):
        from flopyarcade.play import has_arguments
        parser = ArgumentParser()
        parser.add_argument('--foo', default='bar')
        args = parser.parse_args(['--foo', 'baz'])
        assert has_arguments(parser, args) is True


# ─── 3. agent.py: pure math functions ─────────────────────────────────────────

class TestAgentPureMath:
    """Test FloPyAgent pure-math/utility methods without TF training."""

    @pytest.fixture
    def agent(self):
        """Create a minimal FloPyAgent in 'random' mode (no TF init)."""
        from flopyarcade.agent import FloPyAgent
        a = FloPyAgent(mode='random')
        a.hyParams = {
            'MUTATIONPROBABILITY': 0.1,
            'MUTATIONPOWER': 0.01,
            'NNOVELTYNEIGHBORS': 5,
        }
        a.agentsUnique = [0, 1, 2]
        a.agentsDuplicate = [3]
        a.noveltyItemCount = 4
        a.noveltyArchive = {
            'agent1': {'novelty': 0.5, 'actionsUniqueID': 0, 'modelFile': 'm1'},
            'agent2': {'novelty': 0.3, 'actionsUniqueID': 1, 'modelFile': 'm2'},
            'agent3': {'novelty': 0.7, 'actionsUniqueID': 2, 'modelFile': 'm3'},
            'agent4': {'novelty': 0.1, 'actionsUniqueID': 0, 'modelFile': 'm1'},
        }
        a.cores = 2
        a.maxTasksPerWorkerNoveltySearch = 100
        a.neighborLimitReached = False
        a.tempModelPrefix = '/tmp/test_model'
        a.zFill = 4
        a.geneticGeneration = 0
        a.novelties = []
        a.noveltyFilenames = []
        return a

    def test_mutate_decision_true(self, agent):
        """mutateDecision(1.0) always returns True."""
        assert agent.mutateDecision(1.0) is True

    def test_mutate_decision_false(self, agent):
        """mutateDecision(0.0) always returns False."""
        assert agent.mutateDecision(0.0) is False

    def test_mutate_decisions_shape(self, agent):
        """mutateDecisions returns array of correct shape."""
        import numpy as np
        result = agent.mutateDecisions(0.5, (10, 20))
        assert result.shape == (10, 20)
        assert result.dtype == bool

    def test_mutate_decisions_all_true(self, agent):
        """mutateDecisions(1.0, shape) all True."""
        result = agent.mutateDecisions(1.0, (5, 5))
        assert result.all()

    def test_mutate_decisions_all_false(self, agent):
        """mutateDecisions(0.0, shape) all False."""
        result = agent.mutateDecisions(0.0, (5, 5))
        assert not result.any()

    @patch('flopyarcade.agent.randn')
    @patch('flopyarcade.agent.shape')
    @patch('flopyarcade.agent.add')
    def test_mutate_genetic(self, mock_add, mock_shape, mock_randn, agent):
        """mutateGenetic modifies weights via set_weights."""
        import numpy as np
        mock_shape.return_value = (3, 3)
        mock_randn.return_value = np.ones((3, 3)) * 0.01
        mock_add.side_effect = lambda a, b: a + b

        mock_agent = MagicMock()
        mock_agent.get_weights.return_value = [np.zeros((3, 3)), np.zeros((3, 3))]

        result = agent.mutateGenetic(mock_agent, seed=42)
        assert result is mock_agent
        mock_agent.set_weights.assert_called_once()

    def test_yield_chunks_even(self, agent):
        """yieldChunks splits list into even chunks."""
        chunks = list(agent.yieldChunks([1, 2, 3, 4, 5, 6], 3))
        assert chunks == [[1, 2, 3], [4, 5, 6]]

    def test_yield_chunks_uneven(self, agent):
        """yieldChunks handles uneven splits."""
        chunks = list(agent.yieldChunks([1, 2, 3, 4, 5], 2))
        assert chunks == [[1, 2], [3, 4], [5]]

    def test_yield_chunks_single(self, agent):
        """yieldChunks with n=1 yields one element at a time."""
        chunks = list(agent.yieldChunks([1, 2, 3], 1))
        assert chunks == [[1], [2], [3]]

    def test_yield_chunks_empty(self, agent):
        """yieldChunks with empty list yields nothing."""
        chunks = list(agent.yieldChunks([], 3))
        assert chunks == []

    def test_build_novelty_args_below_limit(self, agent):
        """_build_novelty_args when below neighbor limit."""
        agent.neighborLimitReached = False
        args, flags = agent._build_novelty_args()
        assert len(args) == 3
        assert len(flags) == 3
        assert all(f is True for f in flags)
        # Each arg: [i, None, 0, len(agentsUnique), i]
        assert args[0] == [0, None, 0, 3, 0]
        assert args[2] == [2, None, 0, 3, 2]

    def test_build_novelty_args_above_limit(self, agent):
        """_build_novelty_args when above neighbor limit (uses bounds)."""
        # NNOVELTYNEIGHBORS=2, agentsUnique len=3 → 3>2=True → neighborLimitReached=True
        agent.hyParams['NNOVELTYNEIGHBORS'] = 2
        # Mock bounds: returns (range_lower, range_higher, idx, needs_update)
        # First two: needs_update=True (added to args), last: needs_update=False (skipped)
        bounds_calls = []
        def mock_bounds(i):
            bounds_calls.append(i)
            if i < 2:
                return (0, 3, i, True)   # needs_update=True
            else:
                return (1, 2, i, False)  # needs_update=False
        agent.calculateNoveltyNeighborBounds = mock_bounds
        args, flags = agent._build_novelty_args()
        assert len(bounds_calls) == 3  # called for all 3 agents
        assert len(args) == 2         # only first 2 added (needs_update=True)
        assert flags == [True, True, False]

    def test_print_novelty_stats_below(self, agent, capsys):
        """_print_novelty_stats below neighbor limit."""
        agent.hyParams['NNOVELTYNEIGHBORS'] = 10
        agent.noveltyItemCount = 4
        agent._print_novelty_stats()
        captured = capsys.readouterr()
        assert '3 unique agents' in captured.out
        assert '1 duplicate agents' in captured.out

    def test_print_novelty_stats_above(self, agent, capsys):
        """_print_novelty_stats above neighbor limit."""
        agent.hyParams['NNOVELTYNEIGHBORS'] = 2
        agent.noveltyItemCount = 5
        agent._print_novelty_stats()
        captured = capsys.readouterr()
        assert 'neighbor level reached' in captured.out

    def test_update_agent_novelties(self, agent):
        """_update_agent_novelties populates archive."""
        novelties = [0.9, 0.8, 0.7]
        update_flags = [True, True, True]
        agent._update_agent_novelties(novelties, update_flags)
        assert agent.noveltyArchive['agent1']['novelty'] == 0.9
        assert agent.noveltyArchive['agent2']['novelty'] == 0.8
        assert agent.noveltyArchive['agent3']['novelty'] == 0.7

    def test_save_novelties(self, agent, tmp_path):
        """_save_novelties writes pickle and populates lists."""
        agent.tempModelPrefix = str(tmp_path)
        agent.noveltyItemCount = 3
        agent._save_novelties()
        assert len(agent.novelties) == 3
        assert len(agent.noveltyFilenames) == 3
        assert agent.novelties[0] == 0.5
        assert agent.noveltyFilenames[0] == 'm1'


# ─── 4. agent.py: TF model save/load ──────────────────────────────────────────

class TestAgentSaveLoad:
    """Test saveBestAgent and loadAgentModel with mocks."""

    def test_save_best_agent(self, tmp_path):
        """saveBestAgent creates a .keras file."""
        from flopyarcade.agent import FloPyAgent
        a = FloPyAgent(mode='random')
        a.hyParams = {}
        a.tempModelPrefix = str(tmp_path / 'prefix')
        a.modelpth = str(tmp_path / 'models')
        os.makedirs(a.modelpth, exist_ok=True)
        a.geneticGeneration = 2
        a.zFill = 4
        a.bestAgentReward = 85.5

        mock_model = MagicMock()
        with patch('flopyarcade.agent.load_model', return_value=mock_model):
            a.saveBestAgent('TestModel')
            expected_name = f'TestModel_gen0003_avg__85.5'
            mock_model.save.assert_called_once()
            save_path = mock_model.save.call_args[0][0]
            assert 'TestModel_gen0003' in save_path
            assert save_path.endswith('.keras')

    def test_generate_path_prefixes(self, tmp_path):
        """generatePathPrefixes sets temp paths correctly."""
        from flopyarcade.agent import FloPyAgent
        a = FloPyAgent(mode='random')
        a.tempModelpth = str(tmp_path)
        a.envSettings = {'MODELNAME': 'MyModel'}
        a.geneticGeneration = 5
        a.zFill = 6
        a.generatePathPrefixes()
        assert 'MyModel_gen000006' in a.tempModelPrefix
        assert 'MyModel_gen000007' in a.tempNextModelPrefix
        assert 'MyModel_gen000005' in a.tempPrevModelPrefix


# ─── 5. train_rllib_*: config validation ──────────────────────────────────────

class TestTrainRllibConfig:
    """Validate config structure of train_rllib modules.
    These modules call parser.parse_args() at import time,
    so we must control sys.argv before importing.
    """

    def _mock_ray_agents_dqn(self):
        """Pre-mock ray.rllib.agents.dqn for modules that import ApexTrainer."""
        mock_mod = MagicMock()
        mock_mod.ApexTrainer = MagicMock
        sys.modules.setdefault('ray.rllib.agents', MagicMock())
        sys.modules.setdefault('ray.rllib.agents.dqn', mock_mod)

    def _import_module(self, module_name, extra_args=None):
        """Import a train_rllib module with controlled sys.argv."""
        self._mock_ray_agents_dqn()
        args = [module_name] + (extra_args or [])
        with patch.object(sys, 'argv', args):
            # Remove from cache to force re-import
            full_name = f'flopyarcade.examples.{module_name}'
            sys.modules.pop(full_name, None)
            import importlib
            mod = importlib.import_module(full_name)
            return mod

    def test_ppo_config_structure(self):
        mod = self._import_module('train_rllib_ppo', ['--envtype', '1s-d', '--cpus', '2'])
        # config must have required keys
        assert 'env' in mod.config or 'model' in mod.config
        assert 'seed' in mod.config
        assert 'framework' in mod.config
        assert mod.config['framework'] == 'tf'
        assert mod.config['seed'] == 1
        # model config
        assert 'fcnet_hiddens' in mod.config['model']
        assert mod.config['model']['fcnet_hiddens'] == [512, 1024, 512]
        assert mod.config['model']['fcnet_activation'] == 'relu'
        # stop criteria
        assert 'episode_reward_mean' in mod.config_stopCriteria

    def test_ppo_num_workers(self):
        """num_workers logic: cpus-2 if cpus>=3 else cpus."""
        mod = self._import_module('train_rllib_ppo', ['--cpus', '4'])
        assert mod.config['num_workers'] == 2  # 4-2

    def test_ppo_num_workers_low(self):
        mod = self._import_module('train_rllib_ppo', ['--cpus', '2'])
        assert mod.config['num_workers'] == 2  # <3, so cpus directly

    def test_appo_config_structure(self):
        mod = self._import_module('train_rllib_appo', ['--envtype', '2s-d'])
        assert hasattr(mod, 'config')
        cfg = mod.config
        assert 'seed' in cfg
        assert 'framework' in cfg

    def test_impala_config_structure(self):
        mod = self._import_module('train_rllib_impala', ['--envtype', '1s-d'])
        assert hasattr(mod, 'config')
        cfg = mod.config
        assert 'seed' in cfg

    def test_apexdqn_config_structure(self):
        # ApexTrainer doesn't exist in installed ray.rllib.algorithms.dqn
        import ray.rllib.algorithms.dqn as dqn_mod
        if not hasattr(dqn_mod, 'ApexTrainer'):
            dqn_mod.ApexTrainer = MagicMock
        try:
            mod = self._import_module('train_rllib_apexdqn', ['--envtype', '1s-d'])
            assert hasattr(mod, 'config')
            assert isinstance(mod.config, dict)
            assert 'env' in mod.config
        except (ImportError, AttributeError):
            pytest.skip('ApexTrainer not available in installed Ray version')
        finally:
            if hasattr(dqn_mod, 'ApexTrainer') and dqn_mod.ApexTrainer is MagicMock:
                del dqn_mod.ApexTrainer

    def test_env_creator_returns_env(self):
        """env_creator function returns a FloPyEnv instance."""
        mod = self._import_module('train_rllib_ppo', ['--envtype', '1s-d'])
        env = mod.env_creator({})
        assert env is not None
        assert hasattr(env, 'ENVTYPE')
        env.close()

    def test_str_to_bool_in_rllib(self):
        mod = self._import_module('train_rllib_ppo')
        assert mod.str_to_bool('true') is True
        assert mod.str_to_bool('false') is False
        assert mod.str_to_bool('1') is True
        assert mod.str_to_bool('0') is False
        with pytest.raises(ValueError):
            mod.str_to_bool('invalid')


# ─── 6. play_againstai.py: config + test function ─────────────────────────────

class TestPlayAgainstAI:
    """Validate play_againstai config structure and test() function."""

    def _mock_ray_agents_dqn(self):
        """Pre-mock ray.rllib.agents.dqn for play_againstai import."""
        mock_dqn = MagicMock()
        mock_dqn.ApexTrainer = MagicMock
        sys.modules['ray.rllib.agents'] = MagicMock()
        sys.modules['ray.rllib.agents.dqn'] = mock_dqn

    def test_config_structure(self):
        self._mock_ray_agents_dqn()
        with patch.object(sys, 'argv', ['play_againstai.py', '--envtype', '1s-d', '--cpus', '2']):
            sys.modules.pop('flopyarcade.play_againstai', None)
            import importlib
            mod = importlib.import_module('flopyarcade.play_againstai')
            assert hasattr(mod, 'config')
            assert hasattr(mod, 'config_stopCriteria')
            assert 'seed' in mod.config
            assert 'model' in mod.config

    def test_env_creator(self):
        self._mock_ray_agents_dqn()
        with patch.object(sys, 'argv', ['play_againstai.py', '--envtype', '1s-d']):
            sys.modules.pop('flopyarcade.play_againstai', None)
            import importlib
            mod = importlib.import_module('flopyarcade.play_againstai')
            env = mod.env_creator({})
            assert env is not None
            env.close()

    def test_str_to_bool(self):
        self._mock_ray_agents_dqn()
        with patch.object(sys, 'argv', ['play_againstai.py']):
            sys.modules.pop('flopyarcade.play_againstai', None)
            import importlib
            mod = importlib.import_module('flopyarcade.play_againstai')
            assert mod.str_to_bool(True) is True
            assert mod.str_to_bool('yes') is True
            assert mod.str_to_bool('no') is False

    def test_playbenchmark_flag(self):
        """--playbenchmark True triggers the print statement."""
        self._mock_ray_agents_dqn()
        with patch.object(sys, 'argv', ['play_againstai.py', '--playbenchmark', 'True']):
            sys.modules.pop('flopyarcade.play_againstai', None)
            import importlib
            import io
            buf = io.StringIO()
            with patch('builtins.print', new=buf.write):
                importlib.import_module('flopyarcade.play_againstai')
            assert 'benchmark' in buf.getvalue().lower()
