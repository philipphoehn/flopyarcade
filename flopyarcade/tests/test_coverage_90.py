
"""Coverage tests v3 - generated."""
import pytest, numpy as np, os, sys, tempfile, pickle, json, shutil
from unittest.mock import MagicMock, patch
from collections import defaultdict
from itertools import count

class TestBench:
    def test_try_layer_get(self):
        from flopyarcade.benchmark import _try_layer_get
        p = {'fc_1/kernel':np.array([[1.0]]),'fc_1/bias':np.array([1.0])}
        k,b = _try_layer_get(p,'','fc_1')
        assert k is not None
    def test_try_layer_get_nf(self):
        from flopyarcade.benchmark import _try_layer_get
        k,b = _try_layer_get({'x':1},'','y')
        assert k is None
    def test_load_keras_agent_pickle(self):
        from flopyarcade.benchmark import load_keras_agent
        with patch('flopyarcade.benchmark._load_rllib_pickle',return_value=MagicMock()):
            load_keras_agent('/x.p')
    def test_load_keras_agent_dir_empty(self):
        from flopyarcade.benchmark import load_keras_agent
        d=tempfile.mkdtemp()
        try:
            with pytest.raises(FileNotFoundError): load_keras_agent(d)
        finally: shutil.rmtree(d)
    def test_run_benchmark(self):
        from flopyarcade.benchmark import run_benchmark
        with patch('flopyarcade.benchmark.load_keras_agent',return_value=MagicMock()) as lm,              patch('flopyarcade.FloPyArcade') as FA,              patch('flopyarcade.FloPyAgent'):
            FA.return_value.play.return_value=1.0
            r=run_benchmark('/m.keras',envtype='1s-d',n_games=1,verbose=False)
            assert 'mean_reward' in r
    def test_main(self):
        from flopyarcade.benchmark import main
        with patch('sys.argv',['b','--model','/m.keras','--games','1','--quiet']),              patch('flopyarcade.benchmark.run_benchmark',return_value={'mean_reward':1}): 
            main()

class TestAgent:
    def _a(self):
        from flopyarcade.agent import FloPyAgent
        a=object.__new__(FloPyAgent)
        a.hyParams={'EPSILONMIN':0.01,'EPSILONDECAY':0.95,'NAGENTS':4,'NNOVELTYELITES':1,'NNOVELTYNEIGHBORS':2,'NGAMESAVERAGED':1,'NAGENTELITES':2}
        a.epsilon=0.5; a.envSettings={'SEEDSRANDOM':False,'NAGENTSPARALLEL':1}
        a.agentsUnique=[0,1]; a.agentsUniqueIDs=[1,2]; a.agentsDuplicate=[2]
        a.noveltyItemCount=3; a.noveltyArchive={'agent1':{'actionsUniqueID':1},'agent2':{'actionsUniqueID':2},'agent3':{'actionsUniqueID':1}}
        a.zFill=4; a.rewards=[10,20,30,40]; a.tempModelPrefix='/tmp/t'
        a.tempPrevModelPrefix='/tmp/p'; a.actionsUniqueIDMapping=defaultdict(count().__next__)
        a.searchNovelty=False; a.cores=1; a.maxTasksPerWorkerNoveltySearch=100
        return a
    def test_decay_eps(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); FloPyAgent._decay_epsilon(a); assert a.epsilon<0.5
    def test_decay_eps_min(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); a.epsilon=0.005; FloPyAgent._decay_epsilon(a); assert a.epsilon==0.005
    def test_validate_hyp(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); FloPyAgent._validate_hyperparams(a)
    def test_validate_hyp_err(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); a.hyParams['NAGENTS']=1; a.hyParams['NNOVELTYELITES']=2
        with pytest.raises(ValueError): FloPyAgent._validate_hyperparams(a)
    def test_init_env(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); env=MagicMock(); env.actionType='discrete'
        FloPyAgent._initialize_environment(a,env,True)
        assert a.env is env and a.searchNovelty==True
    def test_init_parallel(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); FloPyAgent._initialize_parallel_settings(a); assert a.cores==1
    def test_gen_pid(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); FloPyAgent._generate_unique_pid(a); assert len(a.pid)==36
    def test_get_action_type_d(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); assert FloPyAgent.getActionType(a,'1s-d')=='discrete'
    def test_get_action_type_c(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); assert FloPyAgent.getActionType(a,'1s-c')=='continuous'
    def test_log_gen_rewards(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a()
        with patch('flopyarcade.agent.logger') as lg:
            FloPyAgent._log_generation_rewards(a)
    def test_sort_save_top(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); a.hyParams['NAGENTELITES']=2
        with patch.object(FloPyAgent,'pickleDump') as pd:
            r=FloPyAgent._sort_and_save_top_agents(a)
            assert len(r)==2
    def test_update_discrete(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); a.agentsUniqueIDs=[]; a.agentsUnique=[]; a.agentsDuplicate=[]
        a.noveltyItemCount=0
        FloPyAgent._update_discrete_uniques(a, 99)
        assert 99 in a.agentsUniqueIDs
    def test_reset_cont(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); a.noveltyItemCount=2
        FloPyAgent._reset_continuous_uniques(a)
        assert len(a.agentsUnique)==3
    def test_print_novelty(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a()
        FloPyAgent._print_novelty_stats(a)
    def test_build_novelty_args(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a()
        args,flags=FloPyAgent._build_novelty_args(a)
        assert len(args)==2
    def test_novelty_search_skip(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); a.searchNovelty=False
        FloPyAgent.perform_novelty_search(a, MagicMock())
    def test_novelty_search_run(self):
        from flopyarcade.agent import FloPyAgent
        a=self._a(); a.searchNovelty=True
        env=MagicMock(); env.actionType='discrete'
        with patch.object(FloPyAgent,'_prepare_novelty_archive'),              patch.object(FloPyAgent,'_print_novelty_stats'),              patch.object(FloPyAgent,'_build_novelty_args',return_value=([],[True])),              patch.object(FloPyAgent,'_run_novelty_search',return_value=[1.0]),              patch.object(FloPyAgent,'_update_agent_novelties'),              patch.object(FloPyAgent,'_save_novelties'):
            FloPyAgent.perform_novelty_search(a,env)
