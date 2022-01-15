from argparse import ArgumentParser
from ray import init as rayInit
from ray.tune.registry import register_env
from ray import tune
from flopyarcade import FloPyEnv
from os import makedirs
from os.path import abspath, dirname, exists, join

global ENVTYPE
ENVTYPE = '3s-d'


parser = ArgumentParser(description='FloPyArcade optimization using deep Q networks')
parser.add_argument('--cpus', default='1', type=int,
    help='integer defining number of cpus to use')
parser.add_argument('--gpus', default='0', type=int,
    help='integer defining number of cpus to use')
args = parser.parse_args()

wrkspc = abspath(dirname(__file__))

config_model = {
    "_use_default_native_models": False,
    "fcnet_hiddens": [512, 1024, 512],
    "fcnet_activation": "relu",
    "post_fcnet_hiddens": [],
    "post_fcnet_activation": "relu",
    "free_log_std": False,
    "no_final_linear": False,
    "vf_share_layers": True,
    "dim": 84,
    "grayscale": False,
    "zero_mean": True
    }

config = {
    'env': ENVTYPE,
    'seed': 1,
    'model': config_model,
    "num_workers": args.cpus-2,
    "num_gpus": args.gpus,
    "framework": "tf",
    "env": "my_env",
    "optimizer": {
            "max_weight_sync_delay": 40000,
            "num_replay_buffer_shards": 1,
            "debug": False
        },
    "n_step": 1,
    "buffer_size": 3000000,
    "learning_starts": 50000,
    "train_batch_size": 512,
    "rollout_fragment_length": 50,
    "target_network_update_freq": 100000,
    "timesteps_per_iteration": 10000,
    "exploration_config": {"type": "PerWorkerEpsilonGreedy"},
    "worker_side_prioritization": True,
    "min_iter_time_s": 30,
    "training_intensity": None,
    "lr": 0.00005,
    "evaluation_interval": 1,
    "evaluation_num_episodes": 1,
    "evaluation_num_workers": 0,
}

config_stopCriteria = {
    "training_iteration": 1000000000000,
    "timesteps_total": 1000000000000,
    "episode_reward_mean": 990,
}


def env_creator(env_config):
    env_config = {}
    env_config['ENVTYPE'] = ENVTYPE
    return FloPyEnv(config=env_config)


def test(agent, env, seed):
    """Test trained agent for a single episode. Return the episode reward"""
    # https://github.com/ray-project/ray/issues/9220
    
    observations = env.reset(_seed=seed)

    from matplotlib.pyplot import switch_backend
    switch_backend('TkAgg')
    env.RENDER = True

    reward_total = 0.
    while not env.done:
        action = agent.compute_single_action(observations)
        observations, reward, done, info = env.step(action)
        reward_total += reward
    
    return reward_total


if __name__ == "__main__":
    tune.registry.register_env('my_env', env_creator)
    rayInit()

    if not exists(join(wrkspc, 'temp')):
        makedirs(join(wrkspc, 'temp'))
    if not exists(join(wrkspc, 'temp', 'ray_results')):
        makedirs(join(wrkspc, 'temp', 'ray_results'))

    restore_path = join(wrkspc, 'temp', 'ray_results', 'APEX_my_env_2bce0_00000_0_2022-01-05_23-38-15', 'checkpoint_000540', 'checkpoint-540')
    results = tune.run(
            "APEX",
            name="2s-d-APEX-ffnn-varyseed-stopmean900-lr0-00005-bs3M",
            config=config,
            stop=config_stopCriteria,
            verbose=3,
            checkpoint_freq=1,
            checkpoint_at_end=True,
            reuse_actors=False,
            local_dir=join(wrkspc, 'temp', 'ray_results'),
            # restore=resore_path
            )