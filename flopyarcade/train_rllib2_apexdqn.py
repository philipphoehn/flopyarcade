import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser
from logging import FATAL
from numpy.random import randint
from os import makedirs
from os.path import abspath, dirname, exists, join
from ray import tune
from ray import init as rayInit
from ray.tune.registry import register_env
# with ray==1.9.2
# from ray.rllib.agents.dqn import ApexTrainer
# with ray==2.1.0
# from ray.rllib.algorithms.apex_dqn import apex_dqn as ApexTrainer
from ray.rllib.algorithms.apex_dqn import ApexDQN as ApexTrainer

# must be imported as else a Ray class cannot find the module later
wrkspc = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(wrkspc)
try:
    from flopyarcade.flopyarcade import FloPyEnv
except:
    from flopyarcade import FloPyEnv

# from tensorflow import get_logger
# get_logger().setLevel('ERROR')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', 'SelectableGroups dict interface')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = ArgumentParser(description='FloPyArcade optimization using deep Q networks')
parser.add_argument('--envtype', default='1s-d', type=str,
    help='string defining environment')
parser.add_argument('--suffix', default='', type=str,
    help='string defining environment')
parser.add_argument('--cpus', default='1', type=int,
    help='integer defining number of cpus to use')
parser.add_argument('--gpus', default='0', type=int,
    help='integer defining number of cpus to use')
parser.add_argument('--playbenchmark', default=False, type=str_to_bool,
    help='boolean to define if displaying runs')
parser.add_argument('--external', default=True, type=str_to_bool,
    help='boolean used as temporary helper during exection, can be ignored')
args = parser.parse_args()

if args.playbenchmark:
    if args.external:
        print('Starting benchmark visualization. Note that loading might take a moment.')

global ENVTYPE
ENVTYPE = args.envtype

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
    "num_workers": args.cpus-2 if args.cpus >= 3 else args.cpus, # weird, but 16 workers on 16 cores failed to work
    "num_gpus": args.gpus,
    "framework": "tf",
    "env": "my_env",
    "optimizer": {
            "max_weight_sync_delay": 40000,
            "num_replay_buffer_shards": 1,
            "debug": False
        },
    "n_step": 1,
    "replay_buffer_config": {
            "capacity": 3000000,
        },
    "num_steps_sampled_before_learning_starts": 50000,
    "train_batch_size": 512,
    "rollout_fragment_length": 50,
    "target_network_update_freq": 100000,
    "min_sample_timesteps_per_iteration": 10000,
    "exploration_config": {"type": "PerWorkerEpsilonGreedy"},
    # "worker_side_prioritization": True, # changed in ray==2.1.0?
    "min_time_s_per_iteration": 30,
    "training_intensity": None,
    "lr": 0.00005,
    "evaluation_interval": 1,
    "evaluation_num_episodes": 1,
    "evaluation_num_workers": 0,
    "log_level": "ERROR"
}

config_stopCriteria = {
    "training_iteration": 1000000000000,
    "timesteps_total": 1000000000000,
    "episode_reward_mean": 990,
}


def env_creator(env_config):
    # currently the argument env_config is not piped at creation
    # therefore not using the argument until solved

    env_config = {}
    env_config['ENVTYPE'] = ENVTYPE
    env = FloPyEnv(env_config=env_config)

    return env


def test(agent, env):
    """Test trained agent for a single episode. Return the episode reward"""
    # https://github.com/ray-project/ray/issues/9220
    
    # observations = env.reset(_seed=seed)
    observations = env.observationsVectorNormalized

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

    wrkspc = abspath(dirname(__file__))

    if not args.playbenchmark:
        rayInit()
    elif args.playbenchmark:
        # suppressing Ray messages and warnings here for comfort
        rayInit(log_to_driver=False, logging_level=FATAL)
    register_env('my_env', env_creator)

    if not exists(join(wrkspc, 'temp')):
        makedirs(join(wrkspc, 'temp'))
    if not exists(join(wrkspc, 'temp', 'ray_results')):
        makedirs(join(wrkspc, 'temp', 'ray_results'))

    if not args.playbenchmark:

        # restore_path = join(wrkspc, 'temp', 'ray_results', 'APEX_my_env_2bce0_00000_0_2022-01-05_23-38-15', 'checkpoint_000540', 'checkpoint-540')
        results = tune.run(
                "APEX",
                name=ENVTYPE + args.suffix,
                config=config,
                stop=config_stopCriteria,
                verbose=3,
                checkpoint_freq=1,
                checkpoint_at_end=True,
                reuse_actors=False,
                local_dir=join(wrkspc, 'temp', 'ray_results'),
                # restore=resore_path
                )

    elif args.playbenchmark:
        # external flag currently necessary as using -m is not working
        # as Tensorflow will raise an error of having received
        # an external symbolic tensor

        if not args.external:

            config = {
            "num_workers": 1,
            "num_gpus": 0,
            "framework": "tf",
            "env": "my_env",
            "optimizer": {
                    "max_weight_sync_delay": 400,
                    "num_replay_buffer_shards": 1,
                    "debug": False
                },
            "n_step": 3,
            "replay_buffer_config": {
                    "capacity": 5000000,
                },
            "num_steps_sampled_before_learning_starts": 50000,
            "train_batch_size": 512,
            "rollout_fragment_length": 50,
            "target_network_update_freq": 500000,
            "min_sample_timesteps_per_iteration": 50000,
            "exploration_config": {"type": "PerWorkerEpsilonGreedy"},
            # "worker_side_prioritization": True, # changed in ray==2.1.0?
            "min_time_s_per_iteration": 30,
            "training_intensity": None,
            "lr": 0.00005,
            }

            __name__ == '__main__'
            __package__ == None

            agent = ApexTrainer(config)
            print(join(wrkspc, 'flopyarcade', 'examples', 'policymodels', ENVTYPE, ENVTYPE))
            checkpoint_path = join(wrkspc, 'flopyarcade', 'examples', 'policymodels', ENVTYPE, ENVTYPE)
            # checkpoint_path = join(wrkspc, 'flopyarcade', 'examples', 'policymodels', ENVTYPE)
            agent.restore(checkpoint_path)

            done = False
            while not done:
                env = env_creator(env_config={'ENVTYPE': ENVTYPE})
                reward_total = test(agent, env)
                print('total reward', reward_total)

        if args.external:
            os.chdir(join(wrkspc))

            cmd = 'python train_rllib_apexdqn.py' +\
            ' --envtype ' + str(args.envtype) +\
            ' --cpus ' + str(args.cpus) +\
            ' --gpus ' + str(args.gpus) +\
            ' --playbenchmark ' + str(args.playbenchmark) +\
            ' --external ' + 'False'
            if str(args.suffix) != '':
                cmd = cmd + ' --suffix ' + str(args.suffix)

            os.system(cmd)