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
from ray.rllib.agents.dqn import ApexTrainer

# must be imported as else a Ray class cannot find the module later
wrkspc = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(wrkspc)
try:
    from flopyarcade.flopyarcade import FloPyEnv
    from flopyarcade.flopyarcade import FloPyAgent
except:
    from flopyarcade import FloPyEnv
    from flopyarcade import FloPyAgent

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
    
    import matplotlib.pyplot as plt
    from copy import deepcopy
    from matplotlib.pyplot import switch_backend
    from matplotlib.pyplot import get_current_fig_manager
    from time import sleep, time
    import matplotlib as mpl

    env_man = deepcopy(env)
    agent_man = FloPyAgent()

    switch_backend('Agg')

    # observations = env.reset(_seed=seed)
    observations_ai = env.observationsVectorNormalized
    observations_man = env_man.observationsVectorNormalized

    env_man.MANUALCONTROLTIME = 0.5
    # DPI = 700
    DPI = 100

    env.RENDER = False
    env_man.RENDER = False

    global keyPressed
    keyPressed = 'keep'

    reward_total = 0.
    i = 0

    # while not env.done and not env_man.done:
    while False in [env.done, env_man.done]:

        t0 = time()
        if i == 0:
            plt.cla()
            plt.clf()
            plt.close('all')

        # action_man = agent_man.getAction('manual', keyPressed, actionType=env_man.actionType)
        action_man = 'keep'
        if keyPressed in env_man.actionSpace:
            action_man = keyPressed
        # print('0', time()-t0)

        if not env.done:
            action_ai = agent.compute_single_action(observations_ai)
            # print('0.1', time()-t0)
            observations_ai, reward_ai, done_ai, info_ai = env.step(action_ai)
            # print('0.2', time()-t0)
        reward_total += reward_ai

        '''
        fig2 = plt.figure(constrained_layout=True)
        gs2 = fig2.add_gridspec(ncols=1, nrows=1)
        axtemp = fig2.add_subplot(gs2[0:1, 0:1])
        axtemp.imshow(render_ai)
        axtemp.axis('off')
        fig2.savefig('/home/phil/Downloads/tempimages/' + str(i).zfill(4) + '.png', dpi=DPI)
        plt.close(fig2)
        '''

        # print('1', time()-t0)

        if not env_man.done:
            observations_man, reward_man, done_man, info_man = env_man.step(action_man)
            # print('1.1', time()-t0)

        # rendering individually
        # render_ai = env.render(mode='rgb_array', dpi=DPI)
        # render_man = env_man.render(mode='rgb_array', dpi=DPI)
        # print('2', time()-t0)

        # rendering with multiple processes
        def render_env(env, DPI=100):
            render_result = env.render(mode='rgb_array', dpi=DPI)
            return render_result
        render_results = agent_man.multiprocessChunks_OLD(render_env, [env, env_man], parallelProcesses=2, threadPool=True)
        render_ai, render_man = render_results[0], render_results[1]

        if i == 0:
            switch_backend('TkAgg')
            mpl.rcParams['toolbar'] = 'None'

            fig = plt.figure(constrained_layout=True)
            widths = [1, 1]
            heights = [3, 1]
            gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
            ax1 = fig.add_subplot(gs[0:1, 0:1])
            ax2 = fig.add_subplot(gs[0:1, 1:2])
            ax3 = fig.add_subplot(gs[1:2, 0:1])
            ax4 = fig.add_subplot(gs[1:2, 1:2])

            im_neuralnetwork = join(wrkspc, 'flopyarcade', 'examples', 'images', 'action_neuralnetwork.png')
            im_human = join(wrkspc, 'flopyarcade', 'examples', 'images', 'action_human.png')

            im_neuralnetwork = mpl.image.imread(im_neuralnetwork)
            im_human = mpl.image.imread(im_human)
            ax3.imshow(im_neuralnetwork)
            ax4.imshow(im_human)

            ax3.axis('off')
            ax4.axis('off')

            figManager = get_current_fig_manager()
            maximized = False
            try:
                figManager.window.state('zoomed')
                maximized = True
            except:
                pass
            if not maximized:
                try:
                    figManager.full_screen_toggle()
                except:
                    pass
            plt.tight_layout()
        # print('3', time()-t0)

        i += 1

        ax1.cla()
        ax2.cla()
        ax1.axis('off')
        ax2.axis('off')
        # print('4', time()-t0)

        ax1.imshow(render_ai)
        ax2.imshow(render_man)
        # print('5', time()-t0)

        def captureKeyPress(event):
            """Capture key pressed through manual user interaction."""
            global keyPressed
            keyPressed = event.key

        fig.canvas.mpl_connect('key_press_event', captureKeyPress)
        # print('post action_man', action_man, keyPressed)

        # reset key
        keyPressed = 'keep'

        # print('env_man.MANUALCONTROLTIME', env_man.MANUALCONTROLTIME)

        plt.show(block=False)
        plt.waitforbuttonpress(timeout=(env_man.MANUALCONTROLTIME))
        plt.pause(env_man.MANUALCONTROLTIME)
        # print('6', time()-t0)

        # print('i', i, env.done, env_man.done)
        # print('rendering frame', i+1, 'took', time()-t0, 's')

    # print('reward_total', reward_total)
    
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
            "buffer_size": 5000000,
            "learning_starts": 50000,
            "train_batch_size": 512,
            "rollout_fragment_length": 50,
            "target_network_update_freq": 500000,
            "timesteps_per_iteration": 50000,
            "exploration_config": {"type": "PerWorkerEpsilonGreedy"},
            "worker_side_prioritization": True,
            "min_iter_time_s": 30,
            "training_intensity": None,
            "lr": 0.00005,
            }

            __name__ == '__main__'
            __package__ == None

            agent = ApexTrainer(config)
            # checkpoint_path = join(wrkspc, 'examples', 'policymodels', ENVTYPE, ENVTYPE)
            checkpoint_path = join(wrkspc, 'flopyarcade', 'examples', 'policymodels', ENVTYPE, ENVTYPE)
            agent.restore(checkpoint_path)

            totals = []
            done = False
            while not done:
                env = env_creator(env_config={'ENVTYPE': ENVTYPE})
                reward_total = test(agent, env)
                totals.append(reward_total)
                import numpy as np
                print('total reward', reward_total, np.mean(totals))

        if args.external:
            os.chdir(join(wrkspc))

            cmd = 'python play_againstai.py' +\
            ' --envtype ' + str(args.envtype) +\
            ' --cpus ' + str(args.cpus) +\
            ' --gpus ' + str(args.gpus) +\
            ' --playbenchmark ' + str(args.playbenchmark) +\
            ' --external ' + 'False'
            if str(args.suffix) != '':
                cmd = cmd + ' --suffix ' + str(args.suffix)

            os.system(cmd)