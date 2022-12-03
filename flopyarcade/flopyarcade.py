import numpy as np
import os
import pickle
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

def relu(x):
    import numpy as np
    return np.maximum(0, x)

def calc_action(net, obs):

    # computed via self.compute_actions_from_input_dict in ray.rllib.policy.policy
    # takes argmax of q values

    # compare to q values from rllib in ray.rllib.policy.policy in compute_single_action method
    # print(out[2]['action_dist_inputs'], 'out')

    out = np.array(obs)

    # base model
    out = np.dot(out, net['default_policy/fc_1/kernel'])
    out = np.add(out, net['default_policy/fc_1/bias'])
    out = np.tanh(out)
    out = np.dot(out, net['default_policy/fc_out/kernel'])
    out = np.add(out, net['default_policy/fc_out/bias'])
    out_model = np.tanh(out)

    # model 1: advantage value model
    out = np.dot(out_model, net['default_policy/hidden_0/kernel'])
    out = np.add(out, net['default_policy/hidden_0/bias'])
    out_hidden0 = relu(out)
    out = np.dot(out_hidden0, net['default_policy/dense/kernel'])
    out_model1 = np.add(out, net['default_policy/dense/bias'])

    # model 2: state value model
    out = np.dot(out_model, net['default_policy/dense_1/kernel'])
    out = np.add(out, net['default_policy/dense_1/bias'])
    out = relu(out)
    out = np.dot(out, net['default_policy/dense_2/kernel'])
    out_model2 = np.add(out, net['default_policy/dense_2/bias'])

    # aggregation
    # results in actual q values (i.e., expected return?)
    out = np.add(out_model1, out_model2)
    qvalues = np.subtract(out, np.mean(out_model1))

    action = np.argmax(qvalues)

    return action, qvalues


def test(agent, env):
    """Test trained agent for a single episode. Return the episode reward"""
    # https://github.com/ray-project/ray/issues/9220


    # observations = env.reset(_seed=seed)
    # Note that complex observations must be preprocessed.
    # No observations preprocessing necessary here.
    # An example of preprocessing is examples/saving_experiences.py
    observations = np.array(env.observationsVectorNormalized)

    from matplotlib.pyplot import switch_backend
    switch_backend('TkAgg')
    env.RENDER = True

    # to inspect graph via tensorboard
    # in tf_policy near creation of builder object
    # import tensorflow as tf
    # writer = tf.compat.v1.summary.FileWriter('D:\\tflogs', self.get_session().graph)

    # weights_as_np = agent.get_weights()
    print(env.ENVTYPE)
    with open('D:\\model_' + env.ENVTYPE + '.p', 'rb') as f:
        weights_as_np = pickle.load(f)

    keys = weights_as_np['default_policy'].keys()
    # print(keys)
    for key in keys:
        _shape = np.shape(weights_as_np['default_policy'][key])
        # print(key, _shape)


    # https://docs.ray.io/en/latest/rllib/rllib-training.html

    policy = agent.get_policy()
    model = policy.model
    model_out = model({"obs": observations.reshape(-1, (*np.shape(observations)))})
    # print(model.variables())

    logits, _ = policy.model({"obs": observations.reshape(-1, (*np.shape(observations)))})
    # print(logits)

    # print(model.base_model.summary())
    # base model
    # __________________________________________________________________________________________________
    #  Layer (type)                   Output Shape         Param #     Connected to
    # ==================================================================================================
    #  observations (InputLayer)      [(None, 1165)]       0           []
    #  fc_1 (Dense)                   (None, 256)          298496      ['observations[0][0]']
    #  fc_out (Dense)                 (None, 256)          65792       ['fc_1[0][0]']
    #  value_out (Dense)              (None, 1)            257         ['fc_1[0][0]']
    # ==================================================================================================
    # Total params: 364,545
    # Trainable params: 364,545
    # Non-trainable params: 0

    # q and state value model, specific to DQN
    # print(model.q_value_head.summary())
    # print(model.get_state_value(model_out))

    reward_total = 0.
    while not env.done:
        print('----------------')

        action = agent.compute_single_action(observations)

        # performing forward pass of Dueling DQN explained:
        # http://www.sefidian.com/2021/09/10/double-dqn-and-dueling-dqn-in-reinforcement-learning/
        action_manual, out_manual = calc_action(weights_as_np['default_policy'], np.array(observations))
        print(action, action_manual, 'computed action', 'manually-computed action')

        action_str = env.actionSpace[action]
        print('action_str', action_str, env.actionSpace, action)

        file = 'D:\\Desktop_WINDOWS\\thesis\\presentation\\Disseration_Defensio\\images\\arcade_machine_edited_clipped_' + action_str + '.png'
        print(file)

        observations, reward, done, info = env.step(action)
        # observations_prep = norm_obs(env.observationsVectorNormalized)
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

            # get_config()
            # {'num_workers': 1, 'num_envs_per_worker': 1, 'create_env_on_driver': False, 'rollout_fragment_length': 50, 'batch_mode': 'truncate_episodes', 'gamma': 0.99, 'lr': 5e-05, 'train_batch_size': 512, 'model': {'_use_default_native_models': False, '_disable_preprocessor_api': False, 'fcnet_hiddens': [256, 256], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': None, 'custom_model_config': {}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}, 'optimizer': {'max_weight_sync_delay': 400, 'num_replay_buffer_shards': 1, 'debug': False}, 'horizon': None, 'soft_horizon': False, 'no_done_at_end': False, 'env': 'my_env', 'observation_space': None, 'action_space': None, 'env_config': {}, 'remote_worker_envs': False, 'remote_env_batch_wait_ms': 0, 'env_task_fn': None, 'render_env': False, 'record_env': False, 'clip_rewards': None, 'normalize_actions': True, 'clip_actions': False, 'preprocessor_pref': 'deepmind', 'log_level': 'WARN', 'callbacks': <class 'ray.rllib.agents.callbacks.DefaultCallbacks'>, 'ignore_worker_failures': False, 'log_sys_usage': True, 'fake_sampler': False, 'framework': 'tf', 'eager_tracing': False, 'eager_max_retraces': 20, 'explore': True, 'exploration_config': {'type': 'PerWorkerEpsilonGreedy', 'initial_epsilon': 1.0, 'final_epsilon': 0.02, 'epsilon_timesteps': 10000}, 'evaluation_interval': None, 'evaluation_num_episodes': 10, 'evaluation_parallel_to_training': False, 'in_evaluation': False, 'evaluation_config': {'explore': False}, 'evaluation_num_workers': 0, 'custom_eval_function': None, 'sample_async': False, 'sample_collector': <class 'ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector'>, 'observation_filter': 'NoFilter', 'synchronize_filters': True, 'tf_session_args': {'intra_op_parallelism_threads': 2, 'inter_op_parallelism_threads': 2, 'gpu_options': {'allow_growth': True}, 'log_device_placement': False, 'device_count': {'CPU': 1}, 'allow_soft_placement': True}, 'local_tf_session_args': {'intra_op_parallelism_threads': 8, 'inter_op_parallelism_threads': 8}, 'compress_observations': False, 'collect_metrics_timeout': 180, 'metrics_smoothing_episodes': 100, 'min_iter_time_s': 30, 'timesteps_per_iteration': 50000, 'seed': None, 'extra_python_environs_for_driver': {}, 'extra_python_environs_for_worker': {}, 'num_gpus': 0, '_fake_gpus': False, 'num_cpus_per_worker': 1, 'num_gpus_per_worker': 0, 'custom_resources_per_worker': {}, 'num_cpus_for_driver': 1, 'placement_strategy': 'PACK', 'input': 'sampler', 'input_config': {}, 'actions_in_input_normalized': False, 'input_evaluation': ['is', 'wis'], 'postprocess_inputs': False, 'shuffle_buffer_size': 0, 'output': None, 'output_compress_columns': ['obs', 'new_obs'], 'output_max_file_size': 67108864, 'multiagent': {'policies': {'default_policy': PolicySpec(policy_class=<class 'ray.rllib.policy.tf_policy_template.DQNTFPolicy'>, observation_space=None, action_space=None, config={})}, 'policy_map_capacity': 100, 'policy_map_cache': None, 'policy_mapping_fn': None, 'policies_to_train': None, 'observation_fn': None, 'replay_mode': 'independent', 'count_steps_by': 'env_steps'}, 'logger_config': None, '_tf_policy_handles_more_than_one_loss': False, '_disable_preprocessor_api': False, 'simple_optimizer': False, 'monitor': -1, 'target_network_update_freq': 500000, 'buffer_size': 5000000, 'replay_buffer_config': None, 'store_buffer_in_checkpoints': False, 'replay_sequence_length': 1, 'lr_schedule': None, 'adam_epsilon': 1e-08, 'grad_clip': 40, 'learning_starts': 50000, 'num_atoms': 1, 'v_min': -10.0, 'v_max': 10.0, 'noisy': False, 'sigma0': 0.5, 'dueling': True, 'hiddens': [256], 'double_q': True, 'n_step': 3, 'prioritized_replay': True, 'prioritized_replay_alpha': 0.6, 'prioritized_replay_beta': 0.4, 'final_prioritized_replay_beta': 0.4, 'prioritized_replay_beta_annealing_timesteps': 20000, 'prioritized_replay_eps': 1e-06, 'before_learn_on_batch': None, 'training_intensity': None, 'worker_side_prioritization': True}

            # _disable_preprocessor_api: false
            # _fake_gpus: false
            # _tf_policy_handles_more_than_one_loss: false
            # action_space: null
            # actions_in_input_normalized: false
            # adam_epsilon: 1.0e-08
            # batch_mode: truncate_episodes
            # before_learn_on_batch: null
            # buffer_size: 5000000
            # callbacks: !!python/name:ray.rllib.agents.callbacks.DefaultCallbacks ''
            # clip_actions: false
            # clip_rewards: null
            # collect_metrics_timeout: 180
            # compress_observations: false
            # create_env_on_driver: false
            # custom_eval_function: null
            # custom_resources_per_worker: {}
            # double_q: true
            # dueling: true
            # eager_max_retraces: 20
            # eager_tracing: false
            # env: my_env
            # env_config: {}
            # env_task_fn: null
            # evaluation_config:
            #   explore: false
            # evaluation_interval: null
            # evaluation_num_episodes: 10
            # evaluation_num_workers: 0
            # evaluation_parallel_to_training: false
            # exploration_config:
            #   epsilon_timesteps: 10000
            #   final_epsilon: 0.02
            #   initial_epsilon: 1.0
            #   type: PerWorkerEpsilonGreedy
            # explore: true
            # extra_python_environs_for_driver: {}
            # extra_python_environs_for_worker: {}
            # fake_sampler: false
            # final_prioritized_replay_beta: 0.4
            # framework: tf
            # gamma: 0.99
            # grad_clip: 40
            # hiddens:
            # - 256
            # horizon: null
            # ignore_worker_failures: false
            # in_evaluation: false
            # input: sampler
            # input_config: {}
            # input_evaluation:
            # - is
            # - wis
            # learning_starts: 50000
            # local_tf_session_args:
            #   inter_op_parallelism_threads: 8
            #   intra_op_parallelism_threads: 8
            # log_level: WARN
            # log_sys_usage: true
            # logger_config: null
            # lr: 5.0e-05
            # lr_schedule: null
            # metrics_smoothing_episodes: 100
            # min_iter_time_s: 30
            # model:
            #   _disable_preprocessor_api: false
            #   _time_major: false
            #   _use_default_native_models: false
            #   attention_dim: 64
            #   attention_head_dim: 32
            #   attention_init_gru_gate_bias: 2.0
            #   attention_memory_inference: 50
            #   attention_memory_training: 50
            #   attention_num_heads: 1
            #   attention_num_transformer_units: 1
            #   attention_position_wise_mlp_dim: 32
            #   attention_use_n_prev_actions: 0
            #   attention_use_n_prev_rewards: 0
            #   conv_activation: relu
            #   conv_filters: null
            #   custom_action_dist: null
            #   custom_model: null
            #   custom_model_config: {}
            #   custom_preprocessor: null
            #   dim: 84
            #   fcnet_activation: tanh
            #   fcnet_hiddens:
            #   - 256
            #   - 256
            #   framestack: true
            #   free_log_std: false
            #   grayscale: false
            #   lstm_cell_size: 256
            #   lstm_use_prev_action: false
            #   lstm_use_prev_action_reward: -1
            #   lstm_use_prev_reward: false
            #   max_seq_len: 20
            #   no_final_linear: false
            #   post_fcnet_activation: relu
            #   post_fcnet_hiddens: []
            #   use_attention: false
            #   use_lstm: false
            #   vf_share_layers: true
            #   zero_mean: true
            # monitor: -1
            # multiagent:
            #   count_steps_by: env_steps
            #   observation_fn: null
            #   policies:
            #     default_policy: !!python/object/new:ray.rllib.policy.policy.PolicySpec
            #     - !!python/name:ray.rllib.policy.tf_policy_template.DQNTFPolicy ''
            #     - null
            #     - null
            #     - {}
            #   policies_to_train: null
            #   policy_map_cache: null
            #   policy_map_capacity: 100
            #   policy_mapping_fn: null
            #   replay_mode: independent
            # n_step: 3
            # no_done_at_end: false
            # noisy: false
            # normalize_actions: true
            # num_atoms: 1
            # num_cpus_for_driver: 1
            # num_cpus_per_worker: 1
            # num_envs_per_worker: 1
            # num_gpus: 0
            # num_gpus_per_worker: 0
            # num_workers: 1
            # observation_filter: NoFilter
            # observation_space: null
            # optimizer:
            #   debug: false
            #   max_weight_sync_delay: 400
            #   num_replay_buffer_shards: 1
            # output: null
            # output_compress_columns:
            # - obs
            # - new_obs
            # output_max_file_size: 67108864
            # placement_strategy: PACK
            # postprocess_inputs: false
            # preprocessor_pref: deepmind
            # prioritized_replay: true
            # prioritized_replay_alpha: 0.6
            # prioritized_replay_beta: 0.4
            # prioritized_replay_beta_annealing_timesteps: 20000
            # prioritized_replay_eps: 1.0e-06
            # record_env: false
            # remote_env_batch_wait_ms: 0
            # remote_worker_envs: false
            # render_env: false
            # replay_buffer_config: null
            # replay_sequence_length: 1
            # rollout_fragment_length: 50
            # sample_async: false
            # sample_collector: !!python/name:ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector ''
            # seed: null
            # shuffle_buffer_size: 0
            # sigma0: 0.5
            # simple_optimizer: false
            # soft_horizon: false
            # store_buffer_in_checkpoints: false
            # synchronize_filters: true
            # target_network_update_freq: 500000
            # tf_session_args:
            #   allow_soft_placement: true
            #   device_count:
            #     CPU: 1
            #   gpu_options:
            #     allow_growth: true
            #   inter_op_parallelism_threads: 2
            #   intra_op_parallelism_threads: 2
            #   log_device_placement: false
            # timesteps_per_iteration: 50000
            # train_batch_size: 512
            # training_intensity: null
            # v_max: 10.0
            # v_min: -10.0
            # worker_side_prioritization: true

            # get_default_config()
            # {'num_workers': 32, 'num_envs_per_worker': 1, 'create_env_on_driver': False, 'rollout_fragment_length': 50, 'batch_mode': 'truncate_episodes', 'gamma': 0.99, 'lr': 0.0005, 'train_batch_size': 512, 'model': {'_use_default_native_models': False, '_disable_preprocessor_api': False, 'fcnet_hiddens': [256, 256], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': None, 'custom_model_config': {}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}, 'optimizer': {'max_weight_sync_delay': 400, 'num_replay_buffer_shards': 4, 'debug': False}, 'horizon': None, 'soft_horizon': False, 'no_done_at_end': False, 'env': None, 'observation_space': None, 'action_space': None, 'env_config': {}, 'remote_worker_envs': False, 'remote_env_batch_wait_ms': 0, 'env_task_fn': None, 'render_env': False, 'record_env': False, 'clip_rewards': None, 'normalize_actions': True, 'clip_actions': False, 'preprocessor_pref': 'deepmind', 'log_level': 'WARN', 'callbacks': <class 'ray.rllib.agents.callbacks.DefaultCallbacks'>, 'ignore_worker_failures': False, 'log_sys_usage': True, 'fake_sampler': False, 'framework': 'tf', 'eager_tracing': False, 'eager_max_retraces': 20, 'explore': True, 'exploration_config': {'type': 'PerWorkerEpsilonGreedy', 'initial_epsilon': 1.0, 'final_epsilon': 0.02, 'epsilon_timesteps': 10000}, 'evaluation_interval': None, 'evaluation_num_episodes': 10, 'evaluation_parallel_to_training': False, 'in_evaluation': False, 'evaluation_config': {'explore': False}, 'evaluation_num_workers': 0, 'custom_eval_function': None, 'sample_async': False, 'sample_collector': <class 'ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector'>, 'observation_filter': 'NoFilter', 'synchronize_filters': True, 'tf_session_args': {'intra_op_parallelism_threads': 2, 'inter_op_parallelism_threads': 2, 'gpu_options': {'allow_growth': True}, 'log_device_placement': False, 'device_count': {'CPU': 1}, 'allow_soft_placement': True}, 'local_tf_session_args': {'intra_op_parallelism_threads': 8, 'inter_op_parallelism_threads': 8}, 'compress_observations': False, 'collect_metrics_timeout': 180, 'metrics_smoothing_episodes': 100, 'min_iter_time_s': 30, 'timesteps_per_iteration': 25000, 'seed': None, 'extra_python_environs_for_driver': {}, 'extra_python_environs_for_worker': {}, 'num_gpus': 1, '_fake_gpus': False, 'num_cpus_per_worker': 1, 'num_gpus_per_worker': 0, 'custom_resources_per_worker': {}, 'num_cpus_for_driver': 1, 'placement_strategy': 'PACK', 'input': 'sampler', 'input_config': {}, 'actions_in_input_normalized': False, 'input_evaluation': ['is', 'wis'], 'postprocess_inputs': False, 'shuffle_buffer_size': 0, 'output': None, 'output_compress_columns': ['obs', 'new_obs'], 'output_max_file_size': 67108864, 'multiagent': {'policies': {}, 'policy_map_capacity': 100, 'policy_map_cache': None, 'policy_mapping_fn': None, 'policies_to_train': None, 'observation_fn': None, 'replay_mode': 'independent', 'count_steps_by': 'env_steps'}, 'logger_config': None, '_tf_policy_handles_more_than_one_loss': False, '_disable_preprocessor_api': False, 'simple_optimizer': -1, 'monitor': -1, 'target_network_update_freq': 500000, 'buffer_size': 2000000, 'replay_buffer_config': None, 'store_buffer_in_checkpoints': False, 'replay_sequence_length': 1, 'lr_schedule': None, 'adam_epsilon': 1e-08, 'grad_clip': 40, 'learning_starts': 50000, 'num_atoms': 1, 'v_min': -10.0, 'v_max': 10.0, 'noisy': False, 'sigma0': 0.5, 'dueling': True, 'hiddens': [256], 'double_q': True, 'n_step': 3, 'prioritized_replay': True, 'prioritized_replay_alpha': 0.6, 'prioritized_replay_beta': 0.4, 'final_prioritized_replay_beta': 0.4, 'prioritized_replay_beta_annealing_timesteps': 20000, 'prioritized_replay_eps': 1e-06, 'before_learn_on_batch': None, 'training_intensity': None, 'worker_side_prioritization': True}

            __name__ == '__main__'
            __package__ == None

            agent = ApexTrainer(config)
            checkpoint_path = join(wrkspc, 'flopyarcade', 'examples', 'policymodels', ENVTYPE, ENVTYPE)
            agent.restore(checkpoint_path)

            # import yaml
            # c = agent.get_config()
            # print(yaml.dump(c, default_flow_style=False))

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
