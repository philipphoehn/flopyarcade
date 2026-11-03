#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - per-ENVTYPE configuration
# Replaces 55+ ENVTYPE if/elif chains with a single lookup table.

ENV_CONFIG = {
    '0s-c': {'n': 0, 'action_discrete': None, 'action_continuous': ['up', 'down'], 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_1', 'helper_keys': None},
    '1s-d': {'n': 1, 'action_discrete': ['down', 'up', 'keep'], 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_1', 'helper_keys': None},
    '1s-c': {'n': 1, 'action_discrete': None, 'action_continuous': ['up', 'down'], 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_1', 'helper_keys': None},
    '1r-d': {'n': 1, 'action_discrete': ['down', 'up', 'keep'], 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_1', 'helper_keys': None},
    '1r-c': {'n': 1, 'action_discrete': None, 'action_continuous': ['up', 'down'], 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_1', 'helper_keys': None},
    '2s-d': {'n': 2, 'action_discrete': ['down', 'up', 'keep'], 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_2', 'helper_keys': None},
    '2s-c': {'n': 2, 'action_discrete': None, 'action_continuous': ['up', 'down'], 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_2', 'helper_keys': None},
    '2r-d': {'n': 2, 'action_discrete': ['down', 'up', 'keep'], 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_2', 'helper_keys': None},
    '2r-c': {'n': 2, 'action_discrete': None, 'action_continuous': ['up', 'down'], 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_2', 'helper_keys': None},
    '3s-d': {'n': 3, 'action_discrete': ['right', 'up', 'keep', 'down', 'left'], 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_3', 'helper_keys': None},
    '3s-c': {'n': 3, 'action_discrete': None, 'action_continuous': ['up', 'down', 'left', 'right'], 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_3', 'helper_keys': None},
    '3r-d': {'n': 3, 'action_discrete': ['right', 'up', 'keep', 'down', 'left'], 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_3', 'helper_keys': None},
    '3r-c': {'n': 3, 'action_discrete': None, 'action_continuous': ['up', 'down', 'left', 'right'], 'multi_discrete': None, 'multi_continuous': None, 'action_range': None, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_3', 'helper_keys': None},
    '4s-d': {'n': 4, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': ['up', 'keep', 'down', 'left', 'right'], 'multi_continuous': None, 'action_range': 2.5, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['X', 'Y']},
    '4s-c': {'n': 4, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': ['up', 'down', 'left', 'right'], 'action_range': 2.5, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['X', 'Y']},
    '4r-d': {'n': 4, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': ['up', 'keep', 'down', 'left', 'right'], 'multi_continuous': None, 'action_range': 2.5, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['X', 'Y']},
    '4r-c': {'n': 4, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': ['up', 'down', 'left', 'right'], 'action_range': 2.5, 'action_range_q': None, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['X', 'Y']},
    '5s-d': {'n': 5, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': ['moreQ', 'keepQ', 'lessQ'], 'multi_continuous': None, 'action_range': None, 'action_range_q': 100.0, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['Q']},
    '5s-c': {'n': 5, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': ['moreQ', 'lessQ'], 'action_range': None, 'action_range_q': 100.0, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['Q']},
    '5s-c-cost': {'n': 5, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': ['moreQ', 'lessQ'], 'action_range': None, 'action_range_q': 100.0, 'reward': 'operation_cost', 'state_update': '_update_helper_wells', 'helper_keys': ['Q']},
    '5r-d': {'n': 5, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': ['moreQ', 'keepQ', 'lessQ'], 'multi_continuous': None, 'action_range': None, 'action_range_q': 100.0, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['Q']},
    '5r-c': {'n': 5, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': ['moreQ', 'lessQ'], 'action_range': None, 'action_range_q': 100.0, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['Q']},
    '6s-d': {'n': 6, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': ['up', 'keep', 'down', 'left', 'right', 'moreQ', 'keepQ', 'lessQ'], 'multi_continuous': None, 'action_range': 2.5, 'action_range_q': 100.0, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['X', 'Y', 'Q']},
    '6s-c': {'n': 6, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': ['up', 'down', 'left', 'right', 'moreQ', 'lessQ'], 'action_range': 2.5, 'action_range_q': 50.0, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['X', 'Y', 'Q']},
    '6r-d': {'n': 6, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': ['up', 'keep', 'down', 'left', 'right', 'moreQ', 'keepQ', 'lessQ'], 'multi_continuous': None, 'action_range': 2.5, 'action_range_q': 100.0, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['X', 'Y', 'Q']},
    '6r-c': {'n': 6, 'action_discrete': None, 'action_continuous': None, 'multi_discrete': None, 'multi_continuous': ['up', 'down', 'left', 'right', 'moreQ', 'lessQ'], 'action_range': 2.5, 'action_range_q': 50.0, 'reward': 'trajectory', 'state_update': '_update_helper_wells', 'helper_keys': ['X', 'Y', 'Q']},
}

REWARD_DISPATCH = {
    'trajectory': 'calculateGameRewardTrajectory',
    'operation_cost': 'calculateGameRewardOperationCost',
}

__all__ = ['ENV_CONFIG', 'REWARD_DISPATCH']
