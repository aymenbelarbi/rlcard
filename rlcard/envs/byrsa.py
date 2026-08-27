'''RLCard environment for BYRSA -- the third harness for 02 E-12.

See rlcard/games/byrsa/game.py for the two leak modes and why they matter.
'''
from collections import OrderedDict

import numpy as np

from byrsa_sim import action_space
from byrsa_sim.cards import CARD_VALUE
from rlcard.envs import Env
from rlcard.games.byrsa import Game

DEFAULT_GAME_CONFIG = {
    'game_num_players': 5,
    'game_leak': False,        # False = sealed emulation; True = sequential leak
    'game_delegate': 'BeliefBot:0.5',
    'game_rounds': 6,
    'game_seed': None,      # pin the deal, so E-12 can compare harnesses
}


class ByrsaEnv(Env):
    '''BYRSA environment.'''

    def __init__(self, config):
        self.name = 'byrsa'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        _cfg = DEFAULT_GAME_CONFIG.copy()
        for k in config:
            if k in _cfg:
                _cfg[k] = config[k]
        self.game.configure(_cfg)
        super().__init__(config)
        self.actions = list(action_space.ACTION_NAMES)
        # obs vector: own hand value, own hand size, own suit counts (4),
        # 5 Pillar flags, round, cost, gap, table hand sizes summary (2)
        self.state_shape = [[16] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _get_legal_actions(self):
        return self.game.get_state(self.game.get_player_id())['legal_actions']

    def _extract_state(self, state):
        if state is None:
            obs = np.zeros(16, dtype=np.float32)
            return {'obs': obs, 'legal_actions': OrderedDict(),
                    'raw_obs': None, 'raw_legal_actions': [],
                    'action_record': self.action_recorder}
        o = state['obs']
        counts = o.own_suit_counts()
        # Only A1 §2-legal information enters this vector: own hand value and
        # composition, the public Pillar row, the public cost and gap, and the
        # PUBLIC hand sizes of the table.  No other seat's values appear.
        vec = np.array(
            [o.hand_value(), len(o.hand)] + list(counts)
            + [1.0 if p else 0.0 for p in state['pillars']]
            + [state['round'], state['cost'], state['gap'],
               float(sum(o.hand_sizes)), float(max(o.hand_sizes))],
            dtype=np.float32)
        legal = OrderedDict({a: None for a in state['legal_actions']})
        return {
            'obs': vec,
            'legal_actions': legal,
            'raw_obs': state,
            'raw_legal_actions': [self.actions[a] for a in state['legal_actions']],
            'action_record': self.action_recorder,
        }

    def _decode_action(self, action_id):
        return int(action_id)

    def get_payoffs(self):
        return np.array(self.game.get_payoffs(), dtype=np.float64)

    def get_perfect_information(self):
        return self.game.get_perfect_information()
