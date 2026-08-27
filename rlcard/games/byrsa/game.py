'''BYRSA as an RLCard game.

RLCard is strictly sequential: one player, one action.  BYRSA's Pledge is
simultaneous and sealed.  That mismatch is the whole reason this env exists as
the THIRD harness rather than the first (byrsa-cards/00 §3): it is an
independently written implementation that must agree with OpenSpiel and the
native engine on identical seeds, and 02 E-12 calls that the only automated
defence against the error class that produced the retracted audit.

NO RULE IS IMPLEMENTED HERE.  Everything resolves through ``byrsa_sim.rules``,
which implements A2 v1.2.  If this env and byrsa_sim disagree, the env is the
bug (byrsa-cards/CLAUDE.md §9).

THE TWO MODES.  Because every BYRSA bot is a function of its OWN observation,
stepping seats one at a time must reproduce a simultaneous commit EXACTLY --
provided the sequential adapter does not let a later seat see an earlier
seat's pledge.  So this game runs in two modes:

  ``leak=False`` (default)  every seat's pledge observation is built from a
      snapshot frozen BEFORE any seat committed.  Must match the native and
      OpenSpiel harnesses exactly, card for card.

  ``leak=True``             the pledge lap is genuinely sequential: later seats
      see the shrunken hands and revealed commitments of earlier ones.

The gap between the two IS the measurement of what sequential emulation costs,
which 02 E-12 flags as signal rather than noise.
'''
import random

from byrsa_sim import action_space, observation as byrsa_obs, rules
from byrsa_sim.agents import all_agents  # noqa: F401  (populates the registry)
from byrsa_sim.agents import registry
from byrsa_sim.config import Config

PHASE_PLEDGE = 'pledge'
PHASE_RESCUE = 'rescue'


class _Scripted:
    '''Agent whose pledge/rescue are dictated by the env; everything else --
    Notables, Ballots, flip-buy, Sufet choices, burns -- defers to the
    delegate policy, calling the same engine functions the native harness does.
    '''

    def __init__(self, delegate):
        self._d = delegate
        self.name = f'RLCard/{delegate.name}'
        self.sees_hidden_state = False
        self._pledge = None
        self._rescue = None

    def reset(self, seat, rng):
        self.seat, self.rng = seat, rng
        self._d.reset(seat, rng)

    def pledge(self, obs):
        if self._pledge is None:
            return self._d.pledge(obs)
        c, self._pledge = self._pledge, None
        return tuple(x for x in c if x in obs.hand)

    def rescue(self, obs):
        if self._rescue is None:
            return self._d.rescue(obs)
        c, self._rescue = self._rescue, None
        return tuple(x for x in c if x in obs.hand)

    def __getattr__(self, item):
        return getattr(self._d, item)


class ByrsaGame:
    '''RLCard game wrapper around byrsa_sim.'''

    def __init__(self, allow_step_back=False, num_players=5):
        self.allow_step_back = allow_step_back
        self.num_players = num_players
        self.np_random = random.Random()
        self.leak = False
        self.delegate = 'BeliefBot(0.5)'
        self.rounds = 6
        self.seed = None
        self.fixed_seed = None
        self.state = None

    # -- configuration -----------------------------------------------------
    def configure(self, game_config):
        self.num_players = game_config.get('game_num_players', self.num_players)
        self.leak = bool(game_config.get('game_leak', self.leak))
        self.delegate = game_config.get('game_delegate', self.delegate)
        self.rounds = game_config.get('game_rounds', self.rounds)
        self.fixed_seed = game_config.get('game_seed', None)

    def get_num_players(self):
        return self.num_players

    def get_num_actions(self):
        return action_space.NUM_ACTIONS

    def get_player_id(self):
        return self.current_player

    def is_over(self):
        return bool(self.state.game_over)

    # -- lifecycle ---------------------------------------------------------
    def _draw_seed(self):
        '''RLCard's Env.seed() installs a numpy RandomState on self.np_random,
        but a bare Game() carries a stdlib Random.  Support both so the game
        is usable inside and outside the env.'''
        r = self.np_random
        if hasattr(r, 'randrange'):
            return r.randrange(1 << 30)
        return int(r.randint(1 << 30))

    def init_game(self, seed=None):
        if seed is None:
            seed = self.fixed_seed if self.fixed_seed is not None else self._draw_seed()
        self.seed = seed
        rng = random.Random(seed)
        cfg = Config(players=self.num_players, rounds=self.rounds)
        self.state = rules.setup(cfg, rng)
        self.agents = [_Scripted(registry.make(self.delegate))
                       for _ in range(self.num_players)]
        # Canonical seeding, identical to the native and OpenSpiel harnesses.
        rules.bind_agents(self.state, self.agents, seed)
        self._begin_round()
        return self.get_state(self.current_player), self.current_player

    def _begin_round(self):
        rules.begin_round(self.state, self.agents)
        self.phase = PHASE_PLEDGE
        # A2 §3b makes the Rescue lap start with the Sufet; the Pledge is
        # simultaneous and has no order, so we use the same seat order for the
        # sequential emulation and freeze the observation snapshot here.
        self.order = [(self.state.sufet + i) % self.num_players
                      for i in range(self.num_players)]
        self.idx = 0
        self.pending = {}
        self._frozen = byrsa_obs.public_snapshot(self.state)
        self.running_total = 0

    @property
    def current_player(self):
        if self.state.game_over:
            return -1
        return self.order[self.idx]

    # -- observations ------------------------------------------------------
    def _obs(self, seat):
        if self.phase == PHASE_RESCUE:
            return byrsa_obs.build(self.state, seat, 'rescue',
                                   gap=max(0, self.state.cost - self.running_total))
        if self.leak:
            # genuinely sequential: this seat sees what earlier seats did
            return byrsa_obs.build(self.state, seat, 'pledge')
        # sealed: the snapshot predates every commit this round
        return byrsa_obs.build(self.state, seat, 'pledge', pub=self._frozen)

    def get_state(self, player_id):
        obs = self._obs(player_id)
        return {
            'obs': obs,
            'seat': player_id,
            'phase': self.phase,
            'legal_actions': action_space.legal_actions(
                obs, 'pledge' if self.phase == PHASE_PLEDGE else 'rescue'),
            'round': self.state.round,
            'pillars': tuple(self.state.pillars),
            'cost': self.state.cost,
            'gap': max(0, self.state.cost - self.running_total),
        }

    # -- stepping ----------------------------------------------------------
    def step(self, action):
        seat = self.current_player
        obs = self._obs(seat)
        if self.phase == PHASE_PLEDGE:
            cards = action_space.decode_pledge(obs, int(action))
            self.pending[seat] = cards
            if self.leak:
                # remove now, so the NEXT seat sees the shrunken hand and the
                # revealed commitment -- this is the leak being measured
                for c in cards:
                    self.state.hands[seat].remove(c)
                self.state.pledges[seat] = list(cards)
            self.idx += 1
            if self.idx >= len(self.order):
                self._close_pledge_lap()
        else:
            self.running_total = self._apply_rescue(seat, int(action), obs)
            self.idx += 1
            if (self.idx >= len(self.order)
                    or self.running_total >= self.state.cost):
                self._finish_round()
        if self.state.game_over:
            return None, -1
        return self.get_state(self.current_player), self.current_player

    def _close_pledge_lap(self):
        '''Hand the collected choices to the ENGINE for the real joint commit.

        In leak mode the cards were pulled from hands to build the sequential
        observations; they go back first, so ``rules.step_pledge`` performs the
        identical commit in both modes and no rule is duplicated here.
        '''
        st = self.state
        if self.leak:
            for seat, cards in self.pending.items():
                for c in cards:
                    st.pledges[seat].remove(c)
                    st.hands[seat].append(c)
        for seat, cards in self.pending.items():
            self.agents[seat]._pledge = cards
        self.running_total = rules.step_pledge(st, self.agents)
        if self.running_total >= st.cost:
            self._finish_round()
            return
        self.phase = PHASE_RESCUE
        self.order = [(st.sufet + i) % self.num_players
                      for i in range(self.num_players)]
        self.idx = 0

    def _apply_rescue(self, seat, action, obs):
        self.agents[seat]._rescue = action_space.decode_rescue(obs, action)
        return rules.rescue_one(self.state, self.agents, seat, self.running_total)

    def _finish_round(self):
        st, ag = self.state, self.agents
        rules.step_resolution(st, ag, self.running_total >= st.cost)
        if st.standing == 0:
            st.destroyed = True
            st.game_over = True
        if not st.game_over:
            rules.step_sufet(st, ag)
        st.forbidden_suit = -1
        st.required_suit = -1
        if not st.game_over:
            rules.step_decree(st, ag)
        if st.round >= st.max_rounds:
            st.game_over = True
        if not st.game_over:
            self._begin_round()

    def get_payoffs(self):
        return list(rules.score(self.state)['totals'])

    def get_perfect_information(self):
        st = self.state
        return {
            'round': st.round, 'pillars': tuple(st.pillars),
            'hands': [tuple(h) for h in st.hands],
            'claims': [tuple(c) for c in st.claims],
        }
