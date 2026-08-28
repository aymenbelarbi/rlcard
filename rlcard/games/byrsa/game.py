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
# RULING 12 (v3 1.4): the Envoy declaration is its own lap, ahead of the sealed
# commit.  RLCard is sequential already, so this costs nothing structurally --
# but it has to be walked in the SAME Sufet-clockwise order over the SAME
# eligible seats as the native lap, or E-12 stops comparing like with like.
PHASE_ENVOY = 'envoy'


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

    def clear_script(self):
        '''Discard any scripted move the engine did not consume.

        A2 §6 makes the Envoy *instead of* pledging, so ``rules.step_pledge``
        never calls ``pledge()`` on a seat that declared -- and a script left
        sitting here is then consumed by the NEXT caller, which is Step 5's
        D01 Rebuild.  Caught by the G-W5 declaring arm at 6p, where both
        wrappers agreed with each other and both differed from the engine.
        '''
        self._pledge = None
        self._rescue = None

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

    # declare_envoy is deliberately NOT scripted.  Under ruling 12 the env
    # takes the declaration as an action and writes it straight into the state,
    # and ``step_pledge(prelude_done=True)`` never asks an agent; under the v2
    # flag the sealed commit asks, and __getattr__ forwards to the delegate.

    def __getattr__(self, item):
        # Same deepcopy/pickle recursion trap as the OpenSpiel wrapper: those
        # protocols probe for dunders before __init__ has run, and a naive
        # `getattr(self._d, item)` re-enters this method looking for `_d`.
        if item.startswith("__") or item == "_d":
            raise AttributeError(item)
        try:
            return getattr(self.__dict__["_d"], item)
        except KeyError:
            raise AttributeError(item) from None


class ByrsaGame:
    '''RLCard game wrapper around byrsa_sim.'''

    def __init__(self, allow_step_back=False, num_players=5):
        self.allow_step_back = allow_step_back
        self.num_players = num_players
        self.np_random = random.Random()
        self.leak = False
        self.delegate = 'BeliefBot:0.5'
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
        return action_space.NUM_DISTINCT_ACTIONS

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
        self.running_total = 0
        # Step 3's discussion window comes from the ENGINE, not from here --
        # the Notable window and U7 Silver League both change a hand, and a
        # seat that declares before its draw-2 decides on different
        # information than one that declares after.
        rules.pledge_window(self.state, self.agents)
        self.pending = {}
        # RULING 12's lap.  The order comes from the ENGINE -- one definition
        # of who is offered the Envoy and in what order, shared by the native
        # lap and both wrappers.  Seats that may not declare are absent rather
        # than handed a forced pass, which is what the native guard does.
        self.envoy_order = (
            rules.envoy_declaration_order(self.state)
            if self.state.config.envoy_declared_before_commit else [])
        if self.envoy_order:
            self.phase = PHASE_ENVOY
            self.order = self.envoy_order
            self.idx = 0
        else:
            self._open_pledge_lap()

    def _open_pledge_lap(self):
        '''The sealed commit's seat order, and the snapshot it is sealed on.

        The snapshot is taken HERE and not in ``_begin_round`` because it must
        follow the declaration lap: a pledge observation carries who has
        already declared (``envoy_declared_this_round``), which is the whole
        point of ruling 12.  Freezing it earlier would hide the declarations
        from every seat and diverge from the native engine, which snapshots
        inside ``step_pledge`` after the lap.
        '''
        self.phase = PHASE_PLEDGE
        # A2 §3b makes the Rescue lap start with the Sufet; the Pledge is
        # simultaneous and has no order, so we use the same seat order for the
        # sequential emulation and freeze the observation snapshot here.
        self.order = [(self.state.sufet + i) % self.num_players
                      for i in range(self.num_players)]
        self.idx = 0
        self._frozen = byrsa_obs.public_snapshot(self.state)

    @property
    def current_player(self):
        if self.state.game_over:
            return -1
        return self.order[self.idx]

    # -- observations ------------------------------------------------------
    def _obs(self, seat):
        if self.phase == PHASE_ENVOY:
            return byrsa_obs.build(self.state, seat, 'envoy_declare')
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
            'legal_actions': action_space.legal_actions(obs, self.phase),
            'round': self.state.round,
            'pillars': tuple(self.state.pillars),
            'cost': self.state.cost,
            'gap': max(0, self.state.cost - self.running_total),
        }

    # -- stepping ----------------------------------------------------------
    def step(self, action):
        seat = self.current_player
        obs = self._obs(seat)
        if self.phase == PHASE_ENVOY:
            self._apply_envoy(seat, int(action))
            self.idx += 1
            if self.idx >= len(self.order):
                self._open_pledge_lap()
        elif self.phase == PHASE_PLEDGE:
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
        self.running_total = rules.step_pledge(st, self.agents, prelude_done=True)
        for a in self.agents:
            a.clear_script()
        if self.running_total >= st.cost:
            self._finish_round()
            return
        self.phase = PHASE_RESCUE
        self.order = [(st.sufet + i) % self.num_players
                      for i in range(self.num_players)]
        self.idx = 0

    def _apply_envoy(self, seat, action):
        '''One seat's declaration, written into the state IMMEDIATELY so the
        next seat's observation carries it -- that visibility is the whole
        point of ruling 12.'''
        declared = (action == action_space.DECLARE_ENVOY)
        assert not declared or rules.may_declare_envoy(self.state, seat), (
            'DECLARE_ENVOY offered to a seat that may not declare')
        self.state.envoy_declared[seat] = declared

    def _apply_rescue(self, seat, action, obs):
        self.agents[seat]._rescue = action_space.decode_rescue(obs, action)
        out = rules.rescue_one(self.state, self.agents, seat, self.running_total)
        self.agents[seat].clear_script()
        return out

    def _finish_round(self):
        st, ag = self.state, self.agents
        st.siege_total = self.running_total     # the graded Siege reads this
        rules.step_resolution(st, ag, self.running_total >= st.cost)
        if st.standing == 0:
            st.destroyed = True
            st.game_over = True
        # Steps 4 and 5 live in byrsa_sim.rules, not here.  This block used to
        # duplicate them, so RULING 10 landed natively and not in RLCard and
        # G-W5 diverged.  One rules change, one place.
        rules.steps_4_and_5(st, ag)
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
