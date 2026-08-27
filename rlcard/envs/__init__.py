''' Register new environments
'''
from rlcard.envs.env import Env
from rlcard.envs.registration import register, make

register(
    env_id='blackjack',
    entry_point='rlcard.envs.blackjack:BlackjackEnv',
)

register(
    env_id='doudizhu',
    entry_point='rlcard.envs.doudizhu:DoudizhuEnv',
)

register(
    env_id='limit-holdem',
    entry_point='rlcard.envs.limitholdem:LimitholdemEnv',
)

register(
    env_id='no-limit-holdem',
    entry_point='rlcard.envs.nolimitholdem:NolimitholdemEnv',
)

register(
    env_id='leduc-holdem',
    entry_point='rlcard.envs.leducholdem:LeducholdemEnv'
)

register(
    env_id='uno',
    entry_point='rlcard.envs.uno:UnoEnv',
)

register(
    env_id='mahjong',
    entry_point='rlcard.envs.mahjong:MahjongEnv',
)

register(
    env_id='gin-rummy',
    entry_point='rlcard.envs.gin_rummy:GinRummyEnv',
)

register(
    env_id='bridge',
    entry_point='rlcard.envs.bridge:BridgeEnv',
)

# BYRSA needs the byrsa_sim engine, which lives outside this repo (it is the
# only place a BYRSA rule is implemented; this env merely wraps it).  RLCard
# resolves entry points EAGERLY in EnvSpec.__init__, so an unguarded register()
# here would import rlcard/envs/byrsa.py at package-import time and take the
# whole of `import rlcard` down with it when the engine is absent -- blackjack,
# doudizhu and every other game included.  Degrade to "byrsa is not registered"
# instead; upstream behaviour is then bit-for-bit unchanged.
try:
    register(
        env_id='byrsa',
        entry_point='rlcard.envs.byrsa:ByrsaEnv',
    )
except ImportError:
    pass
