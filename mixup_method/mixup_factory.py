from .cutmix import CutMix
from .inputmix import InputMix
from .no_mix import NoMix
from .r_mix import RMix
from .constants import R_MIX, INPUT_MIX, CUT_MIX, NO_MIX, RL_MIX


class MixupFactory:
    REGISTER_MIXUP_STRATEGY = {
        R_MIX: RMix,
        INPUT_MIX: InputMix,
        CUT_MIX: CutMix,
        NO_MIX: NoMix,
        RL_MIX: RMix
    }
    
    def getMixupStrategy(self, strategyName, *args, **kwargs):
        if strategyName not in self.REGISTER_MIXUP_STRATEGY:
            raise NotImplementedError("Strategy has not been registered!")
        return self.REGISTER_MIXUP_STRATEGY[strategyName](*args, **kwargs)