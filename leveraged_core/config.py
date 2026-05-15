# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass(frozen=True)
class ChallengeConfig:
    STARTING_BALANCE: float = 50000.0
    PROFIT_TARGET_PCT: float = 0.06
    MAX_DAILY_LOSS_PCT: float = -0.03
    MAX_TRAILING_LOSS_PCT: float = -0.06

CHALLENGE = ChallengeConfig()