import pytest
import dataclasses
from leveraged_core.config import ChallengeConfig

def test_challenge_constants_are_accurate():
    config = ChallengeConfig()
    assert config.STARTING_BALANCE == 50000.0
    assert config.MAX_DAILY_LOSS_PCT == -0.03

def test_config_is_immutable():
    config = ChallengeConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.STARTING_BALANCE = 100000.0
