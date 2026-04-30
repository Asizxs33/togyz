from togyzkumalak_gym_env import TogyzkumalakGymEnv


def test_gym_env_smoke():
    env = TogyzkumalakGymEnv()
    obs = env.reset()
    assert len(obs) == 128
    assert env.legal_actions() == list(range(9))

    obs, reward, done, info = env.step(0)
    assert len(obs) == 128
    assert isinstance(reward, float)
    assert done is False
    assert info["invalid"] is False
    assert "legal_actions" in info


if __name__ == "__main__":
    test_gym_env_smoke()
    print("gym env ok")

