"""Small Gym-style environment for Togyzkumalak RL experiments.

This follows the same practical shape as gym-togyzkumalak:
reset() -> observation, step(action) -> observation/reward/done/info.
It intentionally has no gym dependency so Render can deploy it safely.
"""

from game_logic import TogyzkumalakState


class TogyzkumalakGymEnv:
    observation_size = 128
    action_size = 9

    def __init__(self):
        self.state = TogyzkumalakState()

    def reset(self):
        self.state = TogyzkumalakState()
        return self.observation()

    def legal_actions(self):
        start = self.state.currentPlayer * 9
        return [move - start for move in self.state.getPossibleMoves(self.state.currentPlayer)]

    def step(self, action):
        if not isinstance(action, int):
            raise TypeError("action must be an integer from 0 to 8")
        if action < 0 or action >= self.action_size:
            raise ValueError("action must be from 0 to 8")

        player = self.state.currentPlayer
        before = self.state.kazans[player] - self.state.kazans[1 - player]
        move = player * 9 + action

        if not self.state.makeMove(move):
            return self.observation(), -1.0, self.state.isGameOver, {
                "invalid": True,
                "winner": self.state.winner,
                "legal_actions": self.legal_actions(),
            }

        done = self.state.isGameOver
        reward = 0.0
        if done:
            if self.state.winner == player:
                reward = 1.0
            elif self.state.winner == -1:
                reward = 0.0
            else:
                reward = -1.0
        else:
            after = self.state.kazans[player] - self.state.kazans[1 - player]
            reward = max(-0.25, min(0.25, (after - before) / 20.0))

        return self.observation(), reward, done, {
            "invalid": False,
            "winner": self.state.winner,
            "current_player": self.state.currentPlayer,
            "legal_actions": self.legal_actions(),
        }

    def observation(self):
        obs = [0.0] * self.observation_size
        offset = 0

        for player in (0, 1):
            for rel in range(9):
                stones = self.state.board[player * 9 + rel]
                width = 5 if rel == 8 else 7
                values = _encode_ninth_hole(stones) if rel == 8 else _encode_hole(stones)
                for i in range(width):
                    obs[offset + i] = values[i]
                offset += width

            obs[offset] = min(self.state.kazans[player], 82) / 82.0
            offset += 1
            obs[offset] = 1.0 if self.state.kazans[player] > self.state.kazans[1 - player] else 0.0
            offset += 1

        obs[126] = 1.0 if self.state.currentPlayer == 0 else 0.0
        obs[127] = 1.0 if self.state.currentPlayer == 1 else 0.0

        for tuzdyk_owner, pit in enumerate(self.state.tuzdyks):
            if pit == -1:
                continue
            pit_player = pit // 9
            rel = pit % 9
            if rel == 8:
                continue
            hole_offset = (0 if pit_player == 0 else 63) + rel * 7
            obs[hole_offset + 6] = 1.0 if tuzdyk_owner == pit_player else -1.0

        return obs


def _encode_hole(stones):
    return [
        1.0 if stones >= 1 else 0.0,
        1.0 if stones >= 2 else 0.0,
        float(stones % 2),
        (stones % 9) / 8.0,
        (stones % 18) / 17.0,
        min(stones, 18) / 9.0,
        0.0,
    ]


def _encode_ninth_hole(stones):
    return [
        1.0 if stones >= 1 else 0.0,
        float(stones % 2),
        (stones % 9) / 8.0,
        (stones % 18) / 17.0,
        min(stones, 18) / 9.0,
    ]

