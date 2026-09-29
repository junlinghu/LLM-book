"""A small gridworld MDP written from scratch (Sections 10.1, 10.2, and 10.4).

The grid has 5 rows and 5 columns. The agent moves up, down, left, or right;
moves into a wall or off the grid leave it where it is. Entering the goal
gives +1 and entering the pit gives -1; both end the episode. Every other
step gives reward 0, so only discounting makes shorter paths better.
"""
import numpy as np

ROWS, COLS = 5, 5
WALLS = {(1, 1), (1, 2), (3, 3)}
GOAL, PIT = (0, 4), (2, 3)
START = (4, 0)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # up, down, left, right
ACTION_NAMES = ["up", "down", "left", "right"]


class GridWorld:
    def __init__(self):
        self.cells = [(r, c) for r in range(ROWS) for c in range(COLS)
                      if (r, c) not in WALLS]
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.n_states, self.n_actions = len(self.cells), len(ACTIONS)
        self.terminal = np.array([cell in (GOAL, PIT) for cell in self.cells])
        # Deterministic transitions: next_state[s, a] and reward[s, a].
        self.next_state = np.zeros((self.n_states, self.n_actions), int)
        self.reward = np.zeros((self.n_states, self.n_actions))
        for s, (r, c) in enumerate(self.cells):
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ROWS and 0 <= nc < COLS) or (nr, nc) in WALLS:
                    nr, nc = r, c                      # bump: stay put
                self.next_state[s, a] = self.index[(nr, nc)]
                self.reward[s, a] = {GOAL: 1.0, PIT: -1.0}.get((nr, nc), 0.0)

    def step(self, s, a):
        """Return (next_state, reward, done) for action a in state s."""
        s2 = self.next_state[s, a]
        return s2, self.reward[s, a], bool(self.terminal[s2])

    def to_grid(self, v, fill=np.nan):
        """Arrange a per-state vector as a ROWS x COLS array (walls = fill)."""
        g = np.full((ROWS, COLS), fill, dtype=float)
        for s, (r, c) in enumerate(self.cells):
            g[r, c] = v[s]
        return g


def policy_evaluation(env, pi, gamma, tol=1e-12):
    """Iterate the Bellman expectation equation until V stops changing.

    pi[s, a] is the probability of action a in state s.
    """
    V = np.zeros(env.n_states)
    while True:
        # Q[s, a] = r(s, a) + gamma * V(s'), with V = 0 beyond terminal states
        Q = env.reward + gamma * V[env.next_state] * ~env.terminal[env.next_state]
        V_new = np.where(env.terminal, 0.0, (pi * Q).sum(axis=1))
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new


def value_iteration(env, gamma, tol=1e-12):
    """Iterate the Bellman optimality equation; return V* and a greedy policy."""
    V = np.zeros(env.n_states)
    while True:
        Q = env.reward + gamma * V[env.next_state] * ~env.terminal[env.next_state]
        V_new = np.where(env.terminal, 0.0, Q.max(axis=1))
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new


def random_policy(env):
    return np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)
