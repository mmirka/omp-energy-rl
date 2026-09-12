"""The online decision maker: a DQN-shaped Q-network over the 209 configurations.

The controller's job is to pick, every 500 ms, one of the 209 available
configurations (core count x frequency) so as to maximize CpJ. It learns which
one online, while the application it is controlling runs -- there is no offline
training phase and no prior profiling.

**This is deliberately not multi-step reinforcement learning.** ``gamma`` is 0, so
the Bellman update collapses to ``Q[s, a] <- r``: the network is trained to
regress immediate reward, and the machinery around it (experience replay,
epsilon-greedy exploration, a Q-network over state-action pairs) serves
*combinatorial inference* over a large, non-discrete state space rather than
credit assignment over time. The thesis says so explicitly (section 4.2.3): "the
network is only trained to perform combinatorial inference [...] its decisions
depend only on the current state of the system". ``gamma`` is exposed anyway,
because the implementation supports genuine multi-step Q-learning and the
original was written to.

The training schedule has two regimes, split at ``observe_period`` (2048 steps =
1024 s at the 500 ms sampling period):

* **Exploration** -- actions are drawn at random with probability ``epsilon``,
  building up experience across the configuration space.
* **Exploitation** -- the network's ``argmax`` is followed.

Adapted from Jaromir Janisch's DQN walkthrough
(https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/), which is the
lineage the original code names in its own header.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from .platform_config import N_ACTIONS

#: One experience tuple: ``(state, action, reward, next_state)``.
#: ``next_state`` is ``None`` only for a terminal step; the control loop over a
#: continuously running workload never produces one.
Transition = Tuple[np.ndarray, int, float, Optional[np.ndarray]]


@dataclass(frozen=True)
class DQNConfig:
    """Network shape and learning schedule.

    Three presets are provided below. They are not variations invented here --
    they are what the thesis documents and what the two surviving controller
    implementations actually ran. Where those disagree, both are kept rather
    than reconciled; see the README's 'What survived, and what did not' section.
    """

    state_size: int
    """Width of the state vector. 3 without the phase autoencoder
    (``[frequency, CpS, core count]``), 4 with it
    (``[frequency, core count, phase bit 0, phase bit 1]`` -- CpS drops out
    because the phase code supersedes it)."""

    n_actions: int = N_ACTIONS
    hidden_units: Tuple[int, ...] = (8, 64, 256)
    activation: str = "linear"
    optimizer: str = "adam"
    loss: str = "mse"

    batch_size: int = 8
    """Minibatch drawn from replay on every step."""

    replay_capacity: int = 2048
    """Experience replay is a plain FIFO of this many transitions."""

    observe_period: int = 2048
    """Steps of pure exploration before exploitation begins. 2048 steps at a
    500 ms sampling period is 1024 s -- short for a neural network, and chosen
    to keep the exploration cost bearable on a real workload."""

    gamma: float = 0.0
    """Reward discount. 0 in every configuration actually used -- see the module
    docstring."""

    max_epsilon: float = 1.0
    min_epsilon: float = 0.0
    epsilon_decay: float = 0.05
    """Rate of the exponential epsilon decay that starts after
    ``observe_period``."""

    reward_scale: float = 0.1
    """The reward is ``CpJ * reward_scale``. CpJ runs into the thousands for
    compute-bound workloads; scaling keeps the regression targets in a range the
    network handles without a separate normalization step."""

    name: str = "custom"
    provenance: str = ""


#: As documented in thesis Table 4.3: 3 -> 8 / 64 / 256 -> 209, all linear.
THESIS_TABLE_4_3 = DQNConfig(
    state_size=3,
    hidden_units=(8, 64, 256),
    activation="linear",
    batch_size=8,
    replay_capacity=2048,
    observe_period=2048,
    min_epsilon=0.0,
    epsilon_decay=0.05,
    name="thesis-table-4.3",
    provenance="PhD thesis Table 4.3 -- the dimensioning as written up.",
)

#: As the recovered phase-free controller actually ran (``DQN_Seul``). Same
#: widths as the thesis table, but ReLU rather than linear activations.
RECOVERED_NO_AUTOENCODER = replace(
    THESIS_TABLE_4_3,
    activation="relu",
    name="recovered-no-autoencoder",
    provenance=(
        "Recovered DQN_Seul/MyAgent_cpj_F.py. Matches thesis Table 4.3 in width "
        "but uses relu, not linear."
    ),
)

#: As the recovered phase-aware controller actually ran (``DQN_Autoencoder``):
#: a wider, deeper, funnel-shaped net over the 4-dimensional phase state, with a
#: much larger batch and a floor under epsilon.
RECOVERED_WITH_AUTOENCODER = DQNConfig(
    state_size=4,
    hidden_units=(128, 64, 32, 16),
    activation="linear",
    batch_size=1024,
    replay_capacity=2048,
    observe_period=2048,
    min_epsilon=0.05,
    epsilon_decay=0.005,
    name="recovered-with-autoencoder",
    provenance="Recovered DQN_Autoencoder/MyAgent_cpj_F.py.",
)

PRESETS = {
    cfg.name: cfg
    for cfg in (THESIS_TABLE_4_3, RECOVERED_NO_AUTOENCODER, RECOVERED_WITH_AUTOENCODER)
}


class Brain:
    """The Q-network: state in, one Q value per configuration out.

    A plain MLP with a linear output layer -- the output is a regression over
    ``Q[s, a]``, not a distribution, so no softmax.
    """

    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.model = self._build()

    def _build(self) -> Sequential:
        cfg = self.config
        model = Sequential(name=f"q_network_{cfg.name.replace('.', '_')}")
        model.add(
            Dense(
                cfg.hidden_units[0],
                activation=cfg.activation,
                input_dim=cfg.state_size,
            )
        )
        for units in cfg.hidden_units[1:]:
            model.add(Dense(units, activation=cfg.activation))
        model.add(Dense(cfg.n_actions, activation="linear"))
        model.compile(loss=cfg.loss, optimizer=cfg.optimizer)
        return model

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 1, verbose: int = 0):
        return self.model.fit(
            x, y, batch_size=self.config.batch_size, epochs=epochs, verbose=verbose
        )

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self.model.predict(states)

    def predict_one(self, state: np.ndarray) -> np.ndarray:
        return self.predict(np.asarray(state).reshape(1, self.config.state_size)).flatten()


class ExperienceReplay:
    """A fixed-capacity FIFO of transitions, sampled uniformly."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.samples: List[Transition] = []

    def add(self, sample: Transition) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n: int) -> List[Transition]:
        return random.sample(self.samples, min(n, len(self.samples)))

    def __len__(self) -> int:
        return len(self.samples)


class Agent:
    """Epsilon-greedy action selection plus replay-based Q-network training.

    Args:
        config: network shape and schedule; see :data:`PRESETS`.
        on_train: optional callback invoked after each training step with a dict
            of diagnostics (batch size, epsilon, step count, fit duration). The
            original wrote several ad-hoc CSV files from inside the training
            routine; routing that through a callback keeps the agent free of I/O.
    """

    def __init__(
        self,
        config: DQNConfig,
        on_train: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config
        self.brain = Brain(config)
        self.replay = ExperienceReplay(config.replay_capacity)
        self.steps = 0
        self.epsilon = config.max_epsilon
        self._on_train = on_train

    # -- action selection --------------------------------------------------- #

    def act(self, state: np.ndarray) -> int:
        """Choose a configuration for the next sampling period.

        .. note::
           The exploration test is ``random() < epsilon and steps <
           observe_period``, faithfully reproducing the recovered controller.
           The conjunction -- rather than the disjunction that a textbook
           epsilon-greedy schedule uses -- means exploration stops *hard* at
           ``observe_period`` and the decayed epsilon afterwards has no effect on
           behaviour. The original source carries the disjunctive form as a
           commented-out line directly above, so this was a deliberate switch to
           a clean explore-then-exploit split. It is kept because it is what
           produced the published results; :attr:`epsilon` is still tracked, and
           the recovered traces plot it.
        """
        if random.random() < self.epsilon and self.steps < self.config.observe_period:
            return random.randint(0, self.config.n_actions - 1)
        return int(np.argmax(self.brain.predict_one(state)))

    def is_exploring(self) -> bool:
        """Whether the agent is still inside its exploration period."""
        return self.steps < self.config.observe_period

    # -- experience --------------------------------------------------------- #

    def capture(self, sample: Transition) -> None:
        """Record one ``(state, action, reward, next_state)`` transition.

        Also advances the step counter and decays epsilon once the observation
        period is over.
        """
        self.replay.add(sample)
        self.steps += 1
        cfg = self.config
        if self.steps > cfg.observe_period:
            self.epsilon = cfg.min_epsilon + (cfg.max_epsilon - cfg.min_epsilon) * math.exp(
                -cfg.epsilon_decay * (self.steps - cfg.observe_period)
            )

    # -- learning ----------------------------------------------------------- #

    def train_step(self) -> Optional[dict]:
        """Draw a minibatch from replay and take one gradient step.

        Returns the diagnostics dict passed to ``on_train``, or ``None`` if
        replay is still empty.
        """
        import time

        batch = self.replay.sample(self.config.batch_size)
        if not batch:
            return None

        states, targets = self._targets(batch)
        started = time.time()
        history = self.brain.train(states, targets)
        diagnostics = {
            "step": self.steps,
            "epsilon": self.epsilon,
            "batch_size": len(batch),
            "fit_seconds": time.time() - started,
            "loss": float(history.history["loss"][-1]),
        }
        if self._on_train is not None:
            self._on_train(diagnostics)
        return diagnostics

    def _targets(self, batch: Sequence[Transition]) -> Tuple[np.ndarray, np.ndarray]:
        """Build the ``(states, target Q)`` regression problem for one minibatch."""
        cfg = self.config
        no_state = np.zeros(cfg.state_size)

        states = np.array([item[0] for item in batch])
        next_states = np.array(
            [no_state if item[3] is None else item[3] for item in batch]
        )

        predicted_q = self.brain.predict(states)
        predicted_next_q = self.brain.predict(next_states)

        x = np.zeros((len(batch), cfg.state_size))
        y = np.zeros((len(batch), cfg.n_actions))
        for i, (state, action, reward, next_state) in enumerate(batch):
            target = predicted_q[i]
            if next_state is None:
                target[action] = reward
            else:
                target[action] = reward + cfg.gamma * np.amax(predicted_next_q[i])
            x[i] = state
            y[i] = target
        return x, y

    # -- persistence -------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save the Q-network. Replay contents and epsilon are not saved."""
        self.brain.model.save(path)
