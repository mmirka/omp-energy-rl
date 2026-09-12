"""The online control loop that ties metrics, phase detection and the agent together.

Once per sampling period (500 ms), the controller:

1. asks the agent for an action -- a configuration out of the 209 available;
2. applies it to the machine;
3. waits out the rest of the period while the workload runs under it;
4. measures CpS and CpJ over that period;
5. optionally asks the phase autoencoder which phase the workload is now in;
6. rewards the action with ``CpJ * reward_scale`` and stores the transition;
7. takes one gradient step against a minibatch drawn from replay.

Two state definitions are supported, and which one is in use is the single
architectural difference between the thesis's two controllers:

* **Without phase detection** (thesis Fig. 4.15) the state is
  ``[frequency, CpS, core count]``. This works for single-phase workloads --
  DGEMM settles within 3% of its best-known configuration -- but on multi-phase
  workloads the agent cannot tell the phases apart, because raw CpS values from
  different phases overlap once you also vary the configuration.
* **With phase detection** (thesis Fig. 4.17) the state is
  ``[frequency, core count, phase code...]``. CpS drops out of the state
  entirely: the autoencoder has already distilled it into a discrete phase id,
  which is the part the agent can actually generalize over. Removing the
  autoencoder from an otherwise identical experiment costs 34% of mean CpJ
  (thesis Fig. 4.24), which is the cleanest evidence in the work that the phase
  information is what makes multi-phase control work.

The loop is written against the :class:`Environment` protocol rather than against
the hardware, so it can be read and reviewed without a testbed. The only
implementation of that protocol is
:class:`~omp_energy_rl.environment.XeonEnvironment`, which refuses to construct
off-testbed -- deliberately, since there is no honest way to simulate this
environment (see the README's 'What survived, and what did not' section).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple

import numpy as np

from . import platform_config as plat
from .agent import Agent
from .data import FeatureScaler
from .metrics import Efficiency


class Environment(Protocol):
    """What the control loop needs from an environment."""

    @property
    def configuration(self) -> Tuple[int, float]:
        """The configuration currently applied, as ``(n_cores, freq_ghz)``."""

    def apply_action(self, action: int) -> Tuple[int, float]:
        """Apply a DQN action; return the resulting configuration."""

    def wait_for_period(self) -> float:
        """Sleep out the rest of the sampling period; return seconds slept."""

    def sample(self) -> Optional[Efficiency]:
        """Measure CpS/CpJ over the period just elapsed."""


class PhaseDetector:
    """Wraps a trained encoder plus the scaler its inputs were fitted under.

    Both halves are required. An encoder alone is not usable: the phase code it
    emits is a function of *scaled* inputs, and the same CpS scaled against a
    different trace's min/max lands in a different phase. Keeping them together
    is how that mistake is avoided.
    """

    def __init__(self, encoder, scaler: FeatureScaler) -> None:
        self.encoder = encoder
        self.scaler = scaler

    @property
    def code_size(self) -> int:
        return int(self.encoder.output_shape[-1])

    def __call__(self, cps: float, freq_ghz: float, n_cores: int) -> np.ndarray:
        """Return the ``+-1`` phase code for one live reading, shape ``(code_size,)``."""
        scaled = self.scaler.transform_row(cps, freq_ghz, n_cores)
        return self.encoder.predict(scaled)[0]


@dataclass
class ControlRecord:
    """One row of the controller's history, matching the recovered trace schema."""

    elapsed_s: float
    sleep_s: float
    t_act_s: float
    t_autoencoder_s: float
    t_dqn_s: float
    energy_j: float
    cps: float
    cpj: float
    n_cores: int
    freq_ghz: float
    action: int
    reward: float
    epsilon: float
    exploring: bool
    phase_code: Optional[List[float]] = None


@dataclass
class ControllerConfig:
    """Loop-level settings that are not the agent's own hyperparameters."""

    total_steps: int = 3000
    """Sampling periods to run for. 3000 x 500 ms = 25 minutes, of which the
    first 1024 s is exploration."""

    cps_scale: float = 2.5e6
    """Divisor that maps raw CpS into roughly ``[0, 1]`` for the phase-free
    state vector. Workload-specific -- chunk size varies with the parallel loop,
    so CpS has no absolute scale (see :mod:`omp_energy_rl.metrics`). The
    recovered controller hard-coded this value for the two-phase synthetic
    benchmark; it must be re-derived for any other application, e.g. from that
    application's characterization sweep."""

    checkpoint_every: int = 2048
    """Save the Q-network every N steps. The default coincides with the end of
    the exploration period."""


class Controller:
    """Runs the online control loop.

    Args:
        environment: the machine under control.
        agent: the DQN agent. Its ``config.state_size`` decides which state
            definition is used -- 3 without a phase detector, ``2 + code_size``
            with one -- and a mismatch is rejected up front.
        phase_detector: optional; supplying one switches the state definition.
        config: loop settings.
        on_record: called with each :class:`ControlRecord`. Keeping history
            writing outside the loop is why this module does no file I/O.
    """

    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        phase_detector: Optional[PhaseDetector] = None,
        config: Optional[ControllerConfig] = None,
        on_record: Optional[Callable[[ControlRecord], None]] = None,
    ) -> None:
        self.environment = environment
        self.agent = agent
        self.phase_detector = phase_detector
        self.config = config or ControllerConfig()
        self._on_record = on_record

        expected = 3 if phase_detector is None else 2 + phase_detector.code_size
        if agent.config.state_size != expected:
            raise ValueError(
                f"agent expects a state of size {agent.config.state_size}, but this "
                f"controller builds one of size {expected} "
                f"({'without' if phase_detector is None else 'with'} phase detection). "
                "Pick a matching preset from omp_energy_rl.agent.PRESETS."
            )

    # -- state -------------------------------------------------------------- #

    def build_state(
        self,
        cps: float,
        n_cores: int,
        freq_ghz: float,
        phase_code: Optional[np.ndarray],
    ) -> np.ndarray:
        """Assemble the agent's state vector.

        Without a phase detector: ``[frequency, CpS, core count]``, each scaled
        to roughly ``[0, 1]``. With one: ``[frequency, core count, *phase code]``
        -- CpS is gone, replaced by the code derived from it.
        """
        norm_freq, norm_cores = plat.normalize_config(n_cores, freq_ghz)
        if phase_code is None:
            return np.array([norm_freq, cps / self.config.cps_scale, norm_cores])
        return np.concatenate(([norm_freq, norm_cores], np.asarray(phase_code)))

    # -- loop --------------------------------------------------------------- #

    def run(self) -> List[ControlRecord]:
        """Run the control loop for ``config.total_steps`` sampling periods."""
        history: List[ControlRecord] = []
        started = time.time()

        # Apply the lowest configuration and take two measurements before the
        # loop proper: the first only establishes the sampler's baseline (CpS
        # and CpJ are differences, so a single reading yields nothing), and the
        # second -- a full period later -- is the first real measurement, which
        # gives the first step a state to act on.
        self.environment.apply_action(0)
        self.environment.wait_for_period()
        self.environment.sample()
        self.environment.wait_for_period()

        n_cores, freq_ghz = self.environment.configuration
        efficiency = self.environment.sample()
        cps = efficiency.cps if efficiency else 0.0
        phase_code = self._detect(cps, freq_ghz, n_cores)[0]
        state = self.build_state(cps, n_cores, freq_ghz, phase_code)

        for step in range(self.config.total_steps):
            t0 = time.time()
            action = self.agent.act(state)
            n_cores, freq_ghz = self.environment.apply_action(action)
            t_act = time.time() - t0

            slept = self.environment.wait_for_period()
            efficiency = self.environment.sample()
            cps = efficiency.cps if efficiency else 0.0
            cpj = efficiency.cpj if efficiency else 0.0
            energy_j = efficiency.d_energy_j if efficiency else 0.0

            reward = cpj * self.agent.config.reward_scale

            phase_code, t_autoencoder = self._detect(cps, freq_ghz, n_cores)
            next_state = self.build_state(cps, n_cores, freq_ghz, phase_code)

            self.agent.capture((state, action, reward, next_state))

            t1 = time.time()
            self.agent.train_step()
            t_dqn = time.time() - t1

            record = ControlRecord(
                elapsed_s=time.time() - started,
                sleep_s=slept,
                t_act_s=t_act,
                t_autoencoder_s=t_autoencoder,
                t_dqn_s=t_dqn,
                energy_j=energy_j,
                cps=cps,
                cpj=cpj,
                n_cores=n_cores,
                freq_ghz=freq_ghz,
                action=action,
                reward=reward,
                epsilon=self.agent.epsilon,
                exploring=self.agent.is_exploring(),
                phase_code=None if phase_code is None else [float(v) for v in phase_code],
            )
            history.append(record)
            if self._on_record is not None:
                self._on_record(record)

            state = next_state

        return history

    def _detect(
        self, cps: float, freq_ghz: float, n_cores: int
    ) -> Tuple[Optional[np.ndarray], float]:
        if self.phase_detector is None:
            return None, 0.0
        t0 = time.time()
        code = self.phase_detector(cps, freq_ghz, n_cores)
        return code, time.time() - t0
