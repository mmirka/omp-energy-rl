"""Dynamic reinforcement-learning control of OpenMP workload energy efficiency.

An OpenMP program's progress is tracked at the granularity of the *chunk* -- one
iteration of a dynamically scheduled parallel loop -- by a patched GCC ``libgomp``
that publishes a running chunk counter into shared memory. Dividing that counter's
rate by elapsed time gives **CpS** (Chunks per Second, a performance metric), and by
consumed energy gives **CpJ** (Chunks per Joule, an energy-efficiency metric). Both
are available online, with no prior profiling of the application.

On top of those metrics sit two learned components:

* a **phase-detection autoencoder** whose bottleneck is a discrete (binary) layer.
  The system configuration is re-injected at the decoder, so the bottleneck is
  forced to encode what is left -- the application's execution *phase* -- as a
  short binary code. Training is unsupervised: the number of phases is never
  given to the model.
* a **DQN controller** whose state is the current configuration plus either the
  CpS or the autoencoder's phase code, whose action is a new configuration
  (core count x frequency), and whose reward is the resulting CpJ.

The package provides the configuration codec for the experimental platform, the
CpS/CpJ derivation, the characterization-trace loader, the autoencoder, the DQN
agent, and the online control loop. The control loop's only environment is the
original physical testbed (:mod:`omp_energy_rl.environment`); it is documented
and hardware-gated, not simulated. See the README's 'What survived, and what did
not' section for what is reproducible here and what is not.

Reference: PhD thesis Chapter 4; ReCoSoC 2019; GDR SOC2 2019; MOCAST 2020.
"""

__version__ = "0.1.0"
