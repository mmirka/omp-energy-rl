# The shell layer

Two scripts that drive the experimental testbed. They are the non-Python half
of the system: applying a configuration, and starting a controlled run.

All three **require the testbed** — a dual-socket Intel Xeon E5-2630 v4 server
with `cpupower`, the `acpi-cpufreq` driver, readable RAPL MSRs, and a workload
built against the patched GCC/`libgomp`. That machine is no longer available;
see the main [`README`](../README.md) for what it was and what still runs
without it.
`set_config.sh` checks the topology and refuses rather than half-applying a
configuration to the wrong machine.

| Script | What it does |
|---|---|
| `set_config.sh` | Apply one `(core count, frequency)` configuration — one controller action, by hand. |
| `launch_control.sh` | Start a workload and the online controller on disjoint cores. |

## Applying a configuration

```bash
sudo ./set_config.sh 8 1.9 $(pidof -s ./mixed_cxmy)
```

An action is four calls, and the **order matters**:

```bash
cpupower --cpu all frequency-set --freq 1.20GHz     # 1. everything to the floor
cpupower --cpu 19  frequency-set --freq 3.10GHz     # 2. controller core back up
cpupower --cpu 0-N frequency-set --freq 1.90GHz     # 3. allocated cores to target
taskset -pa 0x03C0F <pid>                           # 4. re-pin the workload
```

1. **Everything to the floor first.** Cores dropped from the allocation must not
   stay parked at the previous configuration's frequency. They would keep drawing
   package power that CpJ is then charged for, and the reward signal would
   partly measure the *previous* action.
2. **The controller's own core goes back up.** Core 19 runs the collector and the
   training step; it is held at 3.10 GHz so measurement never becomes the
   bottleneck.
3. **Then raise the allocation.**
4. **`taskset -pa`** — `-a` covers every thread of the process, not just the main
   one. Without `-a` the OpenMP worker threads keep their old affinity and the
   action does nothing.

### Why cores are allocated socket-interleaved

Cores are handed out `0, 10, 1, 11, 2, 12, …` — alternating sockets — not
`0, 1, 2, …`. Filling socket 0 first would confound "more cores" with "the second
socket is now powered up", a step change in package power that has nothing to do
with the workload. Interleaving keeps the two sockets symmetric at every core
count, so the core-count axis measures what it claims to.

The masks this produces are exactly the 19 the original controller hard-coded:
`0x00001, 0x00401, 0x00403, 0x00C03, …, 0x7FFFF`. Two places now generate them
independently — this script in shell, and
`omp_energy_rl.platform_config.affinity_mask()` in Python — from the same rule.

## Running the controller

```bash
sudo ./launch_control.sh ../bench/two_phase 1000000000 5000000 20000000 19

ENCODER=results/autoencoder/encoder_2bit.h5 \
SCALER=results/autoencoder/encoder_2bit_scaler.npz \
  sudo ./launch_control.sh ../bench/two_phase 1000000000 5000000 20000000 19
```

The workload gets `0x7FFFF` (cores 0–18) and the controller gets `0x80000`
(core 19). Disjoint, and not for tidiness: the controller runs a Keras training
step every 500 ms, and sharing cores with the workload would perturb the very CpS
and CpJ it is measuring — the reward would partly reflect the controller's own
load. This is also why a 20-core machine has a 19-core action space.

`launch_control.sh` sleeps 5 seconds before starting the controller, so the
exploration period is not spent rewarding actions against the workload's
allocation and first-touch transient.
