# Training data

One file: **`srad_allConf_19c11f.csv.gz`** — the exhaustive characterization
sweep of SRAD (Rodinia `srad_v2`) on the Xeon testbed.

| | |
|---|---|
| Configurations | 19 core counts (1–19) × 11 frequencies (1.2–2.2 GHz) = **209** |
| Samples per configuration | 1000, one per 500 ms controller sampling period |
| Rows | **210 000** (~29 hours of measured wall time) |
| Columns | 28, no header |
| Size | 9.1 MB gzipped, 39 MB expanded |

It is the input to [`../train_autoencoder.py`](../train_autoencoder.py). pandas
decompresses `.gz` transparently, so no unpacking step is needed.

## Schema

The file has no header. The column meanings come from the `write()` call in the
original collector, which is the only surviving record of them:

```python
history_file.write(
    str(elapsed) + "," + str(time_sleep) + "," + str(dChunk) + "," +
    str(Power_Tot) + "," + str(CPS) + "," + str(CPJ) + "," + str(nbCPU) + "," +
    str(frequency) + "," + str(f0) + "," + ... + str(f19) + "\n"
)
```

| # | original name | name in the loader | meaning |
|---|---|---|---|
| 1 | `elapsed` | `time_s` | seconds since the sweep started |
| 2 | `time_sleep` | `sleep_s` | how long the collector actually slept (≈ 0.499) |
| 3 | `dChunk` | `chunk` | chunks retired during this period |
| 4 | `Power_Tot` | `joules_total` | Joules consumed during this period (RAPL package + DRAM, both sockets) |
| 5 | `CPS` | `cps` | chunks per second = `dChunk / elapsed_period` |
| 6 | `CPJ` | `cpj` | chunks per Joule = `dChunk / Power_Tot` |
| 7 | `nbCPU` | `n_cores` | cores allocated to the workload |
| 8 | `frequency` | `freq_ghz` | frequency the allocated cores were set to |
| 9–28 | `f0`…`f19` | `f0`…`f19` | per-core frequency readback, one per physical core |

The per-core readbacks are loaded for completeness; the autoencoder uses only
`cps`, `freq_ghz` and `n_cores`. Two things about them are worth knowing before
reading the file by hand:

- **`f19` is constant at 2.201 across all 210 000 rows.** Core 19 is the
  controller's own core — it runs the collector and the training step, is held
  above the workload's range so measurement never becomes the bottleneck, and is
  never part of the allocation. That is why a 20-core machine has a 19-core
  action space.
- **The readback lands one 100 MHz step below the request.** At `freq_ghz` 1.5
  the allocated cores read 1.4; at 2.2 they read 2.1. The first `n_cores` entries
  carry that value and the rest sit at the 1.2 GHz floor:

  ```
  n_cores=5, freq_ghz=1.5  ->  1.4 1.4 1.4 1.4 1.4 1.2 ... 1.2 2.201
  n_cores=19, freq_ghz=2.2 ->  2.1 2.1 ...            2.1 2.201
  ```

  Note the raised entries are the *first* `n_cores` of the array, while the
  controller allocates cores socket-interleaved (`0, 10, 1, 11, …`, see
  `omp_energy_rl.platform_config.core_order`). So `f0…f19` are indexed by
  allocation order, not by CPU id — or this sweep's collector wrote them that
  way. Nothing here depends on the distinction, which is why it was never
  resolved.

Column 4 is named `Power_Tot` in the original but holds **energy**, not power:
it is the RAPL counter *delta* over the period, in Joules. `cpj = chunk /
joules_total` only makes sense read that way.

## What CpS and CpJ mean here

Neither has an absolute scale. They count OpenMP chunks, and a chunk is one
iteration of whatever loop the workload happens to be running, so the numbers are
comparable **across configurations of the same workload** and meaningless across
workloads. Two consequences show up in the code: `ControllerConfig.cps_scale` is
workload-specific, and a trained phase encoder is only valid together with the
`FeatureScaler` fitted on the same trace.

## Not included

The archive this was recovered from also held the Odroid XU3 SRAD sweep, the
synthetic-benchmark characterizations, four more controller runs, and the Intel
PCM statistics. None of them are needed by anything here.
