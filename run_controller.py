#!/usr/bin/env python3
"""Run the online DQN controller against the experimental testbed.

**This requires the physical testbed and will refuse to run anywhere else.** The
environment is a dual-socket Intel Xeon E5-2630 v4 server with a workload built
against a patched GCC/``libgomp``, RAPL MSRs readable as root, and ``cpupower``
driving the ``acpi-cpufreq`` governor. That machine is 2016-era hardware and is no
longer available; nothing here is simulated, on purpose (see
the README's 'What survived, and what did not' section).

The script is kept, and kept complete, because it is the only place the whole
loop is assembled: instrumentation -> CpS/CpJ -> phase code -> Q-network ->
configuration change. Read it alongside the README's 'The testbed' section.

On the testbed, with a workload already running (see
``platform/launch_control.sh``)::

    sudo taskset 0x80000 python run_controller.py --pid $(pidof -s ./mixed_cxmy)

Off the testbed it prints exactly which preconditions are unmet and exits 1.
Use ``--check`` to ask that question without providing a PID.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
RESULTS = REPO_ROOT / "results"

from omp_energy_rl import platform_config as plat
from omp_energy_rl.agent import PRESETS, Agent
from omp_energy_rl.control import Controller, ControllerConfig, PhaseDetector
from omp_energy_rl.data import FeatureScaler
from omp_energy_rl.environment import HardwareUnavailable, XeonEnvironment, check_preconditions

HISTORY_COLUMNS = [
    "elapsed_s", "sleep_s", "t_act_s", "t_autoencoder_s", "t_dqn_s", "energy_j",
    "cps", "cpj", "n_cores", "freq_ghz", "action", "reward", "epsilon", "exploring",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pid", type=int, help="PID of the OpenMP workload to control")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether this machine is the testbed, then exit",
    )
    parser.add_argument(
        "--preset",
        default="thesis-table-4.3",
        choices=sorted(PRESETS),
        help="agent hyperparameters (default: thesis-table-4.3)",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help="phase encoder (.h5) -- switches the controller to the phase-aware "
        "state definition of thesis Fig. 4.17",
    )
    parser.add_argument(
        "--scaler",
        default=None,
        help="FeatureScaler (.npz) the encoder was trained under. Required with "
        "--encoder: an encoder without its scaler emits meaningless codes.",
    )
    parser.add_argument("--steps", type=int, default=ControllerConfig.total_steps)
    parser.add_argument(
        "--cps-scale",
        type=float,
        default=ControllerConfig.cps_scale,
        help="CpS normalization for the phase-free state; workload-specific, "
        "derive it from that workload's characterization sweep",
    )
    parser.add_argument("--out-dir", default=str(RESULTS / "controller"))
    args = parser.parse_args()

    problems = check_preconditions()
    if args.check:
        if problems:
            print("This machine is NOT the experimental testbed:")
            for problem in problems:
                print(f"  - {problem}")
            print("\nSee the README's 'The testbed' section for what the testbed was.")
            return 1
        print("All testbed preconditions met.")
        return 0

    if args.pid is None:
        parser.error("--pid is required (or pass --check)")

    if args.encoder and not args.scaler:
        parser.error(
            "--encoder requires --scaler: the phase code depends entirely on the "
            "min-max scaling the encoder was fitted under"
        )

    try:
        environment = XeonEnvironment(workload_pid=args.pid)
    except HardwareUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1

    print(environment.describe())

    phase_detector = None
    if args.encoder:
        from omp_energy_rl.autoencoder import load_pretrained_encoder

        phase_detector = PhaseDetector(
            load_pretrained_encoder(args.encoder), FeatureScaler.load(args.scaler)
        )
        print(f"phase detection: {args.encoder} ({phase_detector.code_size}-bit code)")

    agent = Agent(PRESETS[args.preset])
    print(f"agent: {agent.config.name} -- {agent.config.provenance}")
    print(
        f"exploration: {agent.config.observe_period} steps "
        f"({agent.config.observe_period * plat.SAMPLING_PERIOD_S:.0f} s)"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / f"history_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    handle = open(history_path, "w", newline="")
    writer = csv.writer(handle)
    code_size = phase_detector.code_size if phase_detector else 0
    writer.writerow(HISTORY_COLUMNS + [f"phase_bit{i}" for i in range(code_size)])

    def record(row) -> None:
        writer.writerow(
            [
                row.elapsed_s, row.sleep_s, row.t_act_s, row.t_autoencoder_s,
                row.t_dqn_s, row.energy_j, row.cps, row.cpj, row.n_cores,
                row.freq_ghz, row.action, row.reward, row.epsilon, int(row.exploring),
            ]
            + list(row.phase_code or [])
        )
        handle.flush()

    controller = Controller(
        environment,
        agent,
        phase_detector=phase_detector,
        config=ControllerConfig(total_steps=args.steps, cps_scale=args.cps_scale),
        on_record=record,
    )

    print(f"\nrunning {args.steps} steps -> {history_path}")
    try:
        controller.run()
    finally:
        handle.close()
        agent.save(str(out_dir / "q_network.h5"))
    print(f"wrote {history_path} and {out_dir / 'q_network.h5'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
