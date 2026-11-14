"""Repeat the headline cases over several seeds and report the mean and range of each outcome.

    python bench.py                 # 5 seeds, 62-second runs, two at a time
    python bench.py --seeds 3 --seconds 30

Each case is one headless `run_teleop.py` run with the scripted operator. Seeds change where the balls
are thrown from and the eye's sensor noise. The operator's own movements are scripted and do not
depend on the seed, so the "no balls" cases vary the wrist's unsteadiness instead.
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
OFF, ON = ["--reflex", "none"], ["--reflex", "eye-fly"]
STILL = ["--source", "mouse"]   # nothing steers: the arm holds its place

# name -> (arguments, what varies between repeats)
CASES = {
    "single balls, no load, reflex off": ([*STILL, "--throw-every", "3", *OFF], "seed"),
    "single balls, no load, reflex on": ([*STILL, "--throw-every", "3", *ON], "seed"),
    "volleys of three, no load, reflex off": ([*STILL, "--throw-every", "4", "--volley", "3", *OFF], "seed"),
    "volleys of three, no load, reflex on": ([*STILL, "--throw-every", "4", "--volley", "3", *ON], "seed"),
    "unsteady operator carrying, no balls, reflex off": (["--load", "tray", *OFF], "wobble"),
    "unsteady operator carrying, no balls, reflex on": (["--load", "tray", *ON], "wobble"),
    "unsteady operator carrying, single balls, reflex off": (["--load", "tray", "--throw-every", "3", *OFF], "seed"),
    "unsteady operator carrying, single balls, reflex on": (["--load", "tray", "--throw-every", "3", *ON], "seed"),
}
WOBBLES = ["0.10", "0.15", "0.20", "0.25", "0.30"]   # radians of wrist rocking


def run(case: str, repeat: int, seconds: int) -> dict:
    arguments, varies = CASES[case]
    vary = ["--seed", str(repeat)] if varies == "seed" else ["--wobble", WOBBLES[repeat % len(WOBBLES)]]
    out = subprocess.run([sys.executable, "-W", "ignore", "run_teleop.py", "--headless", "--seconds", str(seconds),
                          *arguments, *vary], cwd=REPO, capture_output=True, text=True).stdout
    result = {"case": case, "repeat": repeat}
    if m := re.search(r"(\d+) throws, (\d+) touched the arm", out):
        result["throws"], result["hits"] = int(m[1]), int(m[2])
    if m := re.search(r"load dropped (\d+) times", out):
        result["drops"] = int(m[1])
    if m := re.search(r"inverse kinematics restarted (\d+) times", out):
        result["restarts"] = int(m[1])
    if "throws" not in result and "drops" not in result:
        result["error"] = out[-300:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=62)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    jobs = [(case, repeat) for case in CASES for repeat in range(args.seeds)]
    with ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(lambda job: run(*job, args.seconds), jobs))
    (REPO / "runs").mkdir(exist_ok=True)
    (REPO / "runs" / "bench.json").write_text(json.dumps(results, indent=1))

    def cell(values: list[int]) -> str:
        return "-" if not values else f"{sum(values) / len(values):.1f} ({min(values)}-{max(values)})"

    print(f"{args.seeds} repeats of {args.seconds} s each; cells are mean (lowest-highest)\n")
    print(f"{'case':<54}{'balls thrown':>14}{'hit the arm':>16}{'load drops':>16}{'solver restarts':>17}")
    for case in CASES:
        rows = [r for r in results if r["case"] == case]
        bad = [r for r in rows if "error" in r]
        print(f"{case:<54}{cell([r['throws'] for r in rows if 'throws' in r]):>14}"
              f"{cell([r['hits'] for r in rows if 'hits' in r]):>16}{cell([r['drops'] for r in rows if 'drops' in r]):>16}"
              f"{cell([r.get('restarts', 0) for r in rows if 'error' not in r]):>17}" + (f"   {len(bad)} failed" if bad else ""))
        for r in bad:
            print("   failed:", r["error"].strip().splitlines()[-1][:120])


if __name__ == "__main__":
    main()
