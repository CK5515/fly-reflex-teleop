"""Start the whole demo with one command: glove driver, calibration check, simulation and page.

    python demo.py

It waits for the glove to be plugged in, starts Rokoko's driver if it is not running, checks the
saved calibration with one forward push (running the full calibration only if that fails or none is
saved), then starts the simulation and opens the page. Unplugging and replugging the glove while it
runs is fine: the arm holds still and picks the hand up again.

    python demo.py --no-glove      # the scripted operator instead: no hardware needed
    python demo.py --recalibrate   # force the full calibration
"""

import argparse
import glob
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent
DRIVER = Path(os.environ.get("ROKOKO_SDK_HOME", Path.home() / ".local/share/rokoko-device-sdk")) / "bin/rokoko-sdk"
DRIVER_PORT = 12276
PAGE_PORT = 8000


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_glove_port() -> None:
    """Wait until a glove is plugged in and its serial port can be opened by this user."""
    said = ""
    while True:
        ports = glob.glob("/dev/ttyACM*")
        usable = [p for p in ports if os.access(p, os.R_OK | os.W_OK)]
        if usable:
            print(f"glove found on {usable[0]}")
            return
        message = ("Plug in the glove's USB cable." if not ports else
                   f"The glove is plugged in, but this user may not open {ports[0]}. Run this once in another terminal:\n"
                   f"  sudo chmod a+rw {ports[0]}\n"
                   "To make it permanent (takes effect after logging out and in): sudo usermod -aG dialout $USER")
        if message != said:
            print(message, flush=True)
            said = message
        time.sleep(1.0)


def start_driver() -> subprocess.Popen | None:
    if port_open(DRIVER_PORT):
        print("glove driver already running")
        return None
    if not DRIVER.exists():
        raise SystemExit(f"Rokoko's driver is not installed at {DRIVER}")
    work = REPO / "data" / "driver"
    work.mkdir(parents=True, exist_ok=True)
    log = open(work / "driver.log", "w")
    driver = subprocess.Popen([str(DRIVER), "--auto-stream-usb"], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(50):
        if port_open(DRIVER_PORT):
            print("glove driver started")
            return driver
        if driver.poll() is not None:
            raise SystemExit(f"the glove driver stopped at once; see {work / 'driver.log'}")
        time.sleep(0.2)
    raise SystemExit(f"the glove driver did not come up; see {work / 'driver.log'}")


def prepare_glove(recalibrate: bool) -> None:
    from mocap.calibrate import calibrate, check, load_calibration
    from mocap.glove import GloveHand
    glove = GloveHand()
    said = time.monotonic()
    print("waiting for the Coil Pro to find the glove...", flush=True)
    while not glove.live:
        if time.monotonic() - said > 8:
            said = time.monotonic()
            print("  still waiting: is the Coil Pro switched on, and the glove within about a metre and a half of it?", flush=True)
        time.sleep(0.2)
    print("glove is tracking")
    saved = None if recalibrate else load_calibration()
    if saved is not None and check(glove, saved):
        glove.close()
        return
    print("\nA full calibration is needed (about 30 seconds).")
    while not calibrate(glove):
        print("\nLet's try that again.")
    glove.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-glove", action="store_true", help="use the scripted operator")
    parser.add_argument("--recalibrate", action="store_true", help="run the full calibration even if one is saved")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    args, passed_on = parser.parse_known_args()

    driver = None
    try:
        if not args.no_glove:
            wait_for_glove_port()
            driver = start_driver()
            prepare_glove(args.recalibrate)
            print("\nRest your hand where it is comfortable: the arm starts from there.\n")
        if not args.no_browser:
            threading.Timer(6.0, webbrowser.open, [f"http://127.0.0.1:{PAGE_PORT}"]).start()
        import run_teleop
        sys.argv = ["run_teleop.py", "--view", "--view-port", str(PAGE_PORT),
                    *(["--load", "grab"] if args.no_glove else ["--source", "glove"]), *passed_on]
        run_teleop.main()
    except KeyboardInterrupt:
        pass
    finally:
        if driver is not None:
            driver.terminate()


if __name__ == "__main__":
    main()
