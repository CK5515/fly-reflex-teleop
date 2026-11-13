# 3D Scene in the Live Web Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the MuJoCo scene inside `view/index.html` and let the operator throw balls, orbit the camera and drag the target from the page.

**Architecture:** With `--view`, `run_teleop.py` launches MuJoCo's bundled web viewer (`mujoco.studio`, `gfx="web"`) on the port after the page's port and syncs it from the existing step loop. The page embeds that viewer in a frame and posts throw requests to `LiveView`, which queues them for the loop.

**Tech Stack:** Python 3.12 standard library, `mujoco` 4.0.1 (`mujoco.studio.launch_passive`, `viewer_protocol.ViewerConfig`), plain HTML/JS. Tests with `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-13-3d-scene-in-live-view-design.md`

## Global Constraints

- No new dependencies. The page server stays standard library and stays bound to `127.0.0.1`.
- Without `--view`, behaviour of `run_teleop.py` is unchanged (native window, T/F keys, `--headless`).
- With `--view` there is no native window.
- Viewer port is `--view-port + 1` (8001 by default).
- Time box: if embedding the frame takes more than about an hour, stop and report. Do not start the image-streaming fallback without asking.
- Do not commit; Cyprian has not asked for commits.
- Run everything with `.venv/bin/python` from the repo root.

## Review Focus

- Throw posted with a bad or missing `aim` (`/throw?aim=cube`, `/throw`): expect 400 and nothing queued. Test in Task 2.
- Throw buttons clicked faster than the loop drains: every click is one throw, none lost, none repeated. Test in Task 2.
- Page opened by `localhost` vs `127.0.0.1` vs machine name: the frame must use the same hostname the page was opened with. Checked in Task 3.
- Port 8001 already taken: the run must fail with a clear message, not show an empty frame. Checked in Task 4.
- Browser tab closed and reopened mid-run: the scene and the chart both come back. Checked in Task 4.

---

### Task 1: Browser check (gate)

Throwaway; nothing here is kept. Work in the session scratchpad, not the repo.

- [ ] **Step 1:** Write a scratch script that builds the scene (`sim.scene.build_model`, `reset`), launches the viewer with `launch_passive.launch(viewer_protocol.ViewerConfig(gfx="web", http_port=8001))`, steps physics in real time calling `handle.sync(model, data)` every 8 steps, and prints `data.mocap_pos[0]` once a second. Alongside it serve, on `127.0.0.1:8000`, a one-line page containing `<iframe src="http://HOST:8001/" allow="cross-origin-isolated">` where HOST is the page's hostname, with response headers `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
- [ ] **Step 2:** Open `http://localhost:8000` in a browser. Pass means: the arm scene renders inside the frame, the camera orbits, and dragging the blue target box changes the printed `mocap_pos`. Note which gesture drags the target and whether the viewer shows its own panels.
- [ ] **Step 3:** If the frame stays blank, read the browser console, try `credentialless` on the frame and `Cross-Origin-Embedder-Policy: credentialless` on the page, then re-check. If it still fails, stop and report to Cyprian.

### Task 2: `LiveView` serves the viewer port, isolation headers and a throw queue

**Files:**
- Modify: `view/server.py`
- Test: `tests/test_live_view.py` (create; also create empty `tests/__init__.py`)

**Interfaces:**
- Produces: `LiveView(port: int = 8000, viewer_port: int | None = None)`; `LiveView.take_throws() -> list[str]` returning queued aims (`"gripper"` / `"forearm"`) in arrival order and emptying the queue; `LiveView.close() -> None` shutting the server down (needed by tests).
- HTTP: `GET /` replaces the literal `__VIEWER_PORT__` in `index.html` with the viewer port, or `0` when `viewer_port` is None, and sends the two isolation headers that passed Task 1. `POST /throw?aim=gripper|forearm` returns 204 and queues the aim; any other aim returns 400.

- [ ] **Step 1: Write the failing tests** in `tests/test_live_view.py`, each starting a `LiveView` on a free port and closing it afterwards:

```python
def test_page_carries_viewer_port_and_isolation_headers(self):
    r = urlopen(f"http://127.0.0.1:{self.port}/")
    self.assertEqual(r.headers["Cross-Origin-Opener-Policy"], "same-origin")
    self.assertIn(r.headers["Cross-Origin-Embedder-Policy"], ("require-corp", "credentialless"))
    body = r.read().decode()
    self.assertNotIn("__VIEWER_PORT__", body)
    self.assertIn("9001", body)          # LiveView(port, viewer_port=9001)

def test_throws_queue_in_order_and_drain_once(self):
    for aim in ("gripper", "forearm", "gripper"):
        self.assertEqual(post(f"/throw?aim={aim}").status, 204)
    self.assertEqual(self.view.take_throws(), ["gripper", "forearm", "gripper"])
    self.assertEqual(self.view.take_throws(), [])

def test_bad_aim_is_rejected(self):
    for path in ("/throw?aim=cube", "/throw"):
        with self.assertRaises(HTTPError) as e: post(path)
        self.assertEqual(e.exception.code, 400)
    self.assertEqual(self.view.take_throws(), [])
```

- [ ] **Step 2:** Run `.venv/bin/python -m unittest tests.test_live_view -v`. Expected: all three fail (no `viewer_port`, no `take_throws`).
- [ ] **Step 3:** Implement in `view/server.py`: the constructor argument, `do_POST`, a lock-guarded list for the queue, `take_throws`, `close`, the placeholder substitution and the headers. `/events` behaviour is unchanged.
- [ ] **Step 4:** Run the tests again. Expected: 3 pass.

### Task 3: Scene card and throw buttons on the page

**Files:**
- Modify: `view/index.html`

**Interfaces:**
- Consumes: `__VIEWER_PORT__` substitution and `POST /throw?aim=` from Task 2.

- [ ] **Step 1:** Add a card as the first child of `<main>` after `.top`, heading "The arm", containing a frame (`id="scene"`, full card width, height 480px, no border, plus the `allow` / `credentialless` attributes that passed Task 1) and under it two buttons: "Throw at gripper" and "Throw at forearm". Style the buttons with the existing colour variables.
- [ ] **Step 2:** In the script, set the frame's `src` to `` `http://${location.hostname}:${VIEWER_PORT}/` `` where `VIEWER_PORT = __VIEWER_PORT__`; when it is 0, hide the card. Buttons and the T / F keys call `fetch("/throw?aim=…", {method: "POST"})`.
- [ ] **Step 3:** Verification is in Task 4, once the loop serves the viewer.

### Task 4: Teleop loop drives the web viewer

**Files:**
- Modify: `run_teleop.py`

**Interfaces:**
- Consumes: `LiveView(port, viewer_port)`, `LiveView.take_throws()` from Task 2.

- [ ] **Step 1:** When `args.view` is set, launch the web viewer on `args.view_port + 1` with `open_browser=False` and pass that port to `LiveView`. A taken port must end the run with a message naming the port.
- [ ] **Step 2:** In `step()`, before the scheduled-throw check, call `throw(aim)` for each aim in `view.take_throws()`.
- [ ] **Step 3:** Restructure the run loop into three cases: no `--view` and not headless (native window, unchanged); headless without `--view` (unchanged, as fast as possible); `--view` (no native window, real-time pacing as the current headless-with-view branch, `handle.sync(model, data)` every `round(1 / 60 / model.opt.timestep)` steps, stop when `done()` or the handle is no longer running; close the handle on exit). `--headless` together with `--view` behaves the same as `--view` alone.
- [ ] **Step 4:** Update the module docstring: `--view` now shows the scene in the page with throw buttons, at http://127.0.0.1:8000, and replaces the native window.
- [ ] **Step 5: Verify without a browser.** Run `.venv/bin/python run_teleop.py --view --seconds 12`. Expected: both addresses printed, the same "cube lifted … cm" line as `--headless --seconds 12` gives, clean exit. While it runs, `curl -X POST "http://127.0.0.1:8000/throw?aim=gripper"` must make the final line report 1 throw.
- [ ] **Step 6: Verify in a browser.** Run `.venv/bin/python run_teleop.py --view --source mouse --throw-every 0`. Expected: scene visible in the card; both buttons and the T / F keys throw a ball and the escape or groom reflex shows in the chart; dragging the target moves the arm; reloading the tab brings back scene and chart; opening the page as `127.0.0.1` and as `localhost` both work.
- [ ] **Step 7:** Run `.venv/bin/python run_teleop.py --headless --seconds 12` and confirm the output matches what it printed before this work.
