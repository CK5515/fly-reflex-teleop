# 3D scene in the live web page

## Intent

The live page becomes the one screen the demo is driven from: the 3D arm scene sits next to the control state
and the reflex-drive chart, and the operator throws balls, moves the camera and drags the target there instead
of in the separate MuJoCo window.

## Goals, non-goals, anti-goals

- Goals: 3D scene visible in the page; orbit camera; throw at gripper / forearm from the page; drag the blue
  target from the page when `--source mouse`.
- Non-goals: a custom renderer, custom look for the scene, hiding every panel of MuJoCo's viewer.
- Anti-goals: anything that eats brain-model time. This is a prototype; if it fights back for more than about an
  hour, stop and report.

## Approaches

- **A. Embed MuJoCo's bundled web viewer (picked).** `mujoco.studio` 4.0.1 ships a browser viewer that renders
  client-side. Least code, real 3D, camera and dragging come with it. Picked by Cyprian as good enough for a
  prototype.
- B. Render on the server and stream images. About 15 fps here (70 ms per 640x480 frame), needs a JPEG
  dependency, and camera and dragging would be hand-written. Kept as the fallback if A cannot be embedded.
- C. Own three.js scene fed with body poses. Smoothest and fully controllable, but the largest build. To be
  tried later only if time allows.

## Design

- `--view` starts two servers: the existing page on `--view-port` (8000) and MuJoCo's web viewer on the next
  port (8001). With `--view` there is no native window; the loop steps in real time and calls the viewer
  handle's `sync(model, data)` about 60 times a second. Without `--view`, behaviour is unchanged.
- The page gets a new card, first on the page, holding the viewer in a frame, with two buttons under it:
  "Throw at gripper" and "Throw at forearm" (T and F keys also work when the page, not the frame, has focus).
- The buttons POST to `/throw?aim=gripper|forearm` on the page's server. `LiveView` queues these; the teleop
  loop drains the queue each step and calls the existing `throw()`.
- The page's server sends the cross-origin isolation headers the viewer needs to run inside a frame. The frame
  address is built in the page from the current hostname and the viewer port, which the server exposes.
- Dragging the target uses the viewer's own object dragging; `sync` writes the result back into `data`, the same
  way the native viewer does today.

## Known limits (accepted for the prototype)

- MuJoCo's viewer listens on all network interfaces with no password and the library has no setting for this.
  Anyone on the same network can open it. The page's own server stays on 127.0.0.1.
- First load downloads a 35 MB model bundle.
- MuJoCo's viewer may show its own panels inside the frame.

## Order of work and what is left to the builder

1. First, in a real browser: confirm the viewer renders inside a frame on the page and that the target can be
   dragged. If the frame cannot be made to work, stop and come back before starting B.
2. Then the throw buttons, the loop changes, and the docstring in `run_teleop.py`.

Left to the builder: card layout and frame height, the exact sync rate, how the port reaches the page, thread
safety of the throw queue.
