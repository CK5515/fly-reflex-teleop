# Submission video rough cut — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 2:00, 1080p30 rough cut of the submission video. Every `[REC-n]` shot is a black placeholder card, and the
voiceover is scratch text-to-speech for timing.

**Architecture:**
- One hyperframes project in `~/WEIRD PROJECTS/fly-reflex-video/`. `index.html` is the main timeline, and each
  storyboard beat is a sub-composition in `compositions/`.
- Sim footage is captured from the real demo page (`run_teleop.py --view`, scripted operator) by headless Chrome into
  `footage/`.
- Existing recordings are cut into `footage/` with ffmpeg.
- The low-poly scenes are three.js inside sub-compositions.

**Tech stack:** hyperframes 0.9 (GSAP, three.js adapter), headless Chrome via puppeteer, ffmpeg 6.1, Python sim from
the repo.

**Spec:** `docs/superpowers/specs/2026-11-14-submission-video-design.md`

## Global Constraints

- 1920×1080, 30 fps, total `data-duration="120"`.
- Brand: page `#0c0c0c`, surface `#151515`, line `#2a2a2a`, text `#f2f2f2`, muted `#b8b8b8`, glove `#ffffff`,
  reflex amber `#e9a23b`, threat red `#ff3b30`. Inter for type and monospace for numbers, with fonts shipped locally
  using `@font-face`.
- Placeholder card: background `#0c0c0c`. Top-left label `REC-n · FULL` or `REC-n · PIP` in monospace amber. Then the
  shot description from the spec's shot list in Inter, `#f2f2f2`, at most 60% of the width.
- No venue audio: all source clips are muted.
- Voiceover text exactly as in the spec's storyboard.
- On-screen numbers: "36 → 0" load drops, "20 of 20 → 4.6" hits.

## Review Focus

1. **Sim capture frame rate.** Headless screencast frames come at irregular times. The output must still be CFR
   30 fps with the sim's real speed preserved: a 20 s run gives 20 s of video ±0.2 s.
2. **WebGL in headless Chrome without a GPU.** If the page's 3D view renders black, capture with
   `--use-angle=swiftshader`. Check a frame for a non-black arm region.
3. **Sub-composition mounts.** Every beat must show at its midpoint (`hyperframes snapshot`), with no unstyled text.
4. **Voiceover overrunning its beat.** Each TTS line must fit its beat's window. Check each clip's duration with ffprobe.
5. **Port 8000 already in use** by a running demo. The capture script takes `--port`.

---

### Task 1: Project scaffold, brand and placeholder card

**Files:** create `fly-reflex-video/` with `npx hyperframes init`, `assets/brand.css` (tokens and `@font-face`),
`compositions/placeholder.html` (variables `label`, `text`).

- [ ] Init the project, copy Inter and a monospace font into `assets/fonts/`, and write `brand.css`.
- [ ] Write `placeholder.html` as a sub-composition with `data-composition-variables` for `label` and `text`.
- [ ] Verify: `npx hyperframes check` passes, and a snapshot at 1 s shows the label and text.

### Task 2: Sim footage capture

**Files:** create `fly-reflex-video/tools/capture_page.mjs`, `tools/capture_sim.sh`.

**Interfaces:** `node tools/capture_page.mjs --url http://127.0.0.1:<port> --seconds <s> --out <file.mp4> [--keys "<t>:<key>,..."] [--crop x,y,w,h]`.
It records with a CDP screencast at a 1920×1080 viewport, writes the frames with their timestamps, and makes a CFR
30 fps file with the ffmpeg concat demuxer.

- [ ] Write the capture script. Verify on a 5 s run: duration 5.0 ± 0.2 s and a frame that isn't black.
- [ ] `capture_sim.sh` renders these from the scripted operator with a fixed `--seed 0`:
  - `carry_off.mp4` and `carry_on.mp4`: `--load grab --wobble 0.2`, reflex `none` vs default, 18 s each.
  - `throw_off.mp4` and `throw_on.mp4`: `--throw-every 3`, still arm, 15 s each.
  - `panel_on.mp4`: the throw run cropped to the looming-circuit panel.
- [ ] Verify each file's duration and look at a contact sheet of each.

### Task 3: Cuts from existing footage

**Files:** create `tools/cut_existing.sh`, writing to `footage/`.

- [ ] From `~/2026-11-14 12-37-11.mp4`, cut the page area (crop to the browser viewport, scaled to 1920×1080):
  `paddle.mp4` (the paddle carry with the "Fly reflex" handover), `blocks.mp4` (stacking). Find the exact times from
  a 1 fps contact sheet of 200–330 s.
- [ ] From `~/2026-11-14 12-28-52.mp4`, cut `venue.mp4` (5 s, the user looking at the screen).
- [ ] All muted, 30 fps, yuv420p.

### Task 4: Low-poly 3D scenes

**Files:** create `compositions/coldopen.html` (beat 1, 9 s), `compositions/flyescape.html` (beat 6, first 10 s).

- [ ] Cold open: a three.js scene with flat-shaded geometry. A low-poly arm made of boxes and cylinders carries a tray
  holding a glass sphere through a dim floor grid. The tray tilts, the sphere rolls to the edge, and the image
  freezes at 8 s.
- [ ] Fly escape: a low-poly fly made of an icosahedron body, a red eye and plane wings. A red disc grows toward it,
  amber facets light up, and the fly leaps sideways at 7 s.
- [ ] All motion is driven from the GSAP timeline (seekable): no rendering on `requestAnimationFrame` clocks.
- [ ] Verify with snapshots at 2, 5 and 8 s for each.

### Task 5: Graphics beats

**Files:** create `compositions/title.html`, `compositions/compare.html` (variables: left/right video, label, number
text), `compositions/limits.html`, `compositions/endcard.html`, and `compositions/caption.html` (a lower-third chip).

- [ ] Write each one, using brand tokens only.
- [ ] Verify with `check` and snapshots.

### Task 6: Scratch voiceover

**Files:** create `audio/vo_<beat>.wav`.

- [ ] Use `npx hyperframes tts` (or espeak-ng as a fallback) on each storyboard line.
- [ ] Verify that each line's duration is no longer than its beat window, and tighten the timing if needed.

### Task 7: Assemble and render the rough cut

**Files:** create `index.html`, and output `renders/roughcut_v1.mp4`.

- [ ] Place beats 1–9 at the spec's times. Beat 4 is `paddle.mp4`/`blocks.mp4` with REC-2/3 placeholders inset as
  PIPs. Beat 7 is a split with a REC-4 placeholder on the left and `throw_on.mp4` on the right. Then the
  `compare.html` off/on beats, and the voiceover on the audio track.
- [ ] Verify: `npx hyperframes check` gives 0 findings, and snapshots at each beat's midpoint look right.
- [ ] Render with `--quality draft`, check that ffprobe gives 120.0 s at 30 fps, and make a contact sheet.
