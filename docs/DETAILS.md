# Evaluation details and disclosures

## Running the evaluation

Start the SAM3 server ([SAM3_SERVICE.md](SAM3_SERVICE.md)), then from the repository root:

```bash
# one cell, all 50 TEST layouts
RSI_SAM3_ADDR=<sam3 socket> RESET_EGL_DEVICES=0 .venv/bin/python eval.py --robot franka --task stack_cups --layouts test

# a subset of layouts, more parallel episodes
RSI_SAM3_ADDR=<sam3 socket> .venv/bin/python eval.py --robot g1 --task lift_box --layouts test --seeds 0-9 --workers 4

# all 32 cells, one after another
RSI_SAM3_ADDR=<sam3 socket> .venv/bin/python eval.py --all --layouts test --workers 3

# print the table of the latest TEST run per cell next to the paper numbers
.venv/bin/python eval.py --table
```

Options: `--seeds` takes layout indices such as `0-49` or `0,5,7` (default `0-49`); `--workers` is the number of
parallel episodes per cell (default 4); `--name` sets the run directory name (default
`<split><first>_<last>_<timestamp>`). `--layouts dev` is only available for `xtrainer` (DEV layouts are sampled on
the fly); for the other robots only the TEST layouts are shipped.

Environment variables:

| variable | meaning | default |
|---|---|---|
| `RSI_SAM3_ADDR` | unix socket of the running SAM3 server | `<system temp dir>/sam3.sock` |
| `RESET_EGL_DEVICES` | comma-separated EGL device indices used for rendering | `0` (`eval.py` passes it to every robot; if you call `rsi/eval/g1/launch.py` directly its own default is `0,1`, so set it on a single-GPU machine) |
| `RESET_PYTHON` | interpreter for xtrainer / franka / lift2 | interpreter running `eval.py` |
| `RESET_G1_PYTHON` | interpreter for g1 | `.venv-g1/bin/python` |

`eval.py` sets `MUJOCO_GL=egl`, single-threaded BLAS, `RESET_ROOT` (the repository root) and `RSI_UNSEAL_TEST=1`
for the child processes.

**Where results land.** Each cell run writes `results/<robot>/<task>/<run>/`: one directory or record per episode
(`result.json`, or `seed_XX.json` for G1, plus logs and a snapshot of the program that was run) and a
`summary.json`. `eval.py` appends one line per cell run to `results/RUNS.jsonl`, which `--table` reads. Existing run
directories are never overwritten.

**Runtime.** Roughly 30 s to 5 min of wall-clock time per episode on one CPU core, depending on the cell; G1 is the
slowest robot (build_tower and collect_cubes in particular), and a few LIFT2 cells (put_mug_on_box, lift_box) are
also slow. Each runner enforces a per-episode wall-clock guard (LIFT2 900-2400 s, Franka 2400 s, G1 3600 s); an
episode that hits it counts as a failure. As a reference, 10 layouts with 4 workers took between about 1 and 12
minutes per cell on our machine.

**Reproducibility.** Layouts, programs, worlds and scorers are fixed, but simulation episodes are not bitwise
reproducible across machines, GPU drivers and CPU instruction sets (rendering and contact solving produce slightly
different floating-point results, and SAM3 runs on the GPU). Individual seeds near a decision boundary can flip
between success and failure, so expect per-cell rates close to, not identical with, the table above.

## Known issues

- **G1 Apple and Cubes, render guard.** These programs wait until consecutive camera renders agree. On some GPU
  drivers multisampled (MSAA) RGB readback differs by a few values from call to call, so the release accepts RGB when
  at most 0.1% of its values differ (`NOTES.md`, change 13).
- **No SAM3 server.** If nothing listens on `RSI_SAM3_ADDR`, every episode fails within about a second with a
  `FileNotFoundError` and is counted as a failure. Start the server first and check that it printed
  `SAM3_SERVICE ready`.
- **Memory.** Peak memory per episode process measured on a 64 GB machine: xTrainer and Franka 0.5-0.9 GB; LIFT2
  9-16 GB on collect_apple, build_tower, open_drawer and put_mug_on_box (short spikes), 0.7 GB on sweep_objects;
  G1 about 6 GB, build_tower 7.5 GB. The SAM3 server uses about 4 GB of GPU memory. Running out of memory kills
  episodes (and, depending on the system, other processes in the same session).

## Disclosures

- **G1 is not walking or balancing.** The G1 pelvis, legs and waist are fixed (no free joint) and the robot body has
  gravity compensation; only arms and hands are actuated. The numbers say nothing about locomotion or balance.
- **G1 per-cell details.** stack_cups runs on layouts shifted by -0.045 m along the table plane (as frozen);
  collect_cubes uses a world variant with the table 0.08 m higher and 0.07 m closer (`worlds/g1/b3_cubes_v2_x2`);
  sweep_objects uses `worlds/g1/b3_sweep_v1` and the best, not formally frozen, version (sweep_r33). G1 TEST runs use
  a 3600 s wall-clock hang guard per seed for all cells.
- **xTrainer TEST layouts** are generated on the fly by `rsi/layouts.py` from the TEST seed stream, which is sealed
  by default (the sampler refuses the TEST split unless `RSI_UNSEAL_TEST=1`; `eval.py` sets it). The frozen copies
  of these layouts are in `layouts/test/source/`; the Franka, LIFT2 and G1 TEST layouts are read from
  `layouts/test/<robot>/`.
- **What programs may read.** At runtime a program uses rendered RGB-D from the calibrated camera(s), the camera
  calibration, joint encoders, the commanded gripper state, the robot model, the task instruction, SAM3 masks and
  demonstration priors. It never reads simulator object state, contacts or the scorer; scorers are separate and were
  frozen before practice.
- **Demonstration-derived contact priors.** The demonstration priors exposed to programs (`io.demo`, built by
  `rsi.env.demo_priors` and re-expressed per robot) are the per-arm TCP pose at each gripper close/open event of the
  demonstration and the demonstration object positions at those events. Per task, the programs use them as follows
  (see each program's header for the exact list):
  - collect_apple: grasp offset of the hand relative to the apple and release height above the bowl.
  - collect_cubes: top-down two-finger pinch of the 32 mm cubes (demo close/open rotations tried first).
  - stack_cups: which cup is grasped (demo close event) and a top grasp of that cup; on G1 also the `mug_wide` hand
    shape and cup dimensions.
  - build_tower: grasp rotations at the demo close/open events (G1: 32 mm cube grasp profile).
  - put_mug_on_box: mug grasp (wall grasp depth below the rim) and the hand-over spot; on G1 the `mug_wide` hand
    shape and mug dimensions.
  - open_drawer_and_place_object: knob grasp rotation of the left arm, knob height above the table (LIFT2), right-arm
    close/open rotations for the block.
  - sweep_objects: initial demo positions of dustpan, brush and cube, the right-arm brush grasp, the left-arm dustpan
    grasp (Franka/LIFT2: re-expressed relative to the held dustpan).
  - lift_box: grasp rotation for the box (G1: grasp across the carton faces with a sphere-68 mm pinch profile).
  G1 grasp profiles (`(kind, size)` per hand) are installed on the trusted side and are listed in
  `rsi/eval/g1/launch.py` (`--profiles`) and `programs/g1/common/g1_common_r2.py`.
- Simulation only. No result here establishes real-robot performance.
