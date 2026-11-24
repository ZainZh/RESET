# RESET: simulation evaluation release

**RESET: One Demonstration, One Verified World, Self-Improved Skills Across Embodiments**
[Project page](https://zainzh.github.io/RESET/) &middot; [Paper (PDF)](https://zainzh.github.io/RESET/assets/RESET.pdf)

RESET turns one real demonstration of a bimanual tabletop task into a rebuilt MuJoCo world of that task. An agent then
improves a code program (perception + motion) by practising in that world, and the frozen program is evaluated on unseen
layouts. This repository contains the **simulation evaluation stage only**: the rebuilt worlds, the frozen programs,
the episode runners and the frozen task scorers needed to re-run the TEST numbers of the paper (4 robots x 8 tasks =
32 cells, TEST layouts 0-49 per cell). The agent pipeline itself (Rebuild / Examine / Self-improve), the real
demonstrations and the real-robot stack are **not** part of this release.

| robot | Apple | Cubes | Stack Cups | Build Tower | Mug-Box | Open Drawer | Sweep | Lift-Box | mean |
|---|---|---|---|---|---|---|---|---|---|
| xTrainer (source) | 100 | 98 | 96 | 100 | 100 | 100 | 88 | 94 | 97 |
| Dual-arm Franka | 94 | 98 | 100 | 100 | 94 | 100 | 78 | 100 | 96 |
| ARX LIFT2 | 86 | 76 | 90 | 92 | 84 | 70 | 58 | 80 | 80 |
| Unitree G1 | 86 | 70 | 100 | 70 | 80 | 100 | 32 | 80 | 77 |

TEST success rate in %, 50 layouts per cell. Task ids used on the command line:
Apple = `collect_apple`, Cubes = `collect_cubes`, Stack Cups = `stack_cups`, Build Tower = `build_tower`,
Mug-Box = `put_mug_on_box`, Open Drawer = `open_drawer_and_place_object`, Sweep = `sweep_objects`, Lift-Box = `lift_box`.
Robot ids: `xtrainer`, `franka`, `lift2`, `g1`.

## Requirements

- Linux (tested on Ubuntu, x86-64).
- NVIDIA GPU with EGL for headless rendering. Every episode renders RGB-D from the calibrated camera(s) with
  `MUJOCO_GL=egl`; `configs/egl/10_nvidia.json` selects the NVIDIA EGL vendor library. No display is needed.
- One GPU for the SAM3 perception service, with enough memory for the SAM3 image model in bfloat16
  (plan for several GB; one service is shared by all episodes). The service and the simulation can share a GPU.
- CPU cores: episodes run one per process, single-threaded (`OMP_NUM_THREADS=1`); `--workers N` runs N in parallel
  per cell, so throughput scales with cores.
- Python 3.12 for `xtrainer`, `franka`, `lift2`, and Python 3.11 for `g1`.

Why two environments: the G1 programs were developed and frozen with MuJoCo 3.3.7, the other three robots with
MuJoCo 3.2.3. Contact dynamics differ between MuJoCo releases, so we keep, per robot, exactly the versions the
reported numbers were produced with (`requirements.txt`, `requirements-g1.txt`).

## Installation

```bash
python3.12 -m venv .venv    && .venv/bin/pip install -r requirements.txt
python3.11 -m venv .venv-g1 && .venv-g1/bin/pip install -r requirements-g1.txt
```

`eval.py` uses `.venv-g1/bin/python` for G1 by default (override with `RESET_G1_PYTHON`) and the interpreter that runs
`eval.py` for the other robots (override with `RESET_PYTHON`).

## Perception service (SAM3)

The runtime programs segment objects with text prompts through SAM3. They do not load SAM3 themselves: every episode
connects to one shared SAM3 server over a unix socket. The socket path is taken from `RSI_SAM3_ADDR`; the connection
authkey is fixed in code (`b"rsi-sam3"`, see `rsi/sam3_service.py` and `rsi/perception_sam3.py`). SAM3 code and
weights are **not** redistributed here.

### Setting up the provided server

1. Create a separate environment for the server (it needs PyTorch with CUDA; versions we used are in
   `requirements-sam3.txt`):
   ```bash
   python3.12 -m venv .venv-sam3
   .venv-sam3/bin/pip install -r requirements-sam3.txt
   ```
2. Install SAM3 from the official repository, https://github.com/facebookresearch/sam3 (we used commit
   `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`), following its instructions, e.g.
   `git clone https://github.com/facebookresearch/sam3 && .venv-sam3/bin/pip install -e ./sam3`.
3. Download the SAM3 checkpoint `sam3.pt` from the official source after accepting the SAM License.
4. Start the server from the repository root (it imports `rsi.perception_sam3`, which also needs numpy and OpenCV):
   ```bash
   SAM3_CHECKPOINT=$SAM3_CHECKPOINT RSI_SAM3_ADDR=<sam3 socket> PYTHONPATH=. \
       .venv-sam3/bin/python -m rsi.sam3_service
   ```
   It prints `SAM3_SERVICE ready <sam3 socket>` when it accepts connections. Use the same `<sam3 socket>` value
   for `eval.py` (the two programs have different built-in defaults, so always set `RSI_SAM3_ADDR` explicitly).

The server builds the SAM3 image model (`build_sam3_image_model`, `Sam3Processor` with confidence threshold 0.3),
handles each client connection in its own thread, and runs inference under one lock with bfloat16 autocast.

### Replacing it

Any server that speaks the same protocol can be used instead (e.g. another segmentation model, or a server on a
different machine behind a forwarded unix socket):

- Transport: `multiprocessing.connection.Listener(<sam3 socket>, "AF_UNIX", authkey=b"rsi-sam3")`; messages are
  Python objects pickled by `Connection.send` / `Connection.recv`. One persistent connection per episode process.
- Request: a tuple `(rgb, prompts)`; `rgb` is a `numpy.uint8` array of shape `H x W x 3` (the rendered camera image),
  `prompts` a list of text prompts (the current client sends one prompt per request, e.g. `"apple"`, `"bowl"`,
  `"small cube"`).
- Response: a list with one entry per prompt, each a tuple `(masks, scores)`: `masks` is a boolean array
  `N x H x W` at the full image resolution (one instance mask per detection), `scores` a float array of length `N`.
  With no detection, return `(np.zeros((0, 1, 1), bool), np.zeros(0))`. The reference server returns every
  instance with score >= 0.3; the programs apply their own, higher thresholds on top.

Results obtained with a different segmenter are of course not the paper's numbers.

## Reproducing the table

Start the SAM3 server (above), then from the repository root:

```bash
# one cell, all 50 TEST layouts
RSI_SAM3_ADDR=<sam3 socket> RESET_EGL_DEVICES=0 .venv/bin/python eval.py --robot franka --task stack_cups --layouts test

# a subset of layouts, more parallel episodes
RSI_SAM3_ADDR=<sam3 socket> .venv/bin/python eval.py --robot g1 --task lift_box --layouts test --seeds 0-9 --workers 8

# all 32 cells, one after another
RSI_SAM3_ADDR=<sam3 socket> .venv/bin/python eval.py --all --layouts test --workers 16

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
| `RSI_SAM3_ADDR` | unix socket of the running SAM3 server | `<system temp dir>/reset_sam3.sock` |
| `RESET_EGL_DEVICES` | comma-separated EGL device indices used for rendering (xtrainer, franka, lift2) | `0` |
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

## Repository layout

| path | contents |
|---|---|
| `eval.py` | single entry point: runs a cell with the robot's own runner, collects successes, prints the table |
| `rsi/` | runtime library: `env.py` (episode IO for the source robot: observe / tcp / move / grip / demo priors), `scorer.py`, `tasks.py` (task specs, privileged), `layouts.py` (layout sampler), shared perception (`perception.py`, `perception_geo.py`, `perception_sam3.py`), the SAM3 server (`sam3_service.py`) |
| `rsi/eval/<robot>/` | trusted episode runners per robot (`xtrainer/` one module per task family, `franka/runner.py`, `lift2/runner.py`, `g1/launch.py` + `g1/runners/`) |
| `rsi/franka_suite/`, `rsi/lift2_suite/`, `rsi/g1_suite/` | per-robot suites: robot model and world builder, kinematics, the `io` facade that gives the source-robot program interface on the target robot, per-robot scorers |
| `rsi/xtrainer_sweep/`, `rsi/xtrainer_lift_box/` | xTrainer environment and scorer variants for the sweep and lift-box tasks |
| `twin/` | the MuJoCo motor world of the source robot (scene assembly, `twin/robot.py`) and the motion library used by the programs (`twin/practice/`: encoder-only FK/IK, motion, runtime, scoring helpers) |
| `programs/<robot>/<task>/controller.py` | the 32 frozen programs (xtrainer, franka, lift2) |
| `programs/g1/` | the G1 programs, as multi-module packages (see below) |
| `worlds/<robot>/<task>/` | the rebuilt scenes (`scene.xml`, object meshes, demonstration reference and build records); G1 worlds are grouped by build: `worlds/g1/b3/<task>`, `worlds/g1/b3_cubes_v2_x2/collect_cubes`, `worlds/g1/b3_sweep_v1/sweep_objects` |
| `robots/` | robot models: `robots/g1` (Unitree G1 with BrainCo hands), `robots/xtrainer` (xTrainer arm meshes, collision parts, reference URDF); the Franka and LIFT2 models live in `rsi/franka_suite/model` and `rsi/lift2_suite/common/models/lift2` |
| `assets/ycb/` | YCB object textures used by the worlds |
| `layouts/test/` | the exact TEST 0-49 layouts per robot and task (`source/`, `franka/`, `lift2/`, `g1/`) and `DERIVE_REPORT.json` (how the target-robot layouts were derived from the source layouts) |
| `configs/` | EGL vendor file and practice scoring config |
| `results/` | evaluation outputs (see above) |

### G1 program entry modules

The G1 launcher (`rsi/eval/g1/launch.py`, table `CELLS`) starts a per-seed runner which puts the program folders on
`sys.path` and imports the entry module:

| task | version | entry module | also uses |
|---|---|---|---|
| collect_apple | apple_b3_r2 | `controller()` in `rsi/eval/g1/runners/apple/apple_b3_r2_seed_eval.py` (program and runner are one file, as frozen) | `programs/g1/collect_apple/render_guard_b3_r1.py` |
| collect_cubes | cubes_v2_r1d_x2 | `controller()` in `rsi/eval/g1/runners/cubes/cubes_v2_r1d_x2_seed_eval.py` (same) | `programs/g1/collect_cubes/render_guard_b3_r1.py` |
| stack_cups | cups_r28 (layouts plane-shifted by -0.045 m) | `programs/g1/stack_cups/cups_r28_ctl.py` | `programs/g1/mug_cups_common/cups_r11_grasp.py`, `programs/g1/common/g1_common_r2.py` |
| build_tower | tc27 | `programs/g1/build_tower/ctl_build_tower_tc27.py` | `programs/g1/build_tower/g1_common_tc27.py` |
| put_mug_on_box | mug_r4_cleanpark | `programs/g1/mug_cups_common/mug_r4_ctl.py` | `mug_r1_grasp.py`, `calib_wide_r1.json`, `programs/g1/common/g1_common_r2.py` |
| open_drawer_and_place_object | drawer_r9_cleanpark | `programs/g1/drawer_box_sweep/drawer_r9_ctl.py` | `drawer_r8_ctl.py`, `tdbs_common_r1.py` |
| lift_box | box_r19 | `programs/g1/drawer_box_sweep/box_r19_ctl.py` | `box_r17/r13/r11/r10/r5/r4_ctl.py`, `drawer_r8_ctl.py`, `tdbs_common_r1.py` |
| sweep_objects | sweep_r33 | `programs/g1/drawer_box_sweep/sweep_r33_ctl.py` | `drawer_r8_ctl.py`, `tdbs_common_r1.py` |

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

See `NOTES.md` for the list of programs and the changes made for this release, and `THIRD_PARTY.md`
for third-party components and their licences. Licence of this repository: MIT (see `LICENSE`).

## Citation

```bibtex
@article{sun2026reset,
  title   = {RESET: One Demonstration, One Verified World, Self-Improved Skills Across Embodiments},
  author  = {Sun, Zheng and Li, Zhuo and Li, Zhihao and Sim, Heng Yi and Wang, Ge and Zhu, Ziyue and Jia, Mingkai and Chen, Fei},
  journal = {arXiv preprint},
  year    = {2026}
}
```
