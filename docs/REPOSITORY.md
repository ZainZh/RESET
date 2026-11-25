# Repository layout

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

## G1 program entry modules

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
