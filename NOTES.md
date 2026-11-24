# Release notes

This repository was built by copying the files needed for the TEST evaluation out of the research code base
and rewriting paths and names. This file lists the 32 frozen programs and every change made for the release. Program logic was not changed.

## (a) The 32 frozen programs

| robot | task | version | release path |
|---|---|---|---|
| xtrainer | collect_apple | a5 | `programs/xtrainer/collect_apple/controller.py` |
| xtrainer | collect_cubes | c30_drop_memory | `programs/xtrainer/collect_cubes/controller.py` |
| xtrainer | stack_cups | c6 | `programs/xtrainer/stack_cups/controller.py` |
| xtrainer | build_tower | t1 | `programs/xtrainer/build_tower/controller.py` |
| xtrainer | put_mug_on_box | m1 | `programs/xtrainer/put_mug_on_box/controller.py` |
| xtrainer | open_drawer_and_place_object | s1 | `programs/xtrainer/open_drawer_and_place_object/controller.py` |
| xtrainer | sweep_objects | rep1 final (round v8) | `programs/xtrainer/sweep_objects/controller.py` |
| xtrainer | lift_box | b8 | `programs/xtrainer/lift_box/controller.py` |
| lift2 | collect_apple | l10 | `programs/lift2/collect_apple/controller.py` |
| lift2 | collect_cubes | k2 | `programs/lift2/collect_cubes/controller.py` |
| lift2 | stack_cups | s3 | `programs/lift2/stack_cups/controller.py` |
| lift2 | build_tower | st6 | `programs/lift2/build_tower/controller.py` |
| lift2 | put_mug_on_box | r2 (lift2_20261111_r2) | `programs/lift2/put_mug_on_box/controller.py` |
| lift2 | open_drawer_and_place_object | sd7 | `programs/lift2/open_drawer_and_place_object/controller.py` |
| lift2 | sweep_objects | sw5 | `programs/lift2/sweep_objects/controller.py` |
| lift2 | lift_box | bx9 | `programs/lift2/lift_box/controller.py` |
| franka | collect_apple | a5 (xTrainer program, unchanged) | `programs/franka/collect_apple/controller.py` |
| franka | collect_cubes | c30_drop_memory (xTrainer program, unchanged) | `programs/franka/collect_cubes/controller.py` |
| franka | stack_cups | c1 | `programs/franka/stack_cups/controller.py` |
| franka | build_tower | t1 (xTrainer program, unchanged) | `programs/franka/build_tower/controller.py` |
| franka | put_mug_on_box | m1 (xTrainer program, unchanged) | `programs/franka/put_mug_on_box/controller.py` |
| franka | open_drawer_and_place_object | s1 (xTrainer program, unchanged) | `programs/franka/open_drawer_and_place_object/controller.py` |
| franka | sweep_objects | s12 | `programs/franka/sweep_objects/controller.py` |
| franka | lift_box | b1 | `programs/franka/lift_box/controller.py` |
| g1 | collect_apple | apple_b3_r2 | `rsi/eval/g1/runners/apple/apple_b3_r2_seed_eval.py` + `programs/g1/collect_apple/render_guard_b3_r1.py` |
| g1 | collect_cubes | cubes_v2_r1d_x2 | `rsi/eval/g1/runners/cubes/cubes_v2_r1d_x2_seed_eval.py` + `programs/g1/collect_cubes/render_guard_b3_r1.py` |
| g1 | stack_cups | cups_r28 (plane shift -0.045 m) | `programs/g1/stack_cups/cups_r28_ctl.py` + `programs/g1/mug_cups_common/cups_r11_grasp.py` + `programs/g1/common/g1_common_r2.py` |
| g1 | build_tower | tc27 | `programs/g1/build_tower/ctl_build_tower_tc27.py` + `programs/g1/build_tower/g1_common_tc27.py` |
| g1 | put_mug_on_box | mug_r4_cleanpark | `programs/g1/mug_cups_common/mug_r4_ctl.py` |
| g1 | open_drawer_and_place_object | drawer_r9_cleanpark | `programs/g1/drawer_box_sweep/drawer_r9_ctl.py` |
| g1 | sweep_objects | sweep_r33 (best version, not frozen) | `programs/g1/drawer_box_sweep/sweep_r33_ctl.py` |
| g1 | lift_box | box_r19 | `programs/g1/drawer_box_sweep/box_r19_ctl.py` |

Notes:
- Five Franka programs are byte-identical to the xTrainer programs (the Franka `io` facade in
  `rsi/franka_suite/common/facade.py` provides the source-robot interface).
- The xTrainer sweep program is the final round (v8) of a self-improvement run that started from a zero-shot
  Code-as-Policies sample; its first header line still describes that starting sample.
- The G1 per-seed runners and the G1 runner chain (`rsi/eval/g1/runners/*`, `rsi/eval/g1/launch.py`) are the TEST
  evaluation copies of the original DEV runners, with an added `--layout-file/--layout-key` input.

## (b) Changes made for the release

No program logic, scorer, world or layout content was changed. The changes are:

1. **Relative paths.** The repository root is computed at run time (`RESET_ROOT`: from the environment, else the
   nearest parent directory of the file that contains `eval.py`); absolute project paths in Python code were
   rewritten relative to it, and absolute paths in data files were made relative.
2. **MuJoCo XML.** Absolute `file=`, `meshdir=`, `texturedir=` and `assetdir=` paths inside scene and robot XML were
   replaced by paths relative to the XML file (respecting the compiler mesh/texture directory).
3. **Integrity pins.** Runners check hashes of the files they load. Because path strings inside some files changed,
   the hash pins in `build.json` (G1 worlds), `layouts/test/DERIVE_REPORT.json` and the derived layout files were
   recomputed; the numeric layout contents were checked to be identical to the originals.
4. **Scene-mesh file hashes** recorded by the Franka and LIFT2 runners are now resolved relative to the scene
   directory (previously the mesh paths were absolute).
5. **Temporary files.** Lock, socket and other temporary paths now use the system temp directory
   (`tempfile.gettempdir()`) instead of a fixed `/tmp` path.
6. **Interpreters.** Hard-coded interpreter paths were replaced by `sys.executable` or by the environment variables
   `RESET_PYTHON` / `RESET_G1_PYTHON`; the SAM3 checkpoint path is read from `SAM3_CHECKPOINT`.
7. **Franka world set.** The default Franka world set is now the shipped one (`worlds/franka`), which is the set the
   paper numbers were produced with.
8. **G1 launcher.** It no longer accepts only two specific SAM3 socket names; any socket from `RSI_SAM3_ADDR` works.
9. **G1 hang guard.** `eval.py` runs every G1 TEST cell with the 3600 s per-seed wall-clock hang guard. The paper's G1
   lift_box number is the outcome after re-running wall-clock timeouts with the same 3600 s guard.
10. **Removed check.** The xTrainer rigid runner no longer pins the hash of a development script that is not part of
    this release (`run_dev.py` of an earlier tower branch); the pin is recorded as `None`.
11. **Packaging.** Empty `__init__.py` files were added where needed so that `rsi.*` and `twin.*` import as packages;
    `eval.py`, `requirements*.txt` and these documentation files are new.
12. **Names.** Internal names were replaced by the public robot and module names used here, in file names, module
    names, file contents and environment-variable prefixes.
