# Dex1 grippers with SONIC

The deployment controller supports `--hand-type dex3|dex1|none`:

- `dex3` (default): the existing seven-motor hands.
- `dex1`: one-motor Unitree Dex1-1 parallel grippers.
- `none`: no hand command publishers or state subscribers.

The Dex1 driver was ported from MichaelLianggg's `Dex1` branch at
`5c45ee9c8112370c449652f40a9c247ffeb1a69b`, including its corrected opening
direction. It uses `unitree_go::MotorCmds_` / `MotorStates_` on
`rt/dex1/{left,right}/{cmd,state}`. Only the selected backend is initialized.

## SONIC v1.1 with PICO

Keep the normal XRoboToolkit service and PICO manager running as described in
[PICO VR Whole-body Teleop](vr_wholebody_teleop.md). The Python manager command
does not need an extra flag. From `gear_sonic_deploy/`, run:

```bash
bash deploy.sh \
    --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --input-type zmq_manager \
    --hand-type dex1 \
    --motor-kp-scale 4,10=1.5 \
    --motor-kd-scale 4,10=1.5 \
    real
```

The launcher rebuilds the C++ executable before starting. Use your configured
deployment environment (native or Docker) with write access to `build/` and
`target/`; the C++ stack requires a C++20 compiler.

The existing PICO trigger/grip mapping still sends seven-joint hand targets.
The Dex1 driver converts finger flexion into a single closure fraction, handling
both the left and mirrored right hand. It preserves the fork's calibration:
`q=5.1` is fully open and `q=0` is fully closed, with `kp=3.0`, `kd=0.05`.
This assumes the same calibrated Dex1-1 hardware as the source implementation.
The existing hand-close limit (X/C keys) also applies to Dex1.

The seven-slot hand feedback layout is retained: Dex1 motor position/velocity
occupy index 0 and the remaining entries are zero. Hand action fields still
contain the seven input targets, not Dex1 motor angles.

Use the stock `gear_sonic/scripts/pico_manager_thread_server.py` with this
backend. Psi0's separate `real/SONIC/run_pico_manager_dex1.py` adapter also drives
Dex1 directly and must not run a second gripper controller at the same time.

## Simulation scope

This is a driver port only. The supplied MuJoCo model and DDS bridge still use
Dex3. Keep `--hand-type dex3` (or omit the flag) with the stock simulator.
Selecting `dex1` does not replace the simulated hands or provide Dex1 gripper
physics; a separate model and bridge would be needed for that.
