# Recorded throws

Ten throws of the orange ball at the catch box on the arm, recorded on 2026-11-25
with the RealSense D435 (serial 832412070191) from the calibrated camera position,
using the RECORD button in `scripts/measure_ball.py` at 640 × 480 and 30 fps. The
arm was held still during each throw. No arm joint data was recorded.

Each `throw_<date>_<time>/` folder contains:

| File | Contents |
|---|---|
| `color.mp4` | Raw color video (MPEG-4 Part 2, no overlays) |
| `depth.npz` | `depth`: aligned depth frames, uint16, one per video frame; multiply by `depth_units_m` for meters |
| `track.csv` | The live tracker's view of every frame: state, ball pixel/radius/camera XYZ and source, box outline and distance, balls in box, events |
| `meta.json` | Camera serial, intrinsics, frame rate, depth units |

Camera XYZ is in the RGB optical frame (x right, y down, z forward, meters). Convert
it to the Franka base frame with `calibration/camera_to_base.json`.

| Throw | Frames | Live tracker had the ball | Tracker saw it enter the box |
|---|---|---|---|
| 183027 | 143 | 15 | no |
| 183039 | 121 | 43 | no |
| 183047 | 113 | 46 | no |
| 183055 | 130 | 0 | no |
| 183102 | 161 | 26 | no |
| 183112 | 138 | 5 | yes |
| 183119 | 150 | 7 | yes |
| 183128 | 124 | 24 | yes |
| 183141 | 147 | 30 | yes |
| 183153 | 70 | 22 | no |

The live tracker loses the ball in flight (it blurs into a streak at 30 fps), so the
counts above are mostly the ball in the hand and near the box. Whether a ball really
went in is best checked in the video.

Use them with:

```sh
# Recreate a throw in MuJoCo (recovers the flight from the video)
python3 scripts/recording_to_sim.py data/videos/throw_20261125_183128

# Run it through the ROS pipeline (vision node -> model -> adapter) in Docker
docker run --rm --network none -v "$PWD":/app:ro -v "$PWD/ros_runs":/out \
    catchbox-ros-vision python3 docker/replay_pipeline.py data/videos/throw_20261125_183128
```

The videos show the lab and the person throwing.

## 2026-11-27: six throws into the box, from the camera stand

Six throws that the live tracker saw end up in the box, recorded on 2026-11-27 with
the same camera on its fixed stand, at **848 × 480 and 60 fps** (RGB auto exposure)
with `scripts/measure_ball.py --fps 60 --ball orange`. Same folder layout as above.

**Use the stand calibration, not the one in this branch:** `calibration/camera_to_base.json`
from `main` at commit `e54f9dc` ("updated calibration", 12 poses, 25 mm RMS check
error). The 2026-11-25 calibration is for the old camera position.

| Throw | Frames | Live tracker had the ball | Positions from depth | Arm |
|---|---|---|---|---|
| 131140 | 230 | 5 | 3 | **not at home** (held about 0.28 m higher) |
| 131159 | 183 | 15 | 8 | **not at home** |
| 131248 | 224 | 20 | 15 | **not at home** |
| 132240 | 177 | 9 | 3 | home |
| 132410 | 80 | 24 | 13 | home |
| 132429 | 198 | 11 | 9 | home |

The arm was not moving during any throw, and no joint data was recorded: a simulation
of the `131…` throws that starts the arm at home does not match the real scene. The
saved frame rate is about 51 fps (the recorder skips frames it has not finished
tracking), and the ball still blurs in flight, so most tracked positions are near the
hand and the box.

Replay a ROS pipeline run of one of these in MuJoCo with
`scripts/ros_run_to_sim.py <run folder> --recording data/videos/throw_... --calibration <stand calibration>`.
