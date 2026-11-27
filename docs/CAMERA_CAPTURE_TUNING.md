# Checking motion blur on the D435

Keep the robot motion disabled while testing the camera. The existing recordings were requested at 30 fps; their timestamps are about 33.6 ms apart. They did not save RGB exposure or gain, so the actual shutter time is unknown. The saved images show a blurred, desaturated ball during flight; the current detector requires a round, sufficiently saturated blob. In `183141`, frames 90–95 show the ball in flight but the saved tracker reported no position. The RGB exposure and stereo-depth exposure are separate controls; this test adjusts RGB only.

From the project root, in the camera environment:

```sh
# Check the higher rate with the camera's existing automatic exposure.
python3 -B scripts/measure_ball.py --ball orange --fps 60
```

Keep the ball out of view while the script learns the background, then use the `v` key or RECORD button to record a few throws. Check the console's processed/camera rate and skipped-frame count. If 60 fps cannot be configured on the camera/USB connection, the script reports an error; use `--fps 30` instead.

To try shorter *manual RGB* exposure, for example:

```sh
python3 -B scripts/measure_ball.py --ball orange --fps 60 --rgb-exposure 30
```

`30` is a raw RealSense SDK RGB exposure value, **not 30 microseconds**; units and supported steps depend on the sensor/backend. The program prints the sensor's read-back value or its supported range if rejected. When available, each recorded `track.csv` row has `rgb_exposure_us` and `rgb_gain` from the frame's actual metadata. Check those fields rather than assuming a conversion. `--rgb-gain VALUE` optionally sets manual gain. The previous RGB settings are restored on normal exit or exceptions; hardware disconnects can prevent restoration. The flags do not change depth sensor exposure. Without these flags the scripts leave camera exposure and gain unchanged.

Short exposures can make the picture too dark; improve lighting and inspect the image. Higher fps gives more samples, while shorter exposure reduces blur; neither by itself guarantees reliable depth. Compare a small number of throws with the same placement and speed. Identify release and arrival frames in the videos, then count `LOCKED` frames with a nonempty `cam_z_m` **during that flight**. The overall tracked-frame count includes the ball in the hand and is not a flight metric. Also inspect false detections and `depth` versus `size` in `track.csv`.

The ROS `ball_vision_node.py` accepts the same `--fps`, `--rgb-exposure` and `--rgb-gain` options and reports actual RGB frame metadata and processing rate where available. It publishes commands indirectly if connected to the model and a motion-enabled adapter; use the standalone `measure_ball.py` to test camera settings without publishing robot targets. This has not been tested on the physical camera after the code change.
