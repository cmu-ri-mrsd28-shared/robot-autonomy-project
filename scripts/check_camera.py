"""Check a RealSense RGB-D stream and optionally preview its frames."""

import argparse
import math
import os
from statistics import median
import sys
import time


def show_frames(cv2, np, colorizer, color, depth):
    color_image = np.asanyarray(color.get_data())
    depth_rgb = np.asanyarray(colorizer.colorize(depth).get_data())
    depth_image = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
    preview = np.hstack((color_image, depth_image))
    window = "RealSense D435 | RGB (left) / Depth (right)"
    cv2.imshow(window, preview)
    return cv2.waitKey(1) & 0xFF not in (ord("q"), 27) and (
        cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds", type=float, help="Stop after this many seconds of capture"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show live RGB and aligned depth"
    )
    args = parser.parse_args()
    if args.seconds is not None and (
        not math.isfinite(args.seconds) or args.seconds <= 0
    ):
        parser.error("--seconds must be positive and finite")

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        print(
            f"Could not import the RealSense SDK ({exc}). "
            "Install requirements-camera.txt in this Python environment.",
            file=sys.stderr,
        )
        return 1

    if args.preview:
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            parser.error("--preview requires a graphical desktop session")
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            print(
                f"Could not import preview dependencies ({exc}). "
                "Install requirements-camera.txt in this Python environment.",
                file=sys.stderr,
            )
            return 1
        colorizer = rs.colorizer()

    try:
        context = rs.context()
        devices = context.query_devices()
        if len(devices) == 0:
            raise RuntimeError(
                "No RealSense device found. Check the USB connection and device permissions."
            )
        device = devices[0]
        info = rs.camera_info
        serial = device.get_info(info.serial_number)
        usb = device.get_info(info.usb_type_descriptor) if device.supports(
            info.usb_type_descriptor
        ) else "unknown"
        print(
            f"Connected  {device.get_info(info.name)} | serial={serial} | "
            f"firmware={device.get_info(info.firmware_version)} | USB={usb}",
            flush=True,
        )

        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline = rs.pipeline(context)
        align = rs.align(rs.stream.color)
        profile = pipeline.start(config)
        try:
            for stream in (rs.stream.color, rs.stream.depth):
                video = profile.get_stream(stream).as_video_stream_profile()
                print(
                    f"Stream     {video.stream_name()} "
                    f"{video.width()}x{video.height()} @ {video.fps()} fps",
                    flush=True,
                )

            started = last_report = time.monotonic()
            frames_seen = 0
            last_pair = started
            timeout_ms = 6000
            while True:
                frames = pipeline.wait_for_frames(timeout_ms)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                now = time.monotonic()
                if color and depth:
                    frames_seen += 1
                    last_pair = now
                    timeout_ms = 2000
                    report_due = now - last_report >= 1 or (
                        args.seconds is not None and now - started >= args.seconds
                    )
                    if report_due or args.preview:
                        aligned_frames = align.process(frames)
                        aligned_depth = aligned_frames.get_depth_frame()
                        if not aligned_depth:
                            raise RuntimeError("Could not align depth to color.")
                    if args.preview and not show_frames(
                        cv2, np, colorizer, color, aligned_depth
                    ):
                        break
                    if report_due:
                        x = aligned_depth.get_width() // 2
                        y = aligned_depth.get_height() // 2
                        distances = [
                            aligned_depth.get_distance(u, v)
                            for v in range(y - 2, y + 3)
                            for u in range(x - 2, x + 3)
                        ]
                        valid = [d for d in distances if math.isfinite(d) and d > 0]
                        depth_text = f"{median(valid):.3f} m" if valid else "invalid"
                        color_clock = color.get_frame_timestamp_domain()
                        depth_clock = depth.get_frame_timestamp_domain()
                        skew = "unavailable"
                        if color_clock == depth_clock == rs.timestamp_domain.global_time:
                            skew = f"{abs(color.get_timestamp() - depth.get_timestamp()):.1f} ms"
                        print(
                            f"Capture    {frames_seen / (now - last_report):.1f} pairs/s | "
                            f"center depth={depth_text} | valid pixels={len(valid)}/25 | "
                            f"RGB-depth skew={skew}",
                            flush=True,
                        )
                        frames_seen = 0
                        last_report = now
                if color and depth and args.seconds is not None and now - started >= args.seconds:
                    break
                if now - last_pair >= timeout_ms / 1000:
                    raise RuntimeError("No paired RGB and depth frames arrived.")
        except BaseException:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            raise
        else:
            pipeline.stop()
    except KeyboardInterrupt:
        print("\nCamera check stopped.")
    except RuntimeError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
