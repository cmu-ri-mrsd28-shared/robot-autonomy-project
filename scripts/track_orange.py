"""Track an ORANGE table-tennis ball. Same as track_ball.py, but detects orange, not white.

All track_ball.py options work here (--preview, --fps, --calibration, ...).
Extra options tune the orange color range (OpenCV HSV: hue 0-179, sat/value 0-255):
  --hue-min / --hue-max   default 5-25 (orange)
  --sat-min               default 120 (how colorful; lower accepts paler orange)
  --value-min             default 110 (how bright; lower accepts shadowed ball)
  --show-mask             show which pixels count as orange (for tuning)

Example:
  python -B scripts/track_orange.py --preview --fps 30
"""

import math
import sys

import track_ball
from track_ball import Detection


COLOR = {"hue_min": 5, "hue_max": 25, "sat_min": 120, "value_min": 110, "show_mask": False}


def detect_orange(cv2, np, image, gray, previous_gray, brightness):
    """Find the strongest moving, orange, roughly circular blob.

    Mirrors track_ball.detect_ball, but selects orange pixels instead of white.
    The `brightness` argument is unused (kept so track_ball.track can call it).
    """
    if previous_gray is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (COLOR["hue_min"], COLOR["sat_min"], COLOR["value_min"]),
                       (COLOR["hue_max"], 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if COLOR["show_mask"]:
        cv2.imshow("orange mask (white = counted as orange)", mask)
    # Pixels that changed help pick the moving ball over still orange objects.
    changed = cv2.absdiff(gray, previous_gray) >= 18
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score, best = 0.0, None
    for contour in contours:
        (u, v), radius = cv2.minEnclosingCircle(contour)
        if not 3 <= radius <= 42:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * cv2.contourArea(contour) / perimeter**2
        if circularity < 0.55:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        inside = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(inside, [contour], -1, 1, cv2.FILLED, offset=(-x, -y))
        motion = changed[y:y + h, x:x + w][inside != 0].mean()
        score = circularity * motion
        if motion >= 0.12 and score > best_score:
            best_score, best = score, Detection(round(u), round(v), radius)
    return best


def main():
    # Pull out the orange-only options, pass everything else to track_ball.
    args, rest = sys.argv[1:], [sys.argv[0]]
    names = {"--hue-min": "hue_min", "--hue-max": "hue_max",
             "--sat-min": "sat_min", "--value-min": "value_min"}
    i = 0
    while i < len(args):
        if args[i] in names and i + 1 < len(args):
            COLOR[names[args[i]]] = int(args[i + 1])
            i += 2
        elif args[i] == "--show-mask":
            COLOR["show_mask"] = True
            i += 1
        else:
            if args[i] in ("-h", "--help"):
                print(__doc__)
            rest.append(args[i])
            i += 1
    if not 0 <= COLOR["hue_min"] <= COLOR["hue_max"] <= 179:
        sys.exit("--hue-min/--hue-max must satisfy 0 <= min <= max <= 179")
    sys.argv = rest
    track_ball.detect_ball = detect_orange  # track() looks this name up at call time
    track_ball.WINDOW = "D435 orange-ball tracker"
    return track_ball.main()


if __name__ == "__main__":
    raise SystemExit(main())
