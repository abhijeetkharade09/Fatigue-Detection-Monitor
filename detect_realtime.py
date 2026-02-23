import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ================================================================
# PROJECT : VR-Assisted Remote Health Monitor
# FILE    : detect_realtime.py
#
# ─── COGNITIVE MODE  ────────────────────────────────────────────
#   Untouched logic. Works via EAR + blink rate + head drop.
#
# ─── EXERCISE MODE (SQUATS + SURYANAMASKAR) ─────────────────────
#   Pure fatigue detection. NO MQS. NO scoring.
#   Fatigue declared ONLY when 2+ conditions simultaneously true.
#   Fatigue Index 0–100% shown on screen.
#
# ─── UI LAYER ────────────────────────────────────────────────────
#   Medical / clinical style. Fullscreen window.
#   Title header bar, clean side panels, bottom status strip.
#   ALL LOGIC IDENTICAL — only drawing functions changed.
# ================================================================

# ──────────────────────────────────────────────────────────────
# MEDIAPIPE INIT
# ──────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Custom skeleton drawing spec — thin white lines for clinical look
POSE_DRAW_SPEC = mp_drawing.DrawingSpec(
    color=(200, 200, 200), thickness=2, circle_radius=3)
POSE_CONN_SPEC = mp_drawing.DrawingSpec(
    color=(160, 160, 160), thickness=2)

# ──────────────────────────────────────────────────────────────
# CAMERA  — request HD for better fullscreen quality
# ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    # Fall back to standard resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Camera not opened")
    exit()

# ── Fullscreen window setup ──────────────────────────────────
WIN_NAME = "VR Health Monitor — Remote Fatigue Detection"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

print("=" * 55)
print("  VR Health Monitor — Fatigue Detection v3")
print("=" * 55)
print("  Q  ->  Quit")
print("  M  ->  Toggle COGNITIVE / EXERCISE mode")
print("  S  ->  Squats          (Exercise mode only)")
print("  N  ->  Surya Namaskar  (Exercise mode only)")
print("=" * 55)

# ================================================================
# ██████████  COGNITIVE MODE CONSTANTS  ██████████
# !! DO NOT MODIFY ANYTHING IN THIS SECTION !!
# ================================================================
EAR_THRESHOLD      = 0.23
BLINK_MIN_DURATION = 0.1
BLINK_MAX_DURATION = 0.4
NOSE_INDEX         = 1
CHIN_INDEX         = 152
LEFT_EYE           = [33, 160, 158, 133, 153, 144]
RIGHT_EYE          = [362, 385, 387, 263, 373, 380]
HEAD_DROP_COG      = 70

# ================================================================
# ██████████  COGNITIVE MODE STATE  ██████████
# !! DO NOT MODIFY ANYTHING IN THIS SECTION !!
# ================================================================
closed_start_time   = None
blink_start         = None
blink_timestamps    = []
fatigue_index_cog   = 0.0
blink_rate_smoothed = 15.0

# ================================================================
# EXERCISE FATIGUE ENGINE — CONSTANTS  (logic — do not modify)
# ================================================================
CALIB_REPS_NEEDED        = 3
SQUAT_KNEE_COLLAPSE_DEG  = 25
SQUAT_KNEE_COLLAPSE_SECS = 0.2
SQUAT_HIP_DROP_PX        = 30
SQUAT_VEL_DECAY_RATIO    = 0.40
SQUAT_EXTENSION_RATIO    = 0.90
SQUAT_EXTENSION_REPS     = 2
SQUAT_BALANCE_MULTIPLIER = 2.5
SURYA_TRANSITION_SLOW_RATIO = 1.50
SURYA_HIP_SAG_PX         = 40
SURYA_KNEE_DROP_DEG      = 30
SURYA_PAUSE_VELOCITY     = 3.5
SURYA_PAUSE_SECS         = 4.0
SURYA_SPEED_DEGRADE_RATIO = 0.35
MIN_CONDITIONS_FOR_FATIGUE = 2
FI_INCREMENT_PER_CONDITION = 8.0
FI_DECAY_PER_FRAME         = 0.15
FI_SMOOTHING_ALPHA         = 0.08
VEL_WINDOW                 = 25
HIP_X_HISTORY_WINDOW       = 30

# ================================================================
# EXERCISE MODE STATE  (logic — do not modify)
# ================================================================
mode          = "COGNITIVE"
exercise_type = "SQUATS"

calib_done              = False
calib_roms              = []
calib_up_velocities     = []
calib_hip_x_std         = None
calib_standing_hip_y    = None
calib_transition_times  = []
calib_avg_velocity      = None
baseline_rom            = None
baseline_up_velocity    = None

prev_hip_y              = None
prev_knee_angle         = None
prev_knee_angle_time    = None
velocity_history        = deque(maxlen=VEL_WINDOW)
hip_x_history           = deque(maxlen=HIP_X_HISTORY_WINDOW)
upward_vel_history      = deque(maxlen=10)

knee_angle_buffer       = deque(maxlen=10)
knee_time_buffer        = deque(maxlen=10)

rep_stage               = "UP"
rep_min_angle           = 180.0
rep_count               = 0
consecutive_incomplete  = 0

surya_last_transition_time = time.time()
surya_pause_start       = None
surya_transition_count  = 0

fatigue_index_ex        = 0.0
fatigue_index_display   = 0.0

active_conditions       = []
alert_text              = ""
alert_until             = 0.0
prev_lm_snap            = None


# ================================================================
# ████████████████  CLINICAL UI PALETTE  ████████████████
# All colours defined once here — easy to adjust.
# OpenCV uses BGR not RGB.
# ================================================================
class C:
    # Backgrounds
    BG_DARK        = (30,  30,  30)    # main canvas behind camera
    BG_PANEL       = (245, 245, 245)   # white panel background
    BG_HEADER      = (255, 255, 255)   # pure white header
    BG_STATUS_OK   = (235, 252, 235)   # pale green
    BG_STATUS_WARN = (255, 243, 224)   # pale orange
    BG_STATUS_CRIT = (255, 230, 230)   # pale red

    # Text
    TEXT_DARK      = (30,  30,  30)    # near-black for white backgrounds
    TEXT_MED       = (80,  80,  80)    # secondary labels
    TEXT_LIGHT     = (200, 200, 200)   # on dark backgrounds
    TEXT_ACCENT    = (0,   120, 200)   # clinical blue (project name etc.)

    # Clinical status colours
    OK             = (34,  139, 34)    # forest green
    WARN           = (0,   140, 255)   # orange
    CRIT           = (30,  30,  200)   # deep red

    # Bar fills
    BAR_OK         = (60,  179, 60)
    BAR_WARN       = (0,   160, 255)
    BAR_CRIT       = (40,  40,  210)
    BAR_BG         = (210, 210, 210)   # unfilled bar track
    BAR_BORDER     = (160, 160, 160)

    # Panel border
    PANEL_BORDER   = (180, 180, 180)
    DIVIDER        = (200, 200, 200)

    # Header accent line
    HEADER_LINE    = (0,   120, 200)   # clinical blue underline


# ================================================================
# UI HELPER FUNCTIONS  — pure drawing, no logic
# ================================================================

FONT      = cv2.FONT_HERSHEY_DUPLEX   # cleaner than SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX


def txt(frame, text, x, y,
        color=C.TEXT_DARK, scale=0.52, thick=1, font=FONT):
    """Unified text draw — all UI text goes through here."""
    cv2.putText(frame, text, (x, y), font, scale, color, thick,
                cv2.LINE_AA)


def panel(frame, x1, y1, x2, y2,
          bg=C.BG_PANEL, border=C.PANEL_BORDER, radius=6):
    """
    Draws a filled rounded-corner panel.
    For simplicity uses a regular rect (cv2 has no built-in rounded rect).
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)


def divider(frame, x1, x2, y, color=C.DIVIDER):
    cv2.line(frame, (x1, y), (x2, y), color, 1, cv2.LINE_AA)


def clinical_bar(frame, value, max_val,
                 x, y, bar_w, bar_h=14,
                 label="", unit="",
                 low_good=True):
    """
    Clinical-style progress bar.
    low_good=True  → green when low  (fatigue: low is good)
    low_good=False → green when high (score: high is good)
    Returns the bar's bottom-right y for stacking.
    """
    ratio  = max(0.0, min(1.0, value / max_val))
    filled = int(ratio * bar_w)

    if low_good:
        col = (C.BAR_OK if ratio < 0.35
               else C.BAR_WARN if ratio < 0.70
               else C.BAR_CRIT)
    else:
        col = (C.BAR_CRIT if ratio < 0.35
               else C.BAR_WARN if ratio < 0.70
               else C.BAR_OK)

    # Label above bar
    if label:
        txt(frame, label, x, y - 3, C.TEXT_MED, scale=0.42)

    # Track
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h),
                  C.BAR_BG, -1)
    # Fill
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), col, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h),
                  C.BAR_BORDER, 1)
    # Value text right-aligned after bar
    val_str = f"{value:.1f}{unit}"
    txt(frame, val_str, x + bar_w + 6, y + bar_h - 1,
        C.TEXT_MED, scale=0.42)

    return y + bar_h


def status_chip(frame, text, x, y, w_chip, level="ok"):
    """
    Small filled status badge — like a pill/chip.
    level: 'ok' | 'warn' | 'crit'
    """
    h_chip = 22
    bg  = {"ok": C.BG_STATUS_OK,
           "warn": C.BG_STATUS_WARN,
           "crit": C.BG_STATUS_CRIT}[level]
    col = {"ok": C.OK, "warn": C.WARN, "crit": C.CRIT}[level]
    cv2.rectangle(frame, (x, y), (x + w_chip, y + h_chip), bg, -1)
    cv2.rectangle(frame, (x, y), (x + w_chip, y + h_chip), col, 1)
    # Centre text
    ts = cv2.getTextSize(text, FONT, 0.44, 1)[0]
    tx = x + (w_chip - ts[0]) // 2
    ty = y + (h_chip + ts[1]) // 2 - 1
    txt(frame, text, tx, ty, col, scale=0.44, thick=1)


def draw_header(canvas, W, mode_label, exercise_label=""):
    """
    Draws the top clinical header bar across the full canvas width.
    Contains: project name left, mode badge centre, time right.
    """
    H_HDR = 52
    # White header background
    cv2.rectangle(canvas, (0, 0), (W, H_HDR), C.BG_HEADER, -1)
    # Blue accent underline
    cv2.line(canvas, (0, H_HDR), (W, H_HDR), C.HEADER_LINE, 2)

    # Left — project name
    txt(canvas, "VR Health Monitor",
        14, 22, C.TEXT_ACCENT, scale=0.65, thick=2)
    txt(canvas, "Remote Fatigue Detection System",
        14, 42, C.TEXT_MED, scale=0.40)

    # Centre — mode badge
    badge = mode_label
    if exercise_label:
        badge += f"  |  {exercise_label}"
    ts = cv2.getTextSize(badge, FONT, 0.58, 2)[0]
    bx = (W - ts[0]) // 2
    txt(canvas, badge, bx, 32, C.TEXT_ACCENT, scale=0.58, thick=2)

    # Right — live clock
    clock_str = time.strftime("%H:%M:%S")
    ts2 = cv2.getTextSize(clock_str, FONT, 0.50, 1)[0]
    txt(canvas, clock_str, W - ts2[0] - 14, 22,
        C.TEXT_MED, scale=0.50)
    txt(canvas, "LIVE", W - 44, 42, C.CRIT, scale=0.38, thick=1)

    return H_HDR   # height consumed


def draw_bottom_bar(canvas, W, H, status_text, status_level, keys_hint):
    """
    Draws the bottom status strip across full canvas width.
    """
    BAR_H = 48
    y0 = H - BAR_H
    bg = {"ok": C.BG_STATUS_OK,
          "warn": C.BG_STATUS_WARN,
          "crit": C.BG_STATUS_CRIT}.get(status_level, C.BG_PANEL)
    col = {"ok": C.OK, "warn": C.WARN,
           "crit": C.CRIT}.get(status_level, C.TEXT_DARK)

    cv2.rectangle(canvas, (0, y0), (W, H), bg, -1)
    cv2.line(canvas, (0, y0), (W, y0), C.PANEL_BORDER, 1)

    # Status text centred vertically, left-padded
    txt(canvas, status_text, 20, y0 + 30, col, scale=0.75, thick=2)

    # Keys hint right side
    txt(canvas, keys_hint, W - 340, y0 + 30,
        C.TEXT_MED, scale=0.38, font=FONT_MONO)


def draw_cognitive_panel(canvas, px, py, pw,
                         ear, blinks_per_min,
                         fatigue_idx, head_status, head_level):
    """
    Draws the cognitive-mode side panel.
    px, py = top-left of panel. pw = panel width.
    All logic values passed in — no computation here.
    """
    ROW_H  = 26
    PAD    = 12
    BARY   = py + 158
    ph     = 230   # panel height

    panel(canvas, px, py, px + pw, py + ph)

    # ── Section title ──
    txt(canvas, "COGNITIVE MODE", px + PAD, py + 20,
        C.TEXT_ACCENT, scale=0.52, thick=2)
    divider(canvas, px + PAD, px + pw - PAD, py + 28)

    # ── EAR row ──
    txt(canvas, "Eye Aspect Ratio (EAR)", px + PAD, py + 50,
        C.TEXT_MED, scale=0.42)
    ear_col = C.CRIT if ear < EAR_THRESHOLD else C.OK
    txt(canvas, f"{ear:.3f}", px + pw - 60, py + 50,
        ear_col, scale=0.50, thick=2)

    # ── Blink rate ──
    txt(canvas, "Blink Rate", px + PAD, py + 76, C.TEXT_MED, scale=0.42)
    br_col = C.CRIT if blinks_per_min < 8 else C.OK
    txt(canvas, f"{int(blinks_per_min)} / min",
        px + pw - 80, py + 76, br_col, scale=0.50, thick=2)

    divider(canvas, px + PAD, px + pw - PAD, py + 88)

    # ── Fatigue Index label ──
    txt(canvas, "Fatigue Index", px + PAD, py + 108,
        C.TEXT_MED, scale=0.42)
    fi_pct = (fatigue_idx / 5.0) * 100.0
    fi_col = (C.OK if fi_pct < 35 else
              C.WARN if fi_pct < 70 else C.CRIT)
    txt(canvas, f"{fi_pct:.0f}%", px + pw - 56, py + 108,
        fi_col, scale=0.50, thick=2)

    # ── Fatigue bar ──
    clinical_bar(canvas, fatigue_idx, 5.0,
                 px + PAD, py + 118,
                 bar_w=pw - PAD*2 - 30,
                 bar_h=14, low_good=True)

    divider(canvas, px + PAD, px + pw - PAD, py + 144)

    # ── Head status chip ──
    txt(canvas, "Head Position", px + PAD, py + 166,
        C.TEXT_MED, scale=0.42)
    status_chip(canvas, head_status,
                px + PAD, py + 174,
                pw - PAD * 2, level=head_level)

    # ── Eye tracking note ──
    txt(canvas, "Eye tracking active", px + PAD, py + 218,
        C.TEXT_MED, scale=0.38)


def draw_exercise_panel(canvas, px, py, pw,
                        fi_pct, status_msgs, conditions_fired,
                        ex_status, ex_level, exercise_type,
                        calib_done, calib_reps_done):
    """
    Draws the exercise-mode side panel.
    px, py = top-left of panel. pw = panel width.
    All logic values passed in — no computation here.
    """
    PAD   = 12
    ROW_H = 24
    ph    = 52 + 42 + (len(status_msgs) + 1) * ROW_H + 50
    ph    = max(ph, 280)

    panel(canvas, px, py, px + pw, py + ph)

    # ── Section title ──
    ex_label = "SQUATS MODE" if exercise_type == "SQUATS" else "SURYA NAMASKAR MODE"
    txt(canvas, ex_label, px + PAD, py + 20,
        C.TEXT_ACCENT, scale=0.52, thick=2)
    divider(canvas, px + PAD, px + pw - PAD, py + 28)

    if not calib_done:
        # ── Calibration in progress ──
        txt(canvas, "Calibrating...", px + PAD, py + 58,
            C.WARN, scale=0.52, thick=1)
        txt(canvas, f"Reps collected: {calib_reps_done} / {CALIB_REPS_NEEDED}",
            px + PAD, py + 82, C.TEXT_MED, scale=0.44)
        txt(canvas, "Perform your normal exercise",
            px + PAD, py + 106, C.TEXT_MED, scale=0.40)
        txt(canvas, "to build personal baseline.",
            px + PAD, py + 124, C.TEXT_MED, scale=0.40)
        return py + ph

    # ── Fatigue Index ──
    txt(canvas, "Fatigue Index", px + PAD, py + 50,
        C.TEXT_MED, scale=0.42)
    fi_col = (C.OK if fi_pct < 31 else
              C.WARN if fi_pct < 61 else C.CRIT)
    txt(canvas, f"{fi_pct:.0f}%", px + pw - 56, py + 50,
        fi_col, scale=0.55, thick=2)

    clinical_bar(canvas, fi_pct, 100.0,
                 px + PAD, py + 60,
                 bar_w=pw - PAD*2 - 30,
                 bar_h=14, low_good=True)

    # ── Fatigue level label ──
    fi_label = ("Normal"          if fi_pct <= 30 else
                "Mild Fatigue"    if fi_pct <= 60 else
                "Moderate Fatigue" if fi_pct <= 80 else
                "Severe Fatigue")
    fl_level = ("ok" if fi_pct <= 30 else
                "warn" if fi_pct <= 60 else "crit")
    status_chip(canvas, fi_label,
                px + PAD, py + 82, pw - PAD*2, level=fl_level)

    divider(canvas, px + PAD, px + pw - PAD, py + 114)

    # ── Condition status rows ──
    txt(canvas, "Active Signals", px + PAD, py + 132,
        C.TEXT_MED, scale=0.42)

    row_y = py + 148
    if status_msgs:
        for msg in status_msgs:
            is_alert = any(kw in msg for kw in [
                "Collapse", "Drop", "Reduced", "Unstable",
                "Incomplete", "Slowing", "Sag", "Pause", "Slow",
                "Detected"])
            chip_lv = "crit" if is_alert else "ok"
            status_chip(canvas, msg,
                        px + PAD, row_y, pw - PAD*2, level=chip_lv)
            row_y += ROW_H + 2
    else:
        status_chip(canvas, "Monitoring...",
                    px + PAD, row_y, pw - PAD*2, level="ok")
        row_y += ROW_H + 2

    divider(canvas, px + PAD, px + pw - PAD, row_y + 6)

    # ── Overall status chip ──
    status_chip(canvas, ex_status,
                px + PAD, row_y + 14,
                pw - PAD*2, level=ex_level)

    return py + ph


def draw_alert_banner(canvas, W, text, alpha=0.72):
    """
    Full-width semi-transparent red alert banner just below header.
    """
    H_HDR  = 52
    BAN_H  = 46
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, H_HDR), (W, H_HDR + BAN_H),
                  (40, 40, 200), -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    ts  = cv2.getTextSize(text, FONT, 0.85, 2)[0]
    tx  = (W - ts[0]) // 2
    ty  = H_HDR + BAN_H // 2 + ts[1] // 2
    txt(canvas, text, tx, ty, (255, 255, 255), scale=0.85, thick=2)


# ================================================================
# ████  LOGIC HELPERS  ████  (unchanged from v3)
# ================================================================

def eye_aspect_ratio(pts):
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C_ = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C_)


def calc_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    rad = (np.arctan2(c[1]-b[1], c[0]-b[0]) -
           np.arctan2(a[1]-b[1], a[0]-b[0]))
    ang = abs(rad * 180.0 / np.pi)
    return 360.0 - ang if ang > 180.0 else ang


def lm_px(lm, i, w, h):
    return [lm[i].x * w, lm[i].y * h]


def avg_knee_angle(lm, w, h):
    LH = lm_px(lm,23,w,h); LK = lm_px(lm,25,w,h); LA = lm_px(lm,27,w,h)
    RH = lm_px(lm,24,w,h); RK = lm_px(lm,26,w,h); RA = lm_px(lm,28,w,h)
    return (calc_angle(LH,LK,LA) + calc_angle(RH,RK,RA)) / 2.0


def hip_mid_y(lm, w, h):
    return (lm[23].y*h + lm[24].y*h) / 2.0


def hip_mid_x(lm, w, h):
    return (lm[23].x*w + lm[24].x*w) / 2.0


def body_avg_velocity(lm, w, h, prev_snap):
    if prev_snap is None:
        return 0.0
    idxs = [11, 12, 23, 24, 25, 26]
    dists = [np.hypot(lm[i].x*w - prev_snap[i][0],
                      lm[i].y*h - prev_snap[i][1]) for i in idxs]
    return float(np.mean(dists))


def snapshot(lm, w, h):
    idxs = [11, 12, 23, 24, 25, 26]
    return {i: (lm[i].x*w, lm[i].y*h) for i in idxs}


def visibility_ok(lm):
    return all(lm[i].visibility > 0.45
               for i in [11,12,23,24,25,26,27,28])


# ================================================================
# EXERCISE RESET  (logic — unchanged)
# ================================================================
def reset_exercise():
    global calib_done, calib_roms, calib_up_velocities
    global calib_hip_x_std, calib_standing_hip_y
    global calib_transition_times, calib_avg_velocity
    global baseline_rom, baseline_up_velocity
    global prev_hip_y, prev_knee_angle, prev_knee_angle_time
    global velocity_history, hip_x_history, upward_vel_history
    global knee_angle_buffer, knee_time_buffer
    global rep_stage, rep_min_angle, rep_count, consecutive_incomplete
    global surya_last_transition_time, surya_pause_start, surya_transition_count
    global fatigue_index_ex, fatigue_index_display
    global active_conditions, alert_text, alert_until, prev_lm_snap

    calib_done              = False
    calib_roms              = []
    calib_up_velocities     = []
    calib_hip_x_std         = None
    calib_standing_hip_y    = None
    calib_transition_times  = []
    calib_avg_velocity      = None
    baseline_rom            = None
    baseline_up_velocity    = None
    prev_hip_y              = None
    prev_knee_angle         = None
    prev_knee_angle_time    = None
    velocity_history        = deque(maxlen=VEL_WINDOW)
    hip_x_history           = deque(maxlen=HIP_X_HISTORY_WINDOW)
    upward_vel_history      = deque(maxlen=10)
    knee_angle_buffer       = deque(maxlen=10)
    knee_time_buffer        = deque(maxlen=10)
    rep_stage               = "UP"
    rep_min_angle           = 180.0
    rep_count               = 0
    consecutive_incomplete  = 0
    surya_last_transition_time = time.time()
    surya_pause_start       = None
    surya_transition_count  = 0
    fatigue_index_ex        = 0.0
    fatigue_index_display   = 0.0
    active_conditions       = []
    alert_text              = ""
    alert_until             = 0.0
    prev_lm_snap            = None


# ================================================================
# SQUATS FATIGUE ENGINE  (logic — unchanged)
# ================================================================
def run_squat_fatigue(lm, w, h, current_time):
    global prev_hip_y, prev_knee_angle, prev_knee_angle_time
    global velocity_history, hip_x_history, upward_vel_history
    global knee_angle_buffer, knee_time_buffer
    global rep_stage, rep_min_angle, rep_count, consecutive_incomplete
    global calib_standing_hip_y

    conditions_fired = []
    status_msgs      = []

    knee_ang = avg_knee_angle(lm, w, h)
    hip_y_   = hip_mid_y(lm, w, h)
    hip_x_   = hip_mid_x(lm, w, h)

    knee_angle_buffer.append(knee_ang)
    knee_time_buffer.append(current_time)
    hip_x_history.append(hip_x_)

    if prev_hip_y is not None:
        hip_delta = hip_y_ - prev_hip_y
        hip_vel   = abs(hip_delta)
        velocity_history.append(hip_vel)
        if hip_delta < 0 and rep_stage == "DOWN":
            upward_vel_history.append(abs(hip_delta))
    else:
        hip_delta = 0.0

    if knee_ang < 140:
        rep_stage     = "DOWN"
        rep_min_angle = min(rep_min_angle, knee_ang)
    elif knee_ang >= 155 and rep_stage == "DOWN":
        rep_count    += 1
        rep_stage     = "UP"
        rep_min_angle = 180.0

    if knee_ang > 155 and calib_standing_hip_y is None:
        calib_standing_hip_y = hip_y_

    # Condition 1 — Sudden Knee Collapse
    c1_fired = False
    if len(knee_angle_buffer) >= 2 and len(knee_time_buffer) >= 2:
        for i in range(len(knee_time_buffer)):
            if (current_time - knee_time_buffer[i]) <= SQUAT_KNEE_COLLAPSE_SECS:
                if (knee_angle_buffer[i] - knee_ang) > SQUAT_KNEE_COLLAPSE_DEG:
                    c1_fired = True
                break
    if c1_fired:
        conditions_fired.append("knee_collapse")
        status_msgs.append("Knee Collapse Detected")
    else:
        status_msgs.append("Movement Normal")

    # Condition 2 — Sudden Hip Drop
    c2_fired = False
    if prev_hip_y is not None and hip_delta > SQUAT_HIP_DROP_PX and rep_stage == "DOWN":
        c2_fired = True
    if c2_fired:
        conditions_fired.append("hip_drop")
        status_msgs.append("Hip Drop Detected")

    # Condition 3 — Upward Velocity Reduction
    c3_fired = False
    if (calib_done and baseline_up_velocity is not None
            and baseline_up_velocity > 0.5 and rep_stage == "DOWN"
            and len(upward_vel_history) >= 3):
        rv = float(np.mean(list(upward_vel_history)[-3:]))
        if rv < baseline_up_velocity * SQUAT_VEL_DECAY_RATIO:
            c3_fired = True
    if c3_fired:
        conditions_fired.append("upward_vel_reduced")
        status_msgs.append("Upward Speed Reduced")
    elif calib_done:
        status_msgs.append("Speed Normal")

    # Condition 4 — Incomplete Extension
    c4_fired = False
    if (calib_done and calib_standing_hip_y is not None
            and rep_stage == "UP" and knee_ang > 150):
        thr = calib_standing_hip_y + (calib_standing_hip_y * (1.0 - SQUAT_EXTENSION_RATIO))
        if hip_y_ > thr:
            consecutive_incomplete += 1
        else:
            consecutive_incomplete = 0
        if consecutive_incomplete >= SQUAT_EXTENSION_REPS * 15:
            c4_fired = True
    if c4_fired:
        conditions_fired.append("incomplete_extension")
        status_msgs.append("Incomplete Extension")
    elif calib_done:
        status_msgs.append("Extension Normal")

    # Condition 5 — Balance Instability
    c5_fired = False
    if (calib_done and calib_hip_x_std is not None
            and calib_hip_x_std > 0.5
            and len(hip_x_history) >= HIP_X_HISTORY_WINDOW):
        if float(np.std(hip_x_history)) > calib_hip_x_std * SQUAT_BALANCE_MULTIPLIER:
            c5_fired = True
    if c5_fired:
        conditions_fired.append("balance_unstable")
        status_msgs.append("Balance Unstable")
    elif calib_done:
        status_msgs.append("Balance Normal")

    prev_hip_y           = hip_y_
    prev_knee_angle      = knee_ang
    prev_knee_angle_time = current_time
    return conditions_fired, status_msgs


# ================================================================
# SURYA FATIGUE ENGINE  (logic — unchanged)
# ================================================================
def run_surya_fatigue(lm, w, h, current_time):
    global prev_hip_y, prev_knee_angle
    global velocity_history, knee_angle_buffer, knee_time_buffer
    global surya_last_transition_time, surya_pause_start
    global surya_transition_count, prev_lm_snap

    conditions_fired = []
    status_msgs      = []

    knee_ang  = avg_knee_angle(lm, w, h)
    hip_y_    = hip_mid_y(lm, w, h)

    knee_angle_buffer.append(knee_ang)
    knee_time_buffer.append(current_time)

    vel     = body_avg_velocity(lm, w, h, prev_lm_snap)
    velocity_history.append(vel)
    avg_vel = float(np.mean(velocity_history)) if velocity_history else 0.0

    # Condition 1 — Transition Speed Slowing
    c1_fired = False
    if vel > 15.0:
        gap = current_time - surya_last_transition_time
        if (gap > 0.5 and calib_done and len(calib_transition_times) >= 2):
            bt = float(np.mean(calib_transition_times))
            if gap > bt * SURYA_TRANSITION_SLOW_RATIO:
                c1_fired = True
        surya_last_transition_time = current_time
        surya_transition_count    += 1
    if c1_fired:
        conditions_fired.append("transition_slow")
        status_msgs.append("Transition Slowing")
    else:
        status_msgs.append("Transition Normal")

    # Condition 2 — Plank Hip Sag
    c2_fired     = False
    shoulder_y   = (lm[11].y*h + lm[12].y*h) / 2.0
    ankle_y      = (lm[27].y*h + lm[28].y*h) / 2.0
    in_plank     = (abs(shoulder_y - ankle_y) < 60 and knee_ang > 140)
    if in_plank:
        exp_hip  = (shoulder_y + ankle_y) / 2.0
        if (hip_y_ - exp_hip) > SURYA_HIP_SAG_PX:
            c2_fired = True
    if c2_fired:
        conditions_fired.append("hip_sag")
        status_msgs.append("Hip Sag Detected")
    elif in_plank:
        status_msgs.append("Plank Normal")
    else:
        status_msgs.append("Pose Normal")

    # Condition 3 — Sudden Knee Drop in Plank
    c3_fired = False
    if in_plank and len(knee_angle_buffer) >= 3:
        if (knee_angle_buffer[0] - knee_ang) > SURYA_KNEE_DROP_DEG:
            c3_fired = True
    if c3_fired:
        conditions_fired.append("knee_drop")
        status_msgs.append("Knee Drop Detected")

    # Condition 4 — Abnormal Long Pause
    c4_fired = False
    if avg_vel < SURYA_PAUSE_VELOCITY:
        if surya_pause_start is None:
            surya_pause_start = current_time
        elif (current_time - surya_pause_start) > SURYA_PAUSE_SECS:
            c4_fired = True
    else:
        surya_pause_start = None
    if c4_fired:
        conditions_fired.append("long_pause")
        status_msgs.append("Abnormal Pause Detected")
    else:
        status_msgs.append("Flow Normal")

    # Condition 5 — Overall Cycle Speed Degradation
    c5_fired = False
    if (calib_done and calib_avg_velocity is not None
            and calib_avg_velocity > 1.0
            and len(velocity_history) >= VEL_WINDOW):
        if (avg_vel / calib_avg_velocity) < SURYA_SPEED_DEGRADE_RATIO:
            c5_fired = True
    if c5_fired:
        conditions_fired.append("cycle_speed_low")
        status_msgs.append("Cycle Speed Reduced")
    elif calib_done:
        status_msgs.append("Cycle Speed Normal")

    prev_hip_y      = hip_y_
    prev_knee_angle = knee_ang
    prev_lm_snap    = snapshot(lm, w, h)
    return conditions_fired, status_msgs


# ================================================================
# CALIBRATION  (logic — unchanged)
# ================================================================
def update_calibration(lm, w, h, current_time):
    global calib_done, calib_roms, calib_up_velocities
    global calib_hip_x_std, calib_standing_hip_y
    global calib_transition_times, calib_avg_velocity
    global baseline_rom, baseline_up_velocity
    global rep_stage, rep_min_angle
    global prev_hip_y, velocity_history, hip_x_history
    global upward_vel_history, surya_last_transition_time, prev_lm_snap

    knee_ang = avg_knee_angle(lm, w, h)
    hip_y_   = hip_mid_y(lm, w, h)
    hip_x_history.append(hip_mid_x(lm, w, h))

    if knee_ang > 155 and calib_standing_hip_y is None:
        calib_standing_hip_y = hip_y_

    if prev_hip_y is not None:
        vel_ = abs(hip_y_ - prev_hip_y)
        velocity_history.append(vel_)
        if hip_y_ < prev_hip_y and rep_stage == "DOWN":
            upward_vel_history.append(abs(hip_y_ - prev_hip_y))

    vel_b = body_avg_velocity(lm, w, h, prev_lm_snap)
    prev_lm_snap = snapshot(lm, w, h)

    if knee_ang < 140:
        rep_stage     = "DOWN"
        rep_min_angle = min(rep_min_angle, knee_ang)
    elif knee_ang >= 155 and rep_stage == "DOWN":
        calib_roms.append(rep_min_angle)
        if upward_vel_history:
            calib_up_velocities.append(float(np.mean(upward_vel_history)))
        gap = current_time - surya_last_transition_time
        if gap > 0.5:
            calib_transition_times.append(gap)
        surya_last_transition_time = current_time
        rep_min_angle = 180.0
        rep_stage     = "UP"
        upward_vel_history.clear()

    prev_hip_y = hip_y_

    if len(calib_roms) >= CALIB_REPS_NEEDED:
        baseline_rom         = float(np.mean(calib_roms))
        baseline_up_velocity = (float(np.mean(calib_up_velocities))
                                if calib_up_velocities else 5.0)
        calib_hip_x_std      = (float(np.std(hip_x_history))
                                if len(hip_x_history) > 5 else 5.0)
        calib_avg_velocity   = (float(np.mean(velocity_history))
                                if velocity_history else 5.0)
        print(f"[CALIB DONE] ROM:{baseline_rom:.1f} deg | "
              f"UpVel:{baseline_up_velocity:.2f} | "
              f"HipXstd:{calib_hip_x_std:.2f} | "
              f"AvgVel:{calib_avg_velocity:.2f}")
        calib_done = True
        return True
    return False


# ================================================================
# FATIGUE INDEX UPDATER  (logic — unchanged)
# ================================================================
def update_fatigue_index(n_conditions):
    global fatigue_index_ex, fatigue_index_display
    if n_conditions >= MIN_CONDITIONS_FOR_FATIGUE:
        fatigue_index_ex = min(100.0,
            fatigue_index_ex + FI_INCREMENT_PER_CONDITION * n_conditions)
    else:
        fatigue_index_ex = max(0.0, fatigue_index_ex - FI_DECAY_PER_FRAME)
    fatigue_index_display = (FI_SMOOTHING_ALPHA * fatigue_index_ex +
                             (1.0 - FI_SMOOTHING_ALPHA) * fatigue_index_display)
    return fatigue_index_display


# ================================================================
# ████████████████████  MAIN LOOP  ████████████████████
# ================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not captured. Retrying...")
        continue

    current_time = time.time()
    fh, fw, _   = frame.shape   # actual camera frame dimensions

    # ── Build fullscreen canvas ─────────────────────────────
    # Get actual display size from the window
    # We build a fixed 1280×720 canvas and scale camera into it
    CW, CH  = 1280, 720          # canvas width / height
    HDR_H   = 52                 # header bar height
    BOT_H   = 48                 # bottom bar height
    PANEL_W = 290                # right-side panel width
    PAD     = 10                 # gap between camera and panel

    # Area available for the camera feed
    CAM_X   = PAD
    CAM_Y   = HDR_H + PAD
    CAM_W   = CW - PANEL_W - PAD * 3
    CAM_H   = CH - HDR_H - BOT_H - PAD * 2

    # Create canvas — white background for clinical look
    canvas = np.full((CH, CW, 3), 240, dtype=np.uint8)

    # Scale camera frame to fit the camera area
    cam_resized = cv2.resize(frame, (CAM_W, CAM_H))

    # Paste camera into canvas
    canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = cam_resized

    # Panel origin
    PNL_X = CW - PANEL_W - PAD
    PNL_Y = HDR_H + PAD

    # Scale factor for landmark drawing on canvas
    scale_x = CAM_W / fw
    scale_y = CAM_H / fh

    # ── Defaults ────────────────────────────────────────────
    head_drop      = False
    ex_main_status = "Calibrating..."
    ex_main_level  = "warn"
    bottom_status  = "Initialising..."
    bottom_level   = "ok"

    rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)
    pose_results = pose.process(rgb)

    # ============================================================
    # ████  EXERCISE MODE — PURE FATIGUE ENGINE (logic unchanged) ████
    # ============================================================
    if mode == "EXERCISE" and pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        if not visibility_ok(lm):
            txt(canvas,
                "Move Back: Full Body Must Be Visible",
                CAM_X + 20, CAM_Y + CAM_H//2,
                C.CRIT, scale=0.70, thick=2)
        else:
            # Draw skeleton scaled to canvas camera area
            # We draw directly on cam_resized then paste back
            mp_drawing.draw_landmarks(
                cam_resized,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                POSE_DRAW_SPEC, POSE_CONN_SPEC)
            canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = cam_resized

            # Calibration phase
            if not calib_done:
                reps_so_far = len(calib_roms)
                update_calibration(lm, fw, fh, current_time)
                draw_exercise_panel(
                    canvas, PNL_X, PNL_Y, PANEL_W,
                    0.0, [], [], "Calibrating", "warn",
                    exercise_type, False, reps_so_far)
                bottom_status = "Calibrating — perform your normal reps"
                bottom_level  = "warn"

            # Fatigue detection phase
            else:
                if exercise_type == "SQUATS":
                    conditions_fired, status_msgs = run_squat_fatigue(
                        lm, fw, fh, current_time)
                else:
                    conditions_fired, status_msgs = run_surya_fatigue(
                        lm, fw, fh, current_time)
                    prev_lm_snap = snapshot(lm, fw, fh)

                active_conditions = conditions_fired
                n_fired           = len(conditions_fired)
                fi_display        = update_fatigue_index(n_fired)
                fatigue_confirmed = (n_fired >= MIN_CONDITIONS_FOR_FATIGUE)

                if fatigue_confirmed:
                    ex_main_status = "FATIGUE DETECTED"
                    ex_main_level  = "crit"
                    alert_text     = "⚠  FATIGUE DETECTED"
                    alert_until    = current_time + 2.5
                    bottom_status  = "FATIGUE DETECTED — consider resting"
                    bottom_level   = "crit"
                    print(f"FATIGUE DETECTED — conditions: {conditions_fired}")
                elif fi_display > 60:
                    ex_main_status = "Moderate Fatigue"
                    ex_main_level  = "crit"
                    bottom_status  = "Moderate Fatigue Building"
                    bottom_level   = "warn"
                elif fi_display > 30:
                    ex_main_status = "Mild Fatigue"
                    ex_main_level  = "warn"
                    bottom_status  = "Mild Fatigue — monitor closely"
                    bottom_level   = "warn"
                else:
                    ex_main_status = "Exercising Normally"
                    ex_main_level  = "ok"
                    bottom_status  = "Exercising Normally"
                    bottom_level   = "ok"

                draw_exercise_panel(
                    canvas, PNL_X, PNL_Y, PANEL_W,
                    fi_display, status_msgs, conditions_fired,
                    ex_main_status, ex_main_level,
                    exercise_type, True, CALIB_REPS_NEEDED)

    # ============================================================
    # ████  COGNITIVE MODE — LOGIC COMPLETELY UNTOUCHED  ████
    # ============================================================
    if face_results.multi_face_landmarks:
        for fl in face_results.multi_face_landmarks:

            nose_y = int(fl.landmark[NOSE_INDEX].y * fh)
            chin_y = int(fl.landmark[CHIN_INDEX].y * fh)
            if (chin_y - nose_y) < HEAD_DROP_COG:
                head_drop = True

            if mode == "COGNITIVE":
                lep, rep_pts = [], []
                for i in LEFT_EYE:
                    lep.append((int(fl.landmark[i].x*fw),
                                int(fl.landmark[i].y*fh)))
                for i in RIGHT_EYE:
                    rep_pts.append((int(fl.landmark[i].x*fw),
                                   int(fl.landmark[i].y*fh)))

                ear = (eye_aspect_ratio(lep) +
                       eye_aspect_ratio(rep_pts)) / 2.0

                # Blink detection
                if ear < EAR_THRESHOLD:
                    if blink_start is None:
                        blink_start = current_time
                    if closed_start_time is None:
                        closed_start_time = current_time
                else:
                    if blink_start is not None:
                        d = current_time - blink_start
                        if BLINK_MIN_DURATION < d < BLINK_MAX_DURATION:
                            blink_timestamps.append(current_time)
                        blink_start = None
                    closed_start_time = None

                blink_timestamps    = [t for t in blink_timestamps
                                       if current_time - t <= 60]
                blink_rate_smoothed = (0.1 * len(blink_timestamps) +
                                       0.9 * blink_rate_smoothed)

                fi_inc = 0.0
                if closed_start_time and current_time - closed_start_time > 2:
                    fi_inc += 0.05
                if head_drop:
                    fi_inc += 0.02
                if blink_rate_smoothed < 8:
                    fi_inc += 0.01

                fatigue_index_cog += fi_inc
                if fi_inc == 0:
                    fatigue_index_cog -= 0.01
                fatigue_index_cog = max(0.0, min(5.0, fatigue_index_cog))

                # Cognitive status strings
                if fatigue_index_cog >= 3:
                    cog_status = "HIGH FATIGUE ALERT!"
                    bottom_level = "crit"
                elif fatigue_index_cog >= 2:
                    cog_status = "MODERATE FATIGUE"
                    bottom_level = "warn"
                elif fatigue_index_cog >= 1:
                    cog_status = "SLIGHT FATIGUE"
                    bottom_level = "warn"
                else:
                    cog_status = "NORMAL"
                    bottom_level = "ok"

                bottom_status   = cog_status
                head_status_str = "Head Drop Detected!" if head_drop else "Head Normal"
                head_lv         = "crit" if head_drop else "ok"

                # Draw camera frame onto canvas with face mesh ON
                # (face mesh drawn on cam_resized)
                canvas[CAM_Y:CAM_Y+CAM_H,
                       CAM_X:CAM_X+CAM_W] = cam_resized

                # Draw cognitive panel on right side
                draw_cognitive_panel(
                    canvas, PNL_X, PNL_Y, PANEL_W,
                    ear, blink_rate_smoothed,
                    fatigue_index_cog,
                    head_status_str, head_lv)

    # ============================================================
    # ████  HEADER BAR  ████
    # ============================================================
    mode_badge    = "COGNITIVE MODE" if mode == "COGNITIVE" else "EXERCISE MODE"
    ex_badge      = exercise_type if mode == "EXERCISE" else ""
    draw_header(canvas, CW, mode_badge, ex_badge)

    # ============================================================
    # ████  FATIGUE ALERT BANNER (exercise)  ████
    # ============================================================
    if mode == "EXERCISE" and current_time < alert_until and alert_text:
        draw_alert_banner(canvas, CW, alert_text)

    # ============================================================
    # ████  BOTTOM STATUS BAR  ████
    # ============================================================
    keys_hint = "[M] Mode    [S] Squats    [N] Surya Namaskar    [Q] Quit"
    draw_bottom_bar(canvas, CW, CH,
                    bottom_status, bottom_level, keys_hint)

    # ── Thin camera frame border ──────────────────────────────
    cv2.rectangle(canvas,
                  (CAM_X-1, CAM_Y-1),
                  (CAM_X+CAM_W, CAM_Y+CAM_H),
                  C.PANEL_BORDER, 1)

    cv2.imshow(WIN_NAME, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('m'):
        if mode == "COGNITIVE":
            mode = "EXERCISE"
            reset_exercise()
        else:
            mode = "COGNITIVE"
            fatigue_index_cog   = 0.0
            blink_timestamps    = []
            blink_rate_smoothed = 15.0
            closed_start_time   = None
            blink_start         = None

    if key == ord('s') and mode == "EXERCISE":
        exercise_type = "SQUATS"
        reset_exercise()

    if key == ord('n') and mode == "EXERCISE":
        exercise_type = "SURYA"
        reset_exercise()

cap.release()
cv2.destroyAllWindows()