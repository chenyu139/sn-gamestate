import cv2
import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path

from tracklab.utils.cv2 import draw_text
from tracklab.visualization import ImageVisualizer

from sn_calibration_baseline.soccerpitch import SoccerPitch

import logging

log = logging.getLogger(__name__)

pitch_file = Path(__file__).parent / "Radar.png"


class Pitch(ImageVisualizer):
    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        draw_pitch(image, detections_pred, detections_gt, image_pred)

class Radar(ImageVisualizer):
    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        for detection, group in zip([detections_pred, detections_gt], ["Predictions", "Ground Truth"]):
            if detection is not None and "bbox_pitch" in detection:
                draw_radar_view(image, detection, group=group)

class Minimap(ImageVisualizer):
    def preproces(self, detections_pred, detections_gt, image_pred, image_gt):
        temporal_smooth_detections(detections_pred)

    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        image_height, image_width = image.shape[:2]
        image[:] = minimap_background(image_width, image_height)
        if detections_pred is not None and "bbox_pitch" in detections_pred:
            draw_minimap_view(image, detections_pred)

class ComparisonMinimap(ImageVisualizer):
    def preproces(self, detections_pred, detections_gt, image_pred, image_gt):
        temporal_smooth_detections(detections_pred)

    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        panel_height, panel_width = image.shape[:2]
        image[:] = compose_comparison_view(
            image,
            detections_pred,
            output_width=panel_width,
            output_height=panel_height,
        )

def draw_pitch(
    patch,
    detections_pred,
    detections_gt,
    image_pred,
    line_thickness=3,
):
    # Draw the lines on the image pitch
    if "lines" in image_pred:
        image_height, image_width, _ = patch.shape
        for name, line in image_pred["lines"].items():
            if name == "Circle central" and len(line) > 4:
                points = np.array([(int(p["x"] * image_width), int(p["y"]*image_height)) for p in line])
                ellipse = cv2.fitEllipse(points)
                cv2.ellipse(patch, ellipse, color=SoccerPitch.palette[name], thickness=line_thickness)
            else:
                for j in np.arange(len(line)-1):
                    cv2.line(
                        patch,
                        (int(line[j]["x"] * image_width), int(line[j]["y"] * image_height)),
                        (int(line[j+1]["x"] * image_width), int(line[j+1]["y"] * image_height)),
                        color=SoccerPitch.palette[name],
                        thickness=line_thickness,  # TODO : make this a parameter
                    )

def draw_radar_view(patch, detections, scale=4, delta=32, group="Ground Truth"):
    pitch_width = 105 + 2 * 10  # pitch size + 2 * margin
    pitch_height = 68 + 2 * 5  # pitch size + 2 * margin
    sign = -1 if group == "Ground Truth" else +1
    y_delta = 3
    radar_center_x = int(1920/2 - pitch_width * scale / 2 * sign - delta * sign)
    radar_center_y = int(1080 - pitch_height * scale / 2 - y_delta)
    radar_top_x = int(radar_center_x - pitch_width * scale / 2)
    radar_top_y = int(1080 - pitch_height * scale - y_delta)
    radar_width = int(pitch_width * scale)
    radar_height = int(pitch_height * scale)
    if pitch_file is not None:
        radar_img = cv2.resize(cv2.imread(str(pitch_file)), (pitch_width * scale, pitch_height * scale))
        cv2.line(radar_img, (0, 0), (0, radar_img.shape[0]), thickness=6, color=(0, 0, 255))
        cv2.line(radar_img, (radar_img.shape[1], 0), (radar_img.shape[1], radar_img.shape[0]), thickness=6, color=(255, 0, 0))
    else:
        radar_img = np.ones((pitch_height * scale, pitch_width * scale, 3)) * 255

    alpha = 0.3
    patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,:] = cv2.addWeighted(patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width, :], 1-alpha, radar_img, alpha, 0.0)
    patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :] = cv2.addWeighted(patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :], 1-alpha, radar_img, alpha, 0.0)
    draw_text(
        patch,
        group,
        (radar_center_x, radar_top_y - 5),
        0, 1, 1,
        color_txt=(255, 255, 255),
        color_bg=None,
        alignH="c",
        alignV="t",
    )
    for name, detection in detections.iterrows():
        if "role" in detection and detection.role == "ball":
            continue
        if "role" in detection and "team" in detection:
            color = (0, 0, 255) if detection.team == "left" else (255, 0, 0)
        else:
            color = (0, 0, 0)
        bbox_name = "bbox_pitch"
        if not isinstance(detection[bbox_name], dict):
            continue
        x_middle = np.clip(detection[bbox_name]["x_bottom_middle"], -10000, 10000)
        y_middle = np.clip(detection[bbox_name]["y_bottom_middle"], -10000, 10000)
        cat = None
        if "jersey_number" in detection and detection.jersey_number is not None:
            if "role" in detection and detection.role == "player":
                if isinstance(detection.jersey_number, float) and np.isnan(detection.jersey_number):
                    cat = None
                else:
                    cat = f"{int(detection.jersey_number)}"

        if "role" in detection:
            if detection.role == "goalkeeper":
                cat = "GK"
            elif detection.role == "referee":
                cat = "RE"
                color = (238, 210, 2)
            elif detection.role == "other":
                cat = "OT"
                color = (0, 255, 0)
        if cat is not None:
            draw_text(
                patch,
                cat,
                (radar_center_x + int(x_middle * scale),
                 radar_center_y + int(y_middle * scale)),
                1,
                0.2*scale,
                1,
                color_txt=color,
                color_bg=None,
                alignH="c",
                alignV="b",
            )
        else:
            cv2.circle(
                patch,
                (radar_center_x + int(x_middle * scale),
                 radar_center_y + int(y_middle * scale)),
                scale,
                color=color,
                thickness=-1
            )

@lru_cache(maxsize=8)
def minimap_background(image_width, image_height):
    background = np.full((image_height, image_width, 3), (34, 120, 44), dtype=np.uint8)
    if pitch_file.is_file():
        pitch = cv2.imread(str(pitch_file))
        pitch = cv2.cvtColor(pitch, cv2.COLOR_BGR2RGB)
    else:
        pitch = np.full((780, 1250, 3), (63, 147, 78), dtype=np.uint8)
    pitch_height, pitch_width = pitch.shape[:2]
    scale = min(image_width / pitch_width, image_height / pitch_height)
    resized_width = max(1, int(round(pitch_width * scale)))
    resized_height = max(1, int(round(pitch_height * scale)))
    pitch = cv2.resize(pitch, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    top = (image_height - resized_height) // 2
    left = (image_width - resized_width) // 2
    background[top:top + resized_height, left:left + resized_width] = pitch
    return background

def draw_minimap_view(patch, detections):
    image_height, image_width = patch.shape[:2]
    pitch_width = 105 + 2 * 10
    pitch_height = 68 + 2 * 5
    scale = min(image_width / pitch_width, image_height / pitch_height)
    x_center = image_width / 2
    y_center = image_height / 2
    radius = max(6, int(round(scale * 0.75)))
    border = max(2, radius // 3)
    for _, detection in detections.iterrows():
        if "role" in detection and detection.role == "ball":
            continue
        bbox_pitch = detection.get("bbox_pitch")
        if not isinstance(bbox_pitch, dict):
            continue
        x_middle = np.clip(bbox_pitch["x_bottom_middle"], -pitch_width, pitch_width)
        y_middle = np.clip(bbox_pitch["y_bottom_middle"], -pitch_height, pitch_height)
        center = (
            int(round(x_center + x_middle * scale)),
            int(round(y_center + y_middle * scale)),
        )
        if "role" in detection and detection.role == "referee":
            color = (238, 210, 2)
        elif "team" in detection and detection.team == "left":
            color = (0, 0, 255)
        elif "team" in detection and detection.team == "right":
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
        cv2.circle(patch, center, radius + border, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(patch, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)

def compose_comparison_view(image, detections, output_width, output_height, separator_width=0):
    left_panel_width = max(1, (output_width - separator_width) // 2)
    right_panel_width = max(1, output_width - separator_width - left_panel_width)
    boxed_image = image.copy()
    draw_detection_ellipses(boxed_image, detections)
    left_panel = fit_image_to_panel(boxed_image, left_panel_width, output_height)
    right_panel = minimap_background(right_panel_width, output_height).copy()
    if detections is not None and "bbox_pitch" in detections:
        draw_minimap_view(right_panel, detections)
    if separator_width > 0:
        separator = np.full((output_height, separator_width, 3), (16, 16, 16), dtype=np.uint8)
        return np.concatenate([left_panel, separator, right_panel], axis=1)
    return np.concatenate([left_panel, right_panel], axis=1)

def fit_image_to_panel(image, panel_width, panel_height):
    source_height, source_width = image.shape[:2]
    scale = min(panel_width / source_width, panel_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    panel = np.full((panel_height, panel_width, 3), (18, 18, 18), dtype=np.uint8)
    top = (panel_height - resized_height) // 2
    left = (panel_width - resized_width) // 2
    panel[top:top + resized_height, left:left + resized_width] = resized
    return panel

def temporal_smooth_detections(detections, bbox_alpha=0.45, pitch_alpha=0.25, max_gap=5):
    if detections is None or "track_id" not in detections or "_vis_smoothed" in detections:
        return detections
    if "image_id" not in detections:
        detections["_vis_smoothed"] = True
        return detections
    for track_id, tracklet in detections.groupby("track_id", sort=False):
        if pd.isna(track_id):
            continue
        tracklet = tracklet.sort_values("image_id")
        previous_bbox = None
        previous_pitch = None
        previous_image_id = None
        for index, detection in tracklet.iterrows():
            image_id = image_order_value(detection.get("image_id"))
            if previous_image_id is not None and image_id - previous_image_id > max_gap:
                previous_bbox = None
                previous_pitch = None
            bbox = detection.get("bbox_ltwh")
            if bbox is not None and len(bbox) == 4:
                current_bbox = np.asarray(bbox, dtype=np.float32)
                if previous_bbox is None:
                    smoothed_bbox = current_bbox
                else:
                    smoothed_bbox = previous_bbox + bbox_alpha * (current_bbox - previous_bbox)
                detections.at[index, "bbox_ltwh"] = smoothed_bbox.tolist()
                previous_bbox = smoothed_bbox
            bbox_pitch = detection.get("bbox_pitch")
            if isinstance(bbox_pitch, dict):
                current_pitch = np.asarray(
                    [
                        bbox_pitch["x_bottom_middle"],
                        bbox_pitch["y_bottom_middle"],
                    ],
                    dtype=np.float32,
                )
                if previous_pitch is None:
                    smoothed_pitch = current_pitch
                else:
                    smoothed_pitch = previous_pitch + pitch_alpha * (current_pitch - previous_pitch)
                smoothed_bbox_pitch = dict(bbox_pitch)
                smoothed_bbox_pitch["x_bottom_middle"] = float(smoothed_pitch[0])
                smoothed_bbox_pitch["y_bottom_middle"] = float(smoothed_pitch[1])
                detections.at[index, "bbox_pitch"] = smoothed_bbox_pitch
                previous_pitch = smoothed_pitch
            previous_image_id = image_id
    detections["_vis_smoothed"] = True
    return detections

def image_order_value(image_id):
    if isinstance(image_id, (int, np.integer, float, np.floating)):
        return int(image_id)
    image_text = str(image_id)
    digits = "".join(char for char in Path(image_text).stem if char.isdigit())
    if digits:
        return int(digits)
    return 0

def draw_detection_boxes(image, detections):
    if detections is None or "bbox_ltwh" not in detections:
        return
    for _, detection in detections.iterrows():
        bbox = detection.get("bbox_ltwh")
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = [int(round(float(v))) for v in bbox]
        if w <= 1 or h <= 1:
            continue
        color = detection_color(detection)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2, lineType=cv2.LINE_AA)
        label = detection_label(detection)
        if label:
            draw_text(
                image,
                label,
                (x, max(0, y - 6)),
                fontFace=1,
                fontScale=0.75,
                thickness=1,
                alignH="l",
                alignV="b",
                color_bg=color,
                color_txt=None,
                alpha_bg=0.6,
            )

def draw_detection_ellipses(image, detections):
    if detections is None or "bbox_ltwh" not in detections:
        return
    for _, detection in detections.iterrows():
        if detection.get("role") == "ball":
            continue
        bbox = detection.get("bbox_ltwh")
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = [int(round(float(v))) for v in bbox]
        if w <= 1 or h <= 1:
            continue
        color = detection_color(detection)
        center = (int(x + w / 2), int(y + h))
        ellipse_width = max(10, int(round(0.58 * w)))
        ellipse_height = max(4, int(round(0.20 * w)))
        cv2.ellipse(
            image,
            center=center,
            axes=(ellipse_width, ellipse_height),
            angle=0.0,
            startAngle=-20.0,
            endAngle=200.0,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        label = detection_overlay_text(detection)
        if label:
            draw_text(
                image,
                label,
                (center[0], center[1] - max(8, ellipse_height + 6)),
                fontFace=1,
                fontScale=0.72,
                thickness=1,
                alignH="c",
                alignV="b",
                color_bg=color,
                color_txt=None,
                alpha_bg=0.6,
            )

def draw_panel_title(image, text):
    draw_text(
        image,
        text,
        (20, 20),
        fontFace=1,
        fontScale=1.0,
        thickness=2,
        alignH="l",
        alignV="t",
        color_bg=(0, 0, 0),
        color_txt=(255, 255, 255),
        alpha_bg=0.5,
    )

def detection_color(detection):
    if "role" in detection and detection.role == "referee":
        return (238, 210, 2)
    if "role" in detection and detection.role == "ball":
        return (255, 255, 255)
    if "team" in detection and detection.team == "left":
        return (0, 0, 255)
    if "team" in detection and detection.team == "right":
        return (255, 0, 0)
    if "role" in detection and detection.role == "goalkeeper":
        return (0, 255, 255)
    return (0, 255, 0)

def detection_label(detection):
    tokens = []
    track_id = detection.get("track_id")
    if not pd.isna(track_id):
        tokens.append(f"ID {int(track_id)}")
    jersey_number = detection.get("jersey_number")
    if isinstance(jersey_number, str) and jersey_number.strip():
        tokens.append(f"JN {jersey_number}")
    role = detection.get("role")
    if isinstance(role, str) and role:
        tokens.append(role.upper())
    return " | ".join(tokens)

def detection_overlay_text(detection):
    role = detection.get("role")
    if role == "ball":
        return ""
    for key in ["jn_tracklet", "jersey_number"]:
        jersey_number = detection.get(key)
        if isinstance(jersey_number, str):
            if jersey_number.strip():
                return jersey_number.strip()
        elif not pd.isna(jersey_number):
            return str(int(jersey_number))
    track_id = detection.get("track_id")
    if role == "goalkeeper":
        if not pd.isna(track_id):
            return f"GK {int(track_id)}"
        return "GK"
    if role == "referee":
        if not pd.isna(track_id):
            return f"RE {int(track_id)}"
        return "RE"
    if role == "other":
        if not pd.isna(track_id):
            return f"OT {int(track_id)}"
        return "OT"
    if not pd.isna(track_id):
        return str(int(track_id))
    return ""
