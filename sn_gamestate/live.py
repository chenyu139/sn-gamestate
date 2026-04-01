import logging
from pathlib import Path
import subprocess
import threading
import time

import cv2
import pandas as pd

from tracklab.callbacks import Callback
from tracklab.datastruct import TrackingSet
from tracklab.datastruct.tracking_dataset import TrackingDataset
from tracklab.engine.engine import TrackingEngine, merge_dataframes
from tracklab.pipeline import Evaluator
from tracklab.utils.coordinates import ltrb_to_ltwh

from sn_gamestate.visualization.players import CompletePlayerEllipse


log = logging.getLogger(__name__)


class LiveStreamDataset(TrackingDataset):
    def __init__(self, dataset_path: str, source: str, eval_set: str = "live", stream_name: str = "live", **kwargs):
        self.source = source
        self.stream_name = stream_name
        video_metadatas = pd.DataFrame([{"id": stream_name, "name": stream_name}], index=[0])
        image_metadatas = pd.DataFrame(columns=["id", "name", "frame", "video_id", "file_path"])
        image_gt = pd.DataFrame(columns=["video_id"])
        detections_gt = pd.DataFrame(columns=["image_id", "video_id"])
        live_set = TrackingSet(
            video_metadatas=video_metadatas,
            image_metadatas=image_metadatas,
            detections_gt=detections_gt,
            image_gt=image_gt,
        )
        super().__init__(dataset_path=dataset_path, sets={eval_set: live_set}, **kwargs)


class NoOpEvaluator(Evaluator):
    def __init__(self, cfg=None, *args, **kwargs):
        self.cfg = cfg or {}

    def run(self, tracker_state):
        return None


class LatestFrameCapture:
    def __init__(self, source, buffer_size: int = 1):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_size)))
        self.buffer_size = max(1, int(buffer_size))
        self.lock = threading.Lock()
        self.latest = None
        self.latest_frame_idx = -1
        self.returned_frame_idx = -1
        self.ended = False
        self.stopped = False
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open live source: {source}")
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def _reader_loop(self):
        while not self.stopped:
            ok, frame = self.capture.read()
            if not ok:
                break
            with self.lock:
                self.latest_frame_idx += 1
                self.latest = (self.latest_frame_idx, frame)
        with self.lock:
            self.ended = True

    def isOpened(self):
        return self.capture.isOpened()

    def get(self, prop_id):
        return self.capture.get(prop_id)

    def read(self, timeout_seconds: float):
        deadline = None if timeout_seconds is None or timeout_seconds < 0 else time.time() + timeout_seconds
        while True:
            with self.lock:
                latest = self.latest
                ended = self.ended
            if latest is not None and latest[0] != self.returned_frame_idx:
                self.returned_frame_idx = latest[0]
                return True, latest[0], latest[1]
            if ended:
                return False, None, None
            if deadline is not None and time.time() >= deadline:
                return None, None, None
            time.sleep(0.001)

    def release(self):
        self.stopped = True
        if self.reader_thread.is_alive():
            self.reader_thread.join(timeout=0.5)
        self.capture.release()


class LiveVisualizationCallback(Callback):
    def __init__(
        self,
        show_window: bool = True,
        save_video: bool = False,
        output_path: str = "outputs/live/live.mp4",
        video_fps: int = 25,
        window_name: str = "sn-gamestate-live",
        display_track_id: bool = True,
        display_jersey: bool = False,
        display_role: bool = True,
        display_team: bool = True,
        colors=None,
        throttle_ms: int = 1,
        rtsp_url: str = None,
        ffmpeg_binary: str = "ffmpeg",
        rtsp_transport: str = "tcp",
        video_codec: str = "libx264",
        video_preset: str = "ultrafast",
        video_tune: str = "zerolatency",
        video_bitrate: str = None,
        **kwargs,
    ):
        self.show_window = show_window
        self.save_video = save_video
        self.output_path = Path(output_path)
        self.video_fps = video_fps
        self.window_name = window_name
        self.throttle_ms = max(1, int(throttle_ms))
        self.writer = None
        self.window_initialized = False
        self.rtsp_url = rtsp_url
        self.ffmpeg_binary = ffmpeg_binary
        self.rtsp_transport = rtsp_transport
        self.video_codec = video_codec
        self.video_preset = video_preset
        self.video_tune = video_tune
        self.video_bitrate = video_bitrate
        self.rtsp_process = None
        self.player_visualizer = CompletePlayerEllipse(
            display_track_id=display_track_id,
            display_jersey=display_jersey,
            display_role=display_role,
            display_team=display_team,
        )
        self.player_visualizer.post_init(colors=colors or {})

    def _init_writer(self, frame_bgr):
        if not self.save_video or self.writer is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame_bgr.shape[:2]
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.video_fps),
            (width, height),
        )
        log.info(f"Live video writer initialized at {self.output_path}")

    def _init_rtsp_stream(self, frame_bgr):
        if not self.rtsp_url or self.rtsp_process is not None:
            return
        height, width = frame_bgr.shape[:2]
        command = [
            self.ffmpeg_binary,
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(float(self.video_fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            self.video_codec,
            "-preset",
            self.video_preset,
            "-tune",
            self.video_tune,
        ]
        if self.video_bitrate:
            command.extend(["-b:v", str(self.video_bitrate)])
        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-f",
                "rtsp",
                "-rtsp_transport",
                self.rtsp_transport,
                self.rtsp_url,
            ]
        )
        self.rtsp_process = subprocess.Popen(command, stdin=subprocess.PIPE)
        log.info(f"Live RTSP publisher initialized at {self.rtsp_url}")

    def on_image_loop_end(self, engine, image_metadata, image, image_idx, detections):
        frame_rgb = image.copy()
        frame_detections = detections
        if hasattr(frame_detections, "image_id") and "image_id" in frame_detections.columns:
            frame_detections = frame_detections[frame_detections.image_id == image_metadata.name]
        for _, detection in frame_detections.iterrows():
            self.player_visualizer.draw_detection(frame_rgb, detection, None)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._init_writer(frame_bgr)
        self._init_rtsp_stream(frame_bgr)
        if self.writer is not None:
            self.writer.write(frame_bgr)
        if self.rtsp_process is not None and self.rtsp_process.stdin is not None:
            try:
                self.rtsp_process.stdin.write(frame_bgr.tobytes())
            except BrokenPipeError:
                log.exception("Live RTSP publisher stopped unexpectedly")
                self.rtsp_process = None
        if self.show_window:
            if not self.window_initialized:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                self.window_initialized = True
            cv2.imshow(self.window_name, frame_bgr)
            cv2.waitKey(self.throttle_ms)

    def on_dataset_track_end(self, engine):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.rtsp_process is not None:
            if self.rtsp_process.stdin is not None:
                self.rtsp_process.stdin.close()
            self.rtsp_process.wait(timeout=2)
            self.rtsp_process = None
        if self.window_initialized:
            cv2.destroyWindow(self.window_name)
            self.window_initialized = False


class LiveTrackingEngine(TrackingEngine):
    def __init__(
        self,
        modules,
        tracker_state,
        num_workers: int,
        source: str,
        target_fps: int = 10,
        max_frames: int = -1,
        visualization_miss_tolerance: int = 2,
        detection_interval: int = 1,
        drop_old_frames: bool = False,
        source_buffer_size: int = 1,
        read_timeout_ms: int = 1000,
        max_history_frames: int = 120,
        callbacks=None,
    ):
        super().__init__(modules=modules, tracker_state=tracker_state, num_workers=num_workers, callbacks=callbacks)
        self.source = source
        self.target_fps = int(target_fps)
        self.max_frames = int(max_frames)
        self.visualization_miss_tolerance = max(0, int(visualization_miss_tolerance))
        self.detection_interval = max(1, int(detection_interval))
        self.drop_old_frames = bool(drop_old_frames)
        self.source_buffer_size = max(1, int(source_buffer_size))
        self.read_timeout_seconds = max(0.001, int(read_timeout_ms) / 1000.0)
        self.max_history_frames = max(1, int(max_history_frames))

    def _open_source(self):
        source = int(self.source) if str(self.source).isdigit() else self.source
        if self.drop_old_frames:
            return LatestFrameCapture(source, buffer_size=self.source_buffer_size)
        capture = cv2.VideoCapture(source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.source_buffer_size)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open live source: {self.source}")
        return capture

    def _read_frame(self, capture):
        if isinstance(capture, LatestFrameCapture):
            return capture.read(timeout_seconds=self.read_timeout_seconds)
        ok, frame_bgr = capture.read()
        return ok, None, frame_bgr

    def _frame_stride(self, capture):
        if self.target_fps <= 0:
            return 1
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0 or source_fps <= self.target_fps:
            return 1
        return max(1, round(source_fps / self.target_fps))

    def _estimated_total_frames(self, capture, frame_stride):
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_frame_count > 0:
            estimated = max(1, (source_frame_count + frame_stride - 1) // frame_stride)
        elif self.max_frames > 0:
            estimated = self.max_frames
        else:
            estimated = 1
        if self.max_frames > 0:
            estimated = min(estimated, self.max_frames)
        return estimated

    def _tracker_visualization_detections(self, current_frame_detections):
        if self.visualization_miss_tolerance <= 0:
            return current_frame_detections
        existing_track_ids = set()
        if "track_id" in current_frame_detections.columns:
            existing_track_ids = {
                int(track_id)
                for track_id in current_frame_detections.track_id.dropna().tolist()
            }
        predicted_tracks = []
        for model in self.models.values():
            tracker = getattr(getattr(model, "model", None), "tracker", None)
            if tracker is not None and hasattr(tracker, "tracks"):
                for track in tracker.tracks:
                    if not track.is_confirmed():
                        continue
                    if track.time_since_update < 1 or track.time_since_update > self.visualization_miss_tolerance:
                        continue
                    if int(track.track_id) in existing_track_ids:
                        continue
                    predicted_tracks.append(
                        pd.Series(
                            {
                                "track_id": int(track.track_id),
                                "track_bbox_kf_ltwh": track.to_ltwh(),
                                "track_bbox_pred_kf_ltwh": track.last_kf_pred_ltwh,
                                "time_since_update": track.time_since_update,
                                "from_tracker_prediction": True,
                            }
                        )
                    )
            trackers = getattr(getattr(model, "model", None), "trackers", None)
            if trackers is not None:
                for track in trackers:
                    track_id = int(track.id + 1)
                    if track.time_since_update < 1 or track.time_since_update > self.visualization_miss_tolerance:
                        continue
                    if track_id in existing_track_ids:
                        continue
                    predicted_tracks.append(
                        pd.Series(
                            {
                                "track_id": track_id,
                                "track_bbox_kf_ltwh": ltrb_to_ltwh(track.get_state()[0]),
                                "time_since_update": track.time_since_update,
                                "from_tracker_prediction": True,
                            }
                        )
                    )
        if not predicted_tracks:
            return current_frame_detections
        predicted_tracks_df = pd.DataFrame(predicted_tracks)
        return pd.concat([current_frame_detections, predicted_tracks_df], ignore_index=False, sort=False)

    def _is_bbox_detector(self, model):
        output_columns = set(getattr(model, "output_columns", []) or [])
        return model.level == "image" and {"bbox_ltwh", "bbox_conf"}.issubset(output_columns)

    def _prune_history(self, detections, image_pred, frame_idx):
        min_frame_idx = frame_idx - self.max_history_frames
        if len(detections) > 0 and "image_id" in detections.columns:
            detections = detections[detections.image_id >= min_frame_idx]
        if not image_pred.empty and "frame" in image_pred.columns:
            image_pred = image_pred[image_pred.frame >= min_frame_idx]
        return detections, image_pred

    def video_loop(self, tracker_state, video_metadata, video_id):
        for model in self.models.values():
            if hasattr(model, "reset"):
                model.reset()
        capture = self._open_source()
        log.info(f"Live source opened: {self.source}")
        frame_stride = self._frame_stride(capture)
        log.info(f"Live processing frame stride: {frame_stride}")
        estimated_total_frames = self._estimated_total_frames(capture, frame_stride)
        for model_name in self.module_names:
            self.callback("on_module_start", task=model_name, dataloader=[None] * estimated_total_frames)
        frame_idx = -1
        processed_frames = 0
        detections = pd.DataFrame()
        image_pred = pd.DataFrame(columns=["video_id", "file_path", "frame"])
        try:
            while capture.isOpened():
                ok, raw_frame_idx, frame_bgr = self._read_frame(capture)
                if ok is None:
                    continue
                if not ok:
                    log.info(f"Live source ended at raw frame {frame_idx}")
                    break
                if raw_frame_idx is None:
                    frame_idx += 1
                else:
                    frame_idx = int(raw_frame_idx)
                if frame_idx % frame_stride != 0:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                metadata = pd.Series(
                    {
                        "id": frame_idx,
                        "name": frame_idx,
                        "frame": frame_idx,
                        "video_id": video_id,
                        "file_path": None,
                        "timestamp": time.time(),
                    },
                    name=frame_idx,
                )
                metadata_df = pd.DataFrame([metadata.to_dict()], index=[metadata.name])
                image_pred = merge_dataframes(image_pred, metadata_df)
                self.callback(
                    "on_image_loop_start",
                    image_metadata=metadata,
                    image_idx=frame_idx,
                    index=processed_frames,
                )
                run_detector = processed_frames % self.detection_interval == 0
                for model_name in self.module_names:
                    model = self.models[model_name]
                    if model.level == "video":
                        raise RuntimeError(f"LiveTrackingEngine does not support video-level module '{model_name}'")
                    frame_detections = (
                        detections[detections.image_id == frame_idx]
                        if len(detections) > 0 and "image_id" in detections.columns
                        else pd.DataFrame()
                    )
                    if model.level == "image":
                        if self._is_bbox_detector(model) and not run_detector:
                            continue
                        batch = model.preprocess(image=frame_rgb, detections=frame_detections, metadata=metadata)
                        batch = type(model).collate_fn([(frame_idx, batch)])
                        detections, image_pred = self.default_step(batch, model_name, detections, image_pred)
                    elif model.level == "detection" and not frame_detections.empty:
                        batch_items = []
                        for detection_idx, detection in frame_detections.iterrows():
                            item = model.preprocess(image=frame_rgb, detection=detection, metadata=metadata)
                            batch_items.append((detection_idx, item))
                        if batch_items:
                            batch = type(model).collate_fn(batch_items)
                            detections, image_pred = self.default_step(batch, model_name, detections, image_pred)
                current_frame_detections = (
                    detections[detections.image_id == frame_idx]
                    if len(detections) > 0 and "image_id" in detections.columns
                    else pd.DataFrame()
                )
                current_frame_detections = self._tracker_visualization_detections(current_frame_detections)
                self.callback(
                    "on_image_loop_end",
                    image_metadata=metadata,
                    image=frame_rgb,
                    image_idx=frame_idx,
                    detections=current_frame_detections,
                )
                processed_frames += 1
                detections, image_pred = self._prune_history(detections, image_pred, frame_idx)
                if processed_frames == 1 or processed_frames % 25 == 0:
                    log.info(f"Live processed frames: {processed_frames}")
                if self.max_frames > 0 and processed_frames >= self.max_frames:
                    break
        except Exception:
            log.exception("Live processing failed")
            raise
        finally:
            capture.release()
            for model_name in self.module_names:
                self.callback("on_module_end", task=model_name, detections=detections)
        log.info(f"Live processing finished after {processed_frames} frames")
        return detections, image_pred
