from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

from tracklab.callbacks import Progressbar
from tracklab.utils.cv2 import cv2_load_image
from tracklab.visualization.visualization_engine import (
    VisualizationEngine,
    create_draw_args,
    get_group,
    process_frame,
)


class GameStateVisualizationEngine(VisualizationEngine):
    def __init__(self, *args, num_workers=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._video_fps_cache = {}
        self.num_workers = max(1, int(num_workers))

    def _empty_detections(self):
        return pd.DataFrame(columns=["image_id", "bbox_ltwh"])

    def _groupable_detections(self, detections):
        if detections is None or "image_id" not in detections.columns:
            return self._empty_detections()
        return detections

    def _video_fps_from_image_path(self, file_path):
        file_path = str(file_path)
        if not file_path.startswith("vid://"):
            return float(self.video_fps)
        video_path = file_path.removeprefix("vid://").rsplit(":", 1)[0]
        if video_path not in self._video_fps_cache:
            cap = cv2.VideoCapture(video_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            self._video_fps_cache[video_path] = fps if fps > 0 else float(self.video_fps)
        return self._video_fps_cache[video_path]

    def visualize(self, tracker_state, video_id, detections, image_preds, progress=None):
        image_metadatas = tracker_state.image_metadatas[tracker_state.image_metadatas.video_id == video_id]
        image_gts = tracker_state.image_gt[tracker_state.image_gt.video_id == video_id]
        nframes = len(image_metadatas)
        video_name = tracker_state.video_metadatas.loc[video_id]["name"]
        for visualizer in self.visualizers.values():
            try:
                visualizer.preproces(detections, tracker_state.detections_gt, image_preds, tracker_state.image_gt)
            except Exception:
                pass
        total = self.max_frames or len(image_metadatas.index)
        progress = progress or Progressbar(dummy=True)
        progress.init_progress_bar("vis", "Visualization", total)
        detections = self._groupable_detections(detections)
        detections_gt = self._groupable_detections(tracker_state.detections_gt)
        detection_preds_by_image = detections.groupby("image_id")
        detection_gts_by_image = detections_gt.groupby("image_id")
        frame_ids = islice(image_metadatas.index, 0, None, max(1, nframes // total))
        args = (
            create_draw_args(
                image_id,
                self,
                image_metadatas,
                get_group(detection_preds_by_image, image_id),
                get_group(detection_gts_by_image, image_id),
                image_gts,
                image_preds,
                nframes,
            )
            for image_id in frame_ids
        )
        if self.save_videos:
            image = cv2_load_image(image_metadatas.iloc[0].file_path)
            filepath = self.save_dir / "videos" / f"{video_name}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._video_fps_from_image_path(image_metadatas.iloc[0].file_path),
                (image.shape[1], image.shape[0]),
            )
        if self.num_workers == 1:
            iterator = map(process_frame, args)
            for output_image, file_name in iterator:
                if self.save_images:
                    filepath = self.save_dir / "images" / str(video_name) / Path(file_name).name
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    assert cv2.imwrite(str(filepath), output_image)
                if self.save_videos:
                    video_writer.write(output_image)
                progress.on_module_step_end(None, "vis", None, None)
        else:
            with Pool(processes=self.num_workers) as pool:
                for output_image, file_name in pool.imap(process_frame, args):
                    if self.save_images:
                        filepath = self.save_dir / "images" / str(video_name) / Path(file_name).name
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        assert cv2.imwrite(str(filepath), output_image)
                    if self.save_videos:
                        video_writer.write(output_image)
                    progress.on_module_step_end(None, "vis", None, None)
        if self.save_videos:
            video_writer.release()
