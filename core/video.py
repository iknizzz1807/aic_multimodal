import cv2
from typing import Iterator, Dict, Any
import config
import numpy as np

# Import thư viện PySceneDetect
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import os


class VideoProcessor:
    """
    Trích xuất keyframe từ file video bằng hai phương pháp có thể cấu hình:
    1. 'interval': Nhanh, lấy frame theo khoảng thời gian cố định.
    2. 'scenedetect': Chậm, phát hiện cảnh thông minh để lấy frame đại diện.
    """

    def __init__(self, frame_interval: float, extraction_method: str = "interval"):
        self.frame_interval = frame_interval
        self.extraction_method = extraction_method
        self.pyscene_threshold = 27.0
        print(
            f"✅ VideoProcessor initialized. Using '{self.extraction_method}' method."
        )

    def extract_keyframes(self, video_path: str) -> Iterator[Dict[str, Any]]:
        """
        Hàm điều phối, gọi phương thức trích xuất keyframe phù hợp dựa trên cấu hình.
        """
        if self.extraction_method == "scenedetect":
            yield from self._extract_keyframes_scenedetect(video_path)
        else:  # Mặc định là 'interval' để đảm bảo tốc độ
            yield from self._extract_keyframes_interval(video_path)

    def _extract_keyframes_interval(self, video_path: str) -> Iterator[Dict[str, Any]]:
        """
        Trích xuất keyframe theo khoảng thời gian cố định. Rất nhanh.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"   Error: Could not open video file {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 25  # Giá trị mặc định nếu không đọc được FPS

            frame_skip = int(fps * self.frame_interval)
            frame_count = 0

            print(
                f"   Extracting frames from {os.path.basename(video_path)} every {self.frame_interval}s..."
            )

            while cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_count / fps

                # Chuyển BGR (OpenCV) sang RGB (PIL/CLIP)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield {
                    "frame_rgb": frame_rgb,
                    "timestamp": round(timestamp, 2),
                }

                frame_count += frame_skip
                if frame_count > cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    break

            cap.release()
        except Exception as e:
            print(
                f"   Error during interval-based keyframe extraction for {video_path}: {e}"
            )
            return

    def _extract_keyframes_scenedetect(
        self, video_path: str
    ) -> Iterator[Dict[str, Any]]:
        """Trích xuất keyframe thông minh từ video sử dụng PySceneDetect (Chậm)."""
        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(threshold=self.pyscene_threshold)
            )

            print(
                f"   Detecting scenes in {os.path.basename(video_path)} (this may take a while)..."
            )
            scene_manager.detect_scenes(video=video, show_progress=False)
            scene_list = scene_manager.get_scene_list()

            if not scene_list:
                print(
                    f"   No scenes detected in {video_path}. Taking one default frame."
                )
                video.seek(1.0)
                frame_img = video.read()
                if frame_img is not False:
                    yield {
                        "frame_rgb": cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB),
                        "timestamp": 1.0,
                    }
                return

            print(f"   Detected {len(scene_list)} scenes in {video_path}.")
            for scene in scene_list:
                start_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()
                middle_sec = start_sec + (end_sec - start_sec) / 2.0

                video.seek(middle_sec)
                frame_img = video.read()

                if frame_img is not False:
                    frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    yield {
                        "frame_rgb": frame_rgb,
                        "timestamp": round(middle_sec, 2),
                    }
        except Exception as e:
            print(
                f"   Error during PySceneDetect keyframe extraction for {video_path}: {e}"
            )
            return
