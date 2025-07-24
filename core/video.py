import cv2
from typing import Iterator, Dict, Any
import config


class VideoProcessor:
    """Extracts frames from video files."""

    def __init__(self, frame_interval: float):
        self.frame_interval = frame_interval
        print(f"✅ VideoProcessor initialized. Frame interval: {self.frame_interval}s")

    def extract_frames_in_memory(self, video_path: str) -> Iterator[Dict[str, Any]]:
        """Trích xuất frame từ video và trả về dưới dạng numpy array trong memory."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   Could not open video file: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            # Gán FPS mặc định nếu không lấy được
            fps = 25
            print(f"   Could not get FPS for {video_path}. Assuming {fps} FPS.")

        frame_interval_in_frames = int(fps * self.frame_interval)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval_in_frames == 0:
                current_time_sec = frame_count / fps

                # Chuyển BGR (OpenCV) sang RGB (PIL/CLIP)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield {
                    "frame_rgb": frame_rgb,  # Trả về numpy array
                    "timestamp": round(current_time_sec, 2),
                }

            frame_count += 1

        cap.release()
