import os
import cv2
from typing import List
import config


class VideoExtractor:
    """Trích xuất các khung hình (frames) từ video."""

    def __init__(self, frame_interval: float = config.VIDEO_FRAME_EXTRACTION_INTERVAL):
        """
        Khởi tạo VideoExtractor.

        Args:
            frame_interval: Khoảng thời gian (giây) giữa các frame được trích xuất.
        """
        self.frame_interval = frame_interval
        print(f"✅ Video indexer initialized. Frame interval: {self.frame_interval}s")

    def extract_frames_from_directory(
        self,
        video_directory: str = config.DATA_DIR,
        output_frame_dir: str = config.FRAME_DIR,
    ) -> List[dict]:
        """
        Quét thư mục, tìm video và trích xuất các khung hình.

        Args:
            video_directory: Thư mục chứa file video.
            output_frame_dir: Thư mục để lưu các frame.

        Returns:
            Một list các dictionary, mỗi dict chứa thông tin về một frame đã lưu.
            Ví dụ:
            [
                {
                    "type": "video_frame",
                    "path": "output/frames/my_video_frame_1.0.jpg",
                    "media_id": "my_video.mp4",
                    "timestamp": 1.0
                }, ...
            ]
        """
        os.makedirs(output_frame_dir, exist_ok=True)

        video_files = [
            f
            for f in os.listdir(video_directory)
            if f.lower().endswith(config.VIDEO_EXTENSIONS)
        ]

        if not video_files:
            print("🔵 No video files found to process.")
            return []

        print(f"📹 Found {len(video_files)} video files. Starting frame extraction...")
        all_frames_data = []
        total_videos = len(video_files)

        for i, filename in enumerate(video_files):
            video_path = os.path.join(video_directory, filename)
            print(f"   Processing video {i+1}/{total_videos}: {filename}")

            try:
                frames_data = self._extract_frames_from_video(
                    video_path, output_frame_dir
                )
                all_frames_data.extend(frames_data)
                print(f"   ✅ Extracted {len(frames_data)} frames from {filename}")
            except Exception as e:
                print(f"   ❌ Error processing video {video_path}: {e}")

        print(
            f"🎉 Finished frame extraction. Total frames extracted: {len(all_frames_data)}"
        )
        return all_frames_data

    def _extract_frames_from_video(
        self, video_path: str, output_frame_dir: str
    ) -> List[dict]:
        """Trích xuất và lưu frame từ một file video cụ thể."""
        frames_data = []
        video_basename = os.path.basename(video_path)
        video_name_no_ext = os.path.splitext(video_basename)[0]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   Could not open video file: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"   Could not get FPS for {video_path}. Skipping.")
            cap.release()
            return []

        frame_interval_in_frames = int(fps * self.frame_interval)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval_in_frames == 0:
                current_time_sec = frame_count / fps

                frame_filename = (
                    f"{video_name_no_ext}_frame_at_{current_time_sec:.2f}s.jpg"
                )
                frame_path = os.path.join(output_frame_dir, frame_filename)

                # Lưu frame vào file
                cv2.imwrite(frame_path, frame)

                frames_data.append(
                    {
                        "type": "video_frame",
                        "path": frame_path,
                        "media_id": video_basename,  # Lưu tên file video gốc
                        "timestamp": round(current_time_sec, 2),
                    }
                )

            frame_count += 1

        cap.release()
        return frames_data
