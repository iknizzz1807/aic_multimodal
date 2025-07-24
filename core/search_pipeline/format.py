import os
import base64
import asyncio
from typing import Dict, Any, Optional
import cv2

import config
from .base import PipelineStep
from api.schemas import UnifiedResult, MatchDetail
from core.vision import VisionProcessor


class FormatStep(PipelineStep):
    """
    Bước FORMAT: Tổng hợp tất cả thông tin, lấy thumbnail,
    và định dạng kết quả cuối cùng theo schema API.
    """

    def __init__(self, vision_processor: VisionProcessor):
        self.vision_processor = vision_processor

    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("🚀 [Pipeline] Step 4: FORMAT")
        top_k = context["top_k"]
        sorted_media = context["sorted_media"]
        visual_candidates = context["visual_candidates"]
        audio_candidates = context["audio_candidates"]
        audio_event_candidates = context["audio_event_candidates"]

        # Thu thập tất cả chi tiết từ các nguồn
        all_details = {}
        for res in visual_candidates:
            all_details.setdefault(res.id, []).append(
                MatchDetail(type="visual", timestamp=res.timestamp, score=res.score)
            )
        for res in audio_candidates:
            all_details.setdefault(res.id, []).append(
                MatchDetail(
                    type="audio",
                    start_time=res.start,
                    end_time=res.end,
                    match_content=res.match,
                    score=res.score,
                )
            )
        for res in audio_event_candidates:
            all_details.setdefault(res.id, []).append(
                MatchDetail(
                    type="audio_event",
                    start_time=res.start,
                    end_time=res.end,
                    match_content=f"Detected audio event (score: {res.score:.2f})",
                    score=res.score,
                )
            )

        final_results = []
        # Tạo các task lấy thumbnail song song
        thumbnail_tasks = [
            self._get_thumbnail_for_media(media_id)
            for media_id, _ in sorted_media[:top_k]
        ]
        thumbnails = await asyncio.gather(*thumbnail_tasks)

        for i, (media_id, data) in enumerate(sorted_media[:top_k]):
            item_details = sorted(
                all_details.get(media_id, []),
                key=lambda d: d.timestamp or d.start_time or 0,
            )
            final_results.append(
                UnifiedResult(
                    id=media_id,
                    score=data["score"],
                    reason=sorted(list(data["reason"])),
                    image=thumbnails[i],
                    details=item_details,
                )
            )

        context["final_results"] = final_results
        return context

    async def _get_thumbnail_for_media(self, media_id: str) -> Optional[str]:
        """Lấy ảnh thumbnail cho một media_id một cách an toàn và bất đồng bộ."""
        file_path = os.path.join(config.DATA_DIR, media_id)
        if not os.path.exists(file_path):
            return None

        if any(media_id.lower().endswith(ext) for ext in config.IMAGE_EXTENSIONS):
            return self.vision_processor.image_to_base64(file_path)

        if any(media_id.lower().endswith(ext) for ext in config.VIDEO_EXTENSIONS):
            return await asyncio.to_thread(
                self._extract_video_thumbnail_sync, file_path
            )

        return None

    def _extract_video_thumbnail_sync(self, file_path: str) -> Optional[str]:
        """Hàm đồng bộ để trích xuất thumbnail. Chỉ để chạy trong một thread."""
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_MSEC, 1000)  # Lấy frame ở giây thứ 1
            ret, frame = cap.read()
            cap.release()
            if ret:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            print(f"Error getting thumbnail for {os.path.basename(file_path)}: {e}")
        return None
