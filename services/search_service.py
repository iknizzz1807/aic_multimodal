import os
import json
from typing import List, Dict, Optional

import faiss
import numpy as np

import config
from image_processor import ImageProcessor
from api.schemas import (
    VisualResult,
    AudioResult,
    MatchDetail,
    UnifiedResult,
)  # (Sẽ tạo file schema này)

# Đảm bảo thư viện MKL không gây lỗi
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SearchService:
    """
    Chứa toàn bộ logic nghiệp vụ cho việc tìm kiếm đa phương tiện.
    Lớp này không biết gì về FastAPI hay HTTP.
    """

    def __init__(self):
        """Tải model, index, và các dữ liệu cần thiết."""
        print("🚀 Initializing Search Service...")
        # Load cấu hình
        self.visual_index_file = config.VISUAL_INDEX_FILE
        self.mapping_file = config.MEDIA_DATA_MAPPING_FILE
        self.transcript_dir = config.TRANSCRIPT_DIR

        # Khởi tạo các processor
        print("  - Initializing image processor...")
        self.image_processor = ImageProcessor(config.CLIP_MODEL_ID)

        # Tải index và dữ liệu
        print("  - Loading FAISS index and media data mapping...")
        self.index = faiss.read_index(self.visual_index_file)
        with open(self.mapping_file, "r", encoding="utf-8") as f:
            self.index_to_media_data = json.load(f)
        print(f"  ✅ Loaded index with {self.index.ntotal} vectors.")

        print("  - Creating media to thumbnail mapping...")
        self.media_id_to_thumbnail_path = self._create_thumbnail_mapping()
        print(
            f"  ✅ Created thumbnail mapping for {len(self.media_id_to_thumbnail_path)} media items."
        )

        print("  - Loading audio transcripts...")
        self.transcripts = self._load_transcripts()
        print(f"  ✅ Loaded {len(self.transcripts)} transcripts.")
        print("✅ Search Service is ready!")

    def _create_thumbnail_mapping(self) -> Dict[str, str]:
        """Tạo dictionary map từ media_id sang đường dẫn ảnh thumbnail."""
        mapping = {}
        sorted_media_data = sorted(
            self.index_to_media_data.values(), key=lambda x: x.get("timestamp") or 0
        )
        for data in sorted_media_data:
            media_id = data["media_id"]
            if media_id not in mapping:
                mapping[media_id] = data["path"]
        return mapping

    def _load_transcripts(self) -> Dict[str, List[Dict]]:
        """Tải tất cả các file transcript."""
        transcripts = {}
        if not os.path.exists(self.transcript_dir):
            return transcripts
        for filename in os.listdir(self.transcript_dir):
            if filename.endswith(".json"):
                base_name = os.path.splitext(filename)[0]
                matching_media_id = next(
                    (
                        key
                        for key in self.media_id_to_thumbnail_path
                        if os.path.splitext(key)[0] == base_name
                    ),
                    None,
                )
                if matching_media_id:
                    file_path = os.path.join(self.transcript_dir, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            transcripts[matching_media_id] = json.load(f)
                    except Exception as e:
                        print(f"Error loading transcript {filename}: {e}")
        return transcripts

    def perform_visual_search(self, text: str, top_k: int) -> List[VisualResult]:
        """Thực hiện tìm kiếm hình ảnh/frame video."""
        print(f"🔍 Performing visual search for: '{text}' (top_k={top_k})")
        query_vector = self.image_processor.text_to_embedding(text)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i in range(len(indices[0])):
            idx_str = str(indices[0][i])
            media_data = self.index_to_media_data.get(idx_str)
            if media_data:
                result_item = VisualResult(
                    id=media_data["media_id"],
                    score=float(distances[0][i]),
                    timestamp=media_data.get("timestamp"),
                    image=ImageProcessor.image_to_base64(media_data["path"]),
                )
                results.append(result_item)
        return results

    def perform_audio_search(self, text: str, top_k: int) -> List[AudioResult]:
        """Thực hiện tìm kiếm trong transcript."""
        print(f"🗣️ Performing audio search for: '{text}' (top_k={top_k})")
        found_matches = []
        query_lower = text.lower()

        for media_id, segments in self.transcripts.items():
            for segment in segments:
                if query_lower in segment["text"].lower():
                    found_matches.append(
                        AudioResult(
                            id=media_id,
                            score=1.0,
                            match=segment["text"],
                            start=segment["start"],
                            end=segment["end"],
                        )
                    )
        return sorted(found_matches, key=lambda x: x.score, reverse=True)[:top_k]

    def perform_unified_search(self, text: str, top_k: int) -> List[UnifiedResult]:
        """Thực hiện tìm kiếm hợp nhất."""
        print(f"🚀 Performing unified search for: '{text}' (top_k={top_k})")
        K_FOR_FUSION = 100
        visual_results = self.perform_visual_search(text, K_FOR_FUSION)
        audio_results = self.perform_audio_search(text, K_FOR_FUSION)

        fused_scores = self._fuse_results_rrf(visual_results, audio_results)

        all_details = {}
        for res in visual_results:
            if res.id not in all_details:
                all_details[res.id] = []
            all_details[res.id].append(
                MatchDetail(type="visual", timestamp=res.timestamp, score=res.score)
            )

        for res in audio_results:
            if res.id not in all_details:
                all_details[res.id] = []
            all_details[res.id].append(
                MatchDetail(
                    type="audio",
                    start_time=res.start,
                    end_time=res.end,
                    text_match=res.match,
                    score=res.score,
                )
            )

        final_results = []
        sorted_media = sorted(
            fused_scores.items(), key=lambda item: item[1]["score"], reverse=True
        )

        for media_id, data in sorted_media[:top_k]:
            thumbnail_path = self.media_id_to_thumbnail_path.get(media_id)
            image_base64 = (
                ImageProcessor.image_to_base64(thumbnail_path)
                if thumbnail_path
                else None
            )
            item_details = sorted(
                all_details.get(media_id, []),
                key=lambda d: d.timestamp or d.start_time or 0,
            )

            final_results.append(
                UnifiedResult(
                    id=media_id,
                    score=data["score"],
                    reason=sorted(data["reason"]),
                    image=image_base64,
                    details=item_details,
                )
            )
        return final_results

    def _fuse_results_rrf(
        self,
        visual_results: List[VisualResult],
        audio_results: List[AudioResult],
        k: int = 60,
    ) -> Dict:
        """Hợp nhất kết quả bằng Reciprocal Rank Fusion (RRF)."""
        fused_scores = {}
        for rank, res in enumerate(visual_results):
            media_id = res.id
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": []}
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[media_id]["score"] += rrf_score
            if "visual" not in fused_scores[media_id]["reason"]:
                fused_scores[media_id]["reason"].append("visual")

        for rank, res in enumerate(audio_results):
            media_id = res.id
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": []}
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[media_id]["score"] += rrf_score
            if "audio" not in fused_scores[media_id]["reason"]:
                fused_scores[media_id]["reason"].append("audio")
        return fused_scores

    def get_server_info(self) -> dict:
        """Lấy thông tin cơ bản về server và dữ liệu."""
        return {
            "message": "AI Multimodal Search API is running!",
            "status": "ok",
            "version": "2.1.0",
            "indexed_visuals": len(self.index_to_media_data),
            "indexed_audios": len(self.transcripts),
            "model_info": self.image_processor.get_model_info(),
        }
