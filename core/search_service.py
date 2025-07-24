import asyncio
import os
import base64
from typing import List, Dict, Optional
import numpy as np
import cv2
from elasticsearch import AsyncElasticsearch
from sentence_transformers import CrossEncoder

import config
from core.vision import VisionProcessor
from api.schemas import (
    VisualResult,
    AudioResult,
    MatchDetail,
    UnifiedResult,
)
from database.milvus_connector import MilvusConnector

# Khởi tạo các tài nguyên dùng chung một lần khi service khởi động
print("🚀 Initializing Search Service...")
vision_processor = VisionProcessor(config.CLIP_MODEL_ID)
milvus_conn = MilvusConnector()
# Đảm bảo collection đã được load
milvus_collection = milvus_conn.get_collection()
if milvus_collection is None:
    milvus_conn.setup_visual_collection()
    milvus_collection = milvus_conn.get_collection()

es_client = AsyncElasticsearch(f"http://{config.ES_HOST}:{config.ES_PORT}")

# Tải model Cross-Encoder cho bước Re-ranking
print(f"Loading Re-ranker model: {config.CROSS_ENCODER_MODEL_ID}")
cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL_ID, max_length=512)
print("✅ Search Service is ready!")


class SearchService:
    """
    Chứa logic tìm kiếm, giao tiếp với Milvus và Elasticsearch.
    Các hàm tìm kiếm được thiết kế bất đồng bộ để tối ưu hiệu năng.
    """

    def __init__(self):
        # Các tài nguyên nặng đã được khởi tạo bên ngoài class
        self.vision_processor = vision_processor
        self.milvus_collection = milvus_collection
        self.es_client = es_client
        self.cross_encoder = cross_encoder

    async def _perform_visual_search(self, text: str, top_k: int) -> List[VisualResult]:
        """Thực hiện tìm kiếm vector bất đồng bộ trên Milvus."""
        print(f"🔍 Performing async visual search for: '{text}' (top_k={top_k})")
        query_vector = self.vision_processor.text_to_embedding(text)

        search_params = {
            "metric_type": "IP",
            "params": {
                "nprobe": 16
            },  # Tinh chỉnh giá trị này để cân bằng tốc độ/độ chính xác
        }

        # Milvus search là I/O-bound, có thể chạy trong aio.to_thread
        # Hoặc chờ pymilvus hỗ trợ async-native
        def search_sync():
            return self.milvus_collection.search(
                data=[query_vector.flatten()],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["media_id", "timestamp"],
            )

        results = await asyncio.to_thread(search_sync)

        visual_results = []
        for hit in results[0]:
            visual_results.append(
                VisualResult(
                    id=hit.entity.get("media_id"),
                    score=hit.distance,
                    timestamp=hit.entity.get("timestamp"),
                )
            )
        return visual_results

    async def _perform_audio_search(self, text: str, top_k: int) -> List[AudioResult]:
        """Thực hiện tìm kiếm text bất đồng bộ trên Elasticsearch."""
        print(f"🗣️ Performing async audio search for: '{text}' (top_k={top_k})")
        query = {
            "match": {
                "text": {
                    "query": text,
                    "operator": "and",  # Yêu cầu tất cả các từ trong query phải xuất hiện
                }
            }
        }

        response = await self.es_client.search(
            index=config.TRANSCRIPT_INDEX_NAME, query=query, size=top_k
        )

        audio_results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            audio_results.append(
                AudioResult(
                    id=source["media_id"],
                    score=hit["_score"],  # Elasticsearch score
                    match=source["text"],
                    start=source["start"],
                    end=source["end"],
                )
            )
        return audio_results

    async def perform_unified_search(
        self, text: str, top_k: int
    ) -> List[UnifiedResult]:
        """
        Thực hiện tìm kiếm hợp nhất theo kiến trúc Recall -> Fusion -> Re-rank.
        """
        print(f"🚀 Performing unified search for: '{text}' (top_k={top_k})")
        K_FOR_RECALL = 100  # Lấy nhiều ứng viên hơn ở bước recall

        # --- BƯỚC 1: RECALL (Bất đồng bộ) ---
        # Tạo các task để chạy song song
        visual_recall_task = self._perform_visual_search(text, K_FOR_RECALL)
        audio_recall_task = self._perform_audio_search(text, K_FOR_RECALL)

        # Chạy và đợi kết quả từ cả hai nguồn
        visual_candidates, audio_candidates = await asyncio.gather(
            visual_recall_task, audio_recall_task
        )

        # --- BƯỚC 2: FUSION ---
        # Dùng RRF để kết hợp kết quả
        fused_scores = self._fuse_results_rrf(visual_candidates, audio_candidates)
        fused_candidates = sorted(
            fused_scores.items(), key=lambda item: item[1]["score"], reverse=True
        )

        # --- BƯỚC 3: RE-RANK ---
        # Chỉ re-rank những ứng viên có kết quả audio match
        rerank_candidates = []
        non_rerank_candidates = []

        # Tạo map từ media_id -> text match đầu tiên để re-rank
        audio_match_map = {}
        for res in audio_candidates:
            if res.id not in audio_match_map:
                audio_match_map[res.id] = res.match

        for media_id, data in fused_candidates:
            if "audio" in data["reason"] and media_id in audio_match_map:
                rerank_candidates.append((media_id, data, audio_match_map[media_id]))
            else:
                non_rerank_candidates.append((media_id, data))

        if rerank_candidates:
            print(
                f"🧠 Re-ranking {len(rerank_candidates)} candidates with CrossEncoder..."
            )
            # Chuẩn bị cặp [query, text]
            pairs = [[text, candidate[2]] for candidate in rerank_candidates]

            # Chạy model (tác vụ nặng, chạy trong thread)
            rerank_scores = await asyncio.to_thread(
                self.cross_encoder.predict, pairs, show_progress_bar=False
            )

            # Cập nhật điểm số với điểm từ Cross-Encoder
            for i, score in enumerate(rerank_scores):
                # Kết hợp điểm RRF và điểm Cross-Encoder
                rerank_candidates[i][1]["score"] += (
                    score * 2.0
                )  # Tăng trọng số cho re-ranker

        # Ghép lại danh sách ứng viên và sắp xếp lần cuối
        all_candidates = [
            (item[0], item[1]) for item in rerank_candidates
        ] + non_rerank_candidates
        sorted_media = sorted(
            all_candidates, key=lambda item: item[1]["score"], reverse=True
        )

        # --- BƯỚC 4: TỔNG HỢP KẾT QUẢ ---
        all_details = {}
        # Thu thập chi tiết từ visual
        for res in visual_candidates:
            if res.id not in all_details:
                all_details[res.id] = []
            all_details[res.id].append(
                MatchDetail(type="visual", timestamp=res.timestamp, score=res.score)
            )
        # Thu thập chi tiết từ audio
        for res in audio_candidates:
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
        for media_id, data in sorted_media[:top_k]:
            # Lấy thumbnail cho kết quả
            image_base64 = await self._get_thumbnail_for_media(media_id)

            item_details = sorted(
                all_details.get(media_id, []),
                key=lambda d: d.timestamp or d.start_time or 0,
            )

            final_results.append(
                UnifiedResult(
                    id=media_id,
                    score=data["score"],
                    reason=sorted(list(data["reason"])),
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

        # Xử lý visual results
        for rank, res in enumerate(visual_results):
            media_id = res.id
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": set()}
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[media_id]["score"] += rrf_score
            fused_scores[media_id]["reason"].add("visual")

        # Xử lý audio results
        for rank, res in enumerate(audio_results):
            media_id = res.id
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": set()}
            # Có thể thêm trọng số cho audio nếu muốn
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[media_id]["score"] += rrf_score
            fused_scores[media_id]["reason"].add("audio")

        return fused_scores

    async def _get_thumbnail_for_media(self, media_id: str) -> Optional[str]:
        """Lấy ảnh thumbnail cho một media_id một cách an toàn và bất đồng bộ."""
        file_path = os.path.join(config.DATA_DIR, media_id)
        if not os.path.exists(file_path):
            return None

        # Nếu là file ảnh, trả về base64 của nó (hàm này đã an toàn)
        if any(media_id.lower().endswith(ext) for ext in config.IMAGE_EXTENSIONS):
            return self.vision_processor.image_to_base64(file_path)

        # Nếu là video, chạy logic blocking trong một thread riêng
        if any(media_id.lower().endswith(ext) for ext in config.VIDEO_EXTENSIONS):
            return await asyncio.to_thread(
                self._extract_video_thumbnail_sync, file_path
            )

        return None

    def _extract_video_thumbnail_sync(self, file_path: str) -> Optional[str]:
        """Hàm đồng bộ để trích xuất thumbnail. Chỉ để chạy trong a thread."""
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return None

            # Lấy frame ở giây thứ 1
            cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
            ret, frame = cap.read()
            cap.release()

            if ret:
                # Chuyển đổi frame (numpy array) sang base64
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            print(f"Error getting thumbnail for {os.path.basename(file_path)}: {e}")
        return None

    async def get_server_info(self) -> dict:
        """Lấy thông tin cơ bản về server và dữ liệu."""
        try:
            es_ok = await self.es_client.ping()
            milvus_ok = (
                self.milvus_collection.has_partition("_default")
                if self.milvus_collection
                else False
            )
        except Exception:
            es_ok = False
            milvus_ok = False

        return {
            "message": "AI Multimodal Search API is running!",
            "version": "3.0.0-beta",
            "database_status": {
                "elasticsearch": "ok" if es_ok else "error",
                "milvus": "ok" if milvus_ok else "error",
            },
            "indexed_visuals": self.milvus_collection.num_entities if milvus_ok else 0,
            "model_info": self.vision_processor.get_model_info(),
        }
