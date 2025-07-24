import asyncio
from typing import Dict, Any
from sentence_transformers import CrossEncoder
from .base import PipelineStep


class RerankStep(PipelineStep):
    """
    Bước RE-RANK: Sắp xếp lại các ứng viên hàng đầu bằng cách sử dụng một mô hình
    Cross-Encoder mạnh mẽ hơn để cải thiện độ chính xác.
    """

    def __init__(self, cross_encoder: CrossEncoder, rerank_weight: float = 2.0):
        self.cross_encoder = cross_encoder
        self.rerank_weight = rerank_weight

    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("🚀 [Pipeline] Step 3: RE-RANK")
        text_query = context["query"]
        fused_candidates = context["fused_candidates"]
        audio_candidates = context["audio_candidates"]

        # Tạo map từ media_id -> text match đầu tiên để re-rank
        audio_match_map = {
            res.id: res.match
            for res in reversed(audio_candidates)  # Đảo ngược để get match đầu tiên
        }

        rerank_candidates = []
        non_rerank_candidates = []
        for media_id, data in fused_candidates:
            if "audio" in data["reason"] and media_id in audio_match_map:
                rerank_candidates.append((media_id, data, audio_match_map[media_id]))
            else:
                non_rerank_candidates.append((media_id, data))

        if rerank_candidates:
            print(
                f"🧠 Re-ranking {len(rerank_candidates)} candidates with CrossEncoder..."
            )
            pairs = [[text_query, candidate[2]] for candidate in rerank_candidates]

            rerank_scores = await asyncio.to_thread(
                self.cross_encoder.predict, pairs, show_progress_bar=False
            )

            for i, score in enumerate(rerank_scores):
                # Kết hợp điểm RRF và điểm Cross-Encoder
                rerank_candidates[i][1]["score"] += score * self.rerank_weight

        all_candidates = [
            (item[0], item[1]) for item in rerank_candidates
        ] + non_rerank_candidates

        sorted_media = sorted(
            all_candidates, key=lambda item: item[1]["score"], reverse=True
        )

        context["sorted_media"] = sorted_media
        return context
