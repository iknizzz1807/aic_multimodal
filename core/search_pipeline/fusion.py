from typing import Dict, Any, List
from .base import PipelineStep
from api.schemas import VisualResult, AudioResult, AudioEventResult


class FusionStep(PipelineStep):
    """
    Bước FUSION: Kết hợp các danh sách ứng viên từ các nguồn khác nhau thành một
    danh sách duy nhất bằng cách sử dụng Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf

    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("🚀 [Pipeline] Step 2: FUSION")
        visual_candidates = context["visual_candidates"]
        audio_candidates = context["audio_candidates"]
        audio_event_candidates = context["audio_event_candidates"]

        fused_scores = self._fuse_results_rrf(
            visual_candidates, audio_candidates, audio_event_candidates
        )
        fused_candidates = sorted(
            fused_scores.items(), key=lambda item: item[1]["score"], reverse=True
        )

        context["fused_candidates"] = fused_candidates
        return context

    def _fuse_results_rrf(
        self,
        visual_results: List[VisualResult],
        audio_results: List[AudioResult],
        audio_event_results: List[AudioEventResult],
    ) -> Dict:
        """Hợp nhất kết quả bằng Reciprocal Rank Fusion (RRF)."""
        fused_scores = {}

        # Xử lý visual results
        for rank, res in enumerate(visual_results):
            media_id = res.id
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": set()}
            rrf_score = 1.0 / (self.k_rrf + rank + 1)
            fused_scores[media_id]["score"] += rrf_score
            fused_scores[media_id]["reason"].add("visual")

        # Xử lý audio results
        for rank, res in enumerate(audio_results):
            media_id = res.id
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": set()}
            rrf_score = 1.0 / (self.k_rrf + rank + 1)
            fused_scores[media_id]["score"] += (
                rrf_score * 1.2
            )  # Tăng nhẹ trọng số cho audio
            fused_scores[media_id]["reason"].add("audio")

        # Xử lý audio event results
        for rank, res in enumerate(audio_event_results):
            media_id = res.id
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": set()}
            rrf_score = 1.0 / (self.k_rrf + rank + 1)
            # Trọng số cho audio event có thể điều chỉnh
            fused_scores[media_id]["score"] += rrf_score * 1.1
            fused_scores[media_id]["reason"].add("audio_event")

        return fused_scores
