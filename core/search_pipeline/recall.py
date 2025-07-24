import asyncio
from typing import List, Dict, Any

from elasticsearch import AsyncElasticsearch

import config
from .base import PipelineStep
from api.schemas import VisualResult, AudioResult, AudioEventResult
from core.vision import VisionProcessor
from core.audio import AudioEventProcessor
from database.milvus_connector import MilvusConnector


class RecallStep(PipelineStep):
    """
    Bước RECALL: Lấy các ứng viên ban đầu từ các nguồn dữ liệu khác nhau (Milvus, Elasticsearch).
    """

    def __init__(
        self,
        vision_processor: VisionProcessor,
        audio_event_processor: AudioEventProcessor,  # Thêm processor mới
        milvus_collection_getter,  # Đổi thành getter
        es_client: AsyncElasticsearch,
    ):
        self.vision_processor = vision_processor
        self.audio_event_processor = audio_event_processor
        self.milvus_collection_getter = milvus_collection_getter
        self.es_client = es_client

    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context["query"]
        k_recall = context["k_for_recall"]
        print(f"🚀 [Pipeline] Step 1: RECALL for '{text}' (k={k_recall})")

        visual_recall_task = self._perform_visual_search(text, k_recall)
        audio_recall_task = self._perform_audio_search(text, k_recall)
        audio_event_recall_task = self._perform_audio_event_search(
            text, k_recall
        )  # Task mới

        visual_candidates, audio_candidates, audio_event_candidates = (
            await asyncio.gather(
                visual_recall_task, audio_recall_task, audio_event_recall_task
            )
        )

        context["visual_candidates"] = visual_candidates
        context["audio_candidates"] = audio_candidates
        context["audio_event_candidates"] = audio_event_candidates
        return context

    async def _perform_visual_search(self, text: str, top_k: int) -> List[VisualResult]:
        """Thực hiện tìm kiếm vector bất đồng bộ trên Milvus."""
        print(f"🔍 Performing async visual search for: '{text}' (top_k={top_k})")
        query_vector = self.vision_processor.text_to_embedding(text)

        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}

        visual_collection = self.milvus_collection_getter(config.VISUAL_COLLECTION_NAME)

        def search_sync():
            return visual_collection.search(
                data=[query_vector.flatten()],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["media_id", "timestamp"],
            )

        results = await asyncio.to_thread(search_sync)

        return [
            VisualResult(
                id=hit.entity.get("media_id"),
                score=hit.distance,
                timestamp=hit.entity.get("timestamp"),
            )
            for hit in results[0]
        ]

    async def _perform_audio_search(self, text: str, top_k: int) -> List[AudioResult]:
        """Thực hiện tìm kiếm text bất đồng bộ trên Elasticsearch."""
        print(f"🗣️ Performing async audio search for: '{text}' (top_k={top_k})")
        query = {"match": {"text": {"query": text, "operator": "and"}}}
        response = await self.es_client.search(
            index=config.TRANSCRIPT_INDEX_NAME, query=query, size=top_k
        )

        return [
            AudioResult(
                id=hit["_source"]["media_id"],
                score=hit["_score"],
                match=hit["_source"]["text"],
                start=hit["_source"]["start"],
                end=hit["_source"]["end"],
            )
            for hit in response["hits"]["hits"]
        ]

    async def _perform_audio_event_search(
        self, text: str, top_k: int
    ) -> List[AudioEventResult]:
        """Thực hiện tìm kiếm sự kiện âm thanh bất đồng bộ trên Milvus bằng CLAP."""
        print(f"🔊 Performing async audio event search for: '{text}' (top_k={top_k})")

        # Dùng CLAP để tạo text embedding
        query_vector = self.audio_event_processor.get_text_embedding(text)

        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}

        audio_event_collection = self.milvus_collection_getter(
            config.AUDIO_EVENT_COLLECTION_NAME
        )

        def search_sync():
            return audio_event_collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["media_id", "start", "end"],
            )

        results = await asyncio.to_thread(search_sync)

        return [
            AudioEventResult(
                id=hit.entity.get("media_id"),
                score=hit.distance,
                start=hit.entity.get("start"),
                end=hit.entity.get("end"),
            )
            for hit in results[0]
        ]
