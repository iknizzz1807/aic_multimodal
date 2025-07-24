from typing import List
from elasticsearch import AsyncElasticsearch
from sentence_transformers import CrossEncoder

import config
from core.vision import VisionProcessor
from core.audio import AudioEventProcessor  # Import processor mới
from api.schemas import UnifiedResult
from database.milvus_connector import MilvusConnector

# Import các bước của pipeline
from .search_pipeline.recall import RecallStep
from .search_pipeline.fusion import FusionStep
from .search_pipeline.rerank import RerankStep
from .search_pipeline.format import FormatStep

# --- KHỞI TẠO CÁC TÀI NGUYÊN DÙNG CHUNG ---
# Khởi tạo một lần khi service khởi động để tối ưu hiệu suất
print("🚀 Initializing Search Service Resources...")

# 1. Các model AI
vision_processor = VisionProcessor(config.CLIP_MODEL_ID)
audio_event_processor = AudioEventProcessor(config.CLAP_MODEL_ID)  # Khởi tạo CLAP
print(f"Loading Re-ranker model: {config.CROSS_ENCODER_MODEL_ID}")
cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL_ID, max_length=512)

# 2. Kết nối cơ sở dữ liệu
milvus_conn = MilvusConnector()
# Tạo một hàm để get collection thay vì một biến instance
get_milvus_collection = lambda name: milvus_conn.get_collection(name)
es_client = AsyncElasticsearch(f"http://{config.ES_HOST}:{config.ES_PORT}")

print("✅ Search Service is ready!")


class SearchService:
    """
    Dịch vụ tìm kiếm, điều phối một pipeline gồm các bước:
    Recall -> Fusion -> Re-rank -> Format.
    """

    def __init__(self):
        # Các tài nguyên nặng đã được khởi tạo bên ngoài và được chia sẻ
        self.vision_processor = vision_processor
        self.audio_event_processor = audio_event_processor
        self.get_milvus_collection = get_milvus_collection
        self.es_client = es_client
        self.cross_encoder = cross_encoder

        # --- Xây dựng Pipeline ---
        # Mỗi bước được khởi tạo với các dependency cần thiết của nó.
        self.search_pipeline = [
            RecallStep(
                self.vision_processor,
                self.audio_event_processor,
                self.get_milvus_collection,
                self.es_client,
            ),
            FusionStep(k_rrf=60),
            RerankStep(self.cross_encoder, rerank_weight=2.0),
            FormatStep(self.vision_processor),
        ]

    async def perform_unified_search(
        self, text: str, top_k: int
    ) -> List[UnifiedResult]:
        """
        Thực hiện tìm kiếm hợp nhất bằng cách chạy qua pipeline đã được định nghĩa.
        """
        print(f"🚀 Performing unified search for: '{text}' (top_k={top_k})")

        # 1. Chuẩn bị context ban đầu cho pipeline
        context = {
            "query": text,
            "top_k": top_k,
            "k_for_recall": 100,  # Lấy nhiều ứng viên hơn ở bước recall
        }

        # 2. Chạy tuần tự các bước trong pipeline
        for step in self.search_pipeline:
            context = await step(context)

        # 3. Trả về kết quả cuối cùng từ context
        return context.get("final_results", [])

    async def get_server_info(self) -> dict:
        """Lấy thông tin cơ bản về server và dữ liệu."""
        visual_collection = None
        audio_event_collection = None
        try:
            es_ok = await self.es_client.ping()
            # Kiểm tra cả hai collection
            visual_collection = self.get_milvus_collection(
                config.VISUAL_COLLECTION_NAME
            )
            audio_event_collection = self.get_milvus_collection(
                config.AUDIO_EVENT_COLLECTION_NAME
            )
            milvus_visual_ok = (
                visual_collection.has_partition("_default")
                if visual_collection
                else False
            )
            milvus_audio_event_ok = (
                audio_event_collection.has_partition("_default")
                if audio_event_collection
                else False
            )
        except Exception:
            es_ok = False
            milvus_visual_ok = False
            milvus_audio_event_ok = False

        return {
            "message": "AI Multimodal Search API is running!",
            "version": "3.2.0-advanced-features",
            "database_status": {
                "elasticsearch": "ok" if es_ok else "error",
                "milvus_visual": "ok" if milvus_visual_ok else "error",
                "milvus_audio_event": "ok" if milvus_audio_event_ok else "error",
            },
            "indexed_visuals": (
                visual_collection.num_entities
                if milvus_visual_ok and visual_collection
                else 0
            ),
            "indexed_audio_events": (
                audio_event_collection.num_entities
                if milvus_audio_event_ok and audio_event_collection
                else 0
            ),
            "model_info": self.vision_processor.get_model_info(),
        }
