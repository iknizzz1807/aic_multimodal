import os
import json
from typing import List, Optional, Dict

import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.image_processor import ImageProcessor
from src import config

# Đảm bảo thư viện MKL không gây lỗi trên một số hệ thống
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SearchQuery(BaseModel):
    text: str = Field(
        ...,
        description="Natural language description",
        example="a person walking on the beach",
    )
    top_k: int = Field(default=12, ge=1, le=50, description="Number of results (1-50)")


class VisualResult(BaseModel):
    id: str = Field(description="Image/Video identifier")
    score: float = Field(description="Similarity score")
    timestamp: Optional[float] = Field(description="Timestamp in video if applicable")
    image: Optional[str] = Field(
        description="Base64-encoded image data of the specific frame"
    )


class AudioResult(BaseModel):
    id: str = Field(description="Video/Audio identifier")
    score: float = Field(description="Match score (naive implementation)")
    match: str = Field(description="The matched text segment")
    start: float = Field(description="Start time of the segment")
    end: float = Field(description="End time of the segment")


class UnifiedResult(BaseModel):
    id: str = Field(description="Video/Media identifier")
    score: float = Field(description="Final fused score from Reciprocal Rank Fusion")
    reason: List[str] = Field(
        description="Reasons for the match (e.g., ['visual', 'audio'])"
    )
    image: Optional[str] = Field(
        description="Base64-encoded thumbnail for the media item"
    )


class VisualSearchResponse(BaseModel):
    results: List[VisualResult]


class AudioSearchResponse(BaseModel):
    results: List[AudioResult]


class UnifiedSearchResponse(BaseModel):
    results: List[UnifiedResult]


# --- API Server Class ---


class APIServer:
    """FastAPI server cho tìm kiếm đa phương tiện."""

    def __init__(self):
        """Khởi tạo server, tải model và các file index."""
        # Load cấu hình từ config.py
        self.visual_index_file = config.VISUAL_INDEX_FILE
        self.mapping_file = config.MEDIA_DATA_MAPPING_FILE
        self.transcript_dir = config.TRANSCRIPT_DIR

        print("Initializing image processor...")
        self.image_processor = ImageProcessor(config.CLIP_MODEL_ID)

        print("Loading FAISS index and media data mapping...")
        self.index = faiss.read_index(self.visual_index_file)
        with open(self.mapping_file, "r", encoding="utf-8") as f:
            self.index_to_media_data = json.load(f)
        print(f"✅ Loaded index with {self.index.ntotal} vectors.")

        # Tối ưu: Tạo mapping để tra cứu thumbnail nhanh (O(1) lookup)
        print("Creating media to thumbnail mapping for fast lookups...")
        self.media_id_to_thumbnail_path = self._create_thumbnail_mapping()
        print(
            f"✅ Created thumbnail mapping for {len(self.media_id_to_thumbnail_path)} media items."
        )

        print("Loading audio transcripts...")
        self.transcripts = self._load_transcripts()
        print(f"✅ Loaded {len(self.transcripts)} transcripts.")

        self.app = self._create_app()

    def _create_thumbnail_mapping(self) -> Dict[str, str]:
        """
        Tạo dictionary map từ media_id sang đường dẫn ảnh thumbnail.
        Điều này giải quyết vấn đề hiệu năng được ghi trong todo.md.
        Với video, nó sẽ lấy frame đầu tiên được index.
        Với ảnh, nó sẽ lấy chính đường dẫn ảnh đó.
        """
        mapping = {}
        # Sắp xếp theo timestamp để đảm bảo lấy frame sớm nhất cho video
        sorted_media_data = sorted(
            self.index_to_media_data.values(), key=lambda x: x.get("timestamp") or 0
        )
        for data in sorted_media_data:
            media_id = data["media_id"]
            if media_id not in mapping:
                mapping[media_id] = data["path"]
        return mapping

    def _load_transcripts(self) -> Dict[str, List[Dict]]:
        """Tải tất cả các file transcript từ thư mục output."""
        transcripts = {}
        if not os.path.exists(self.transcript_dir):
            print(f"⚠️ Transcript directory not found: {self.transcript_dir}")
            return transcripts

        for filename in os.listdir(self.transcript_dir):
            if filename.endswith(".json"):
                # Tên file transcript (bỏ .json) phải khớp với tên file media (bỏ .mp4, .jpg...)
                base_name = os.path.splitext(filename)[0]

                # Tìm media_id đầy đủ (ví dụ: my_video.mp4) từ key của dict thumbnail
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
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {filename}: {e}")
                    except Exception as e:
                        print(f"Error loading transcript {filename}: {e}")
        return transcripts

    def _create_app(self) -> FastAPI:
        """Tạo và cấu hình ứng dụng FastAPI."""
        app = FastAPI(
            title="AI Multimodal Search API",
            description="""
            🔍 **AI-powered multimodal search API**
            
            Search through images and videos using natural language descriptions.
            Powered by OpenAI's CLIP, Whisper, and FAISS.
            
            **Features:**
            - 🖼️ Visual search using text descriptions (for images and video frames)
            - 🗣️ Audio search in video transcripts
            - 🚀 Unified search combining visual and audio modalities
            """,
            version="2.0.0",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._add_routes(app)
        return app

    def _add_routes(self, app: FastAPI):
        """Thêm các route vào ứng dụng FastAPI."""
        app.get("/")(self.root)
        app.post("/search_visual", response_model=VisualSearchResponse)(
            self.search_visual
        )
        app.post("/search_audio", response_model=AudioSearchResponse)(self.search_audio)
        app.post("/unified_search", response_model=UnifiedSearchResponse)(
            self.unified_search
        )

    # --- API Endpoint Logic ---

    def root(self):
        """Kiểm tra health và thông tin cơ bản của API."""
        return {
            "message": "AI Multimodal Search API is running!",
            "status": "ok",
            "version": "2.0.0",
            "indexed_visuals": len(self.index_to_media_data),
            "indexed_audios": len(self.transcripts),
            "model_info": self.image_processor.get_model_info(),
        }

    def search_visual(self, query: SearchQuery) -> Dict:
        """Tìm kiếm hình ảnh/frame video dựa trên mô tả văn bản."""
        print(f"🔍 Visual search for: '{query.text}' (top_k={query.top_k})")
        try:
            query_vector = self.image_processor.text_to_embedding(query.text)
            distances, indices = self.index.search(query_vector, query.top_k)

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

            return {"results": results}
        except Exception as e:
            print(f"❌ Error in visual search: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def search_audio(self, query: SearchQuery) -> Dict:
        """Tìm kiếm trong transcript (triển khai đơn giản)."""
        print(f"🗣️ Audio search for: '{query.text}' (top_k={query.top_k})")
        # Vấn đề: Tìm kiếm tuần tự, chậm với dữ liệu lớn.
        # Gợi ý: Dùng Elasticsearch/OpenSearch để cải thiện.
        found_matches = []
        query_lower = query.text.lower()

        for media_id, segments in self.transcripts.items():
            for segment in segments:
                if query_lower in segment["text"].lower():
                    found_matches.append(
                        AudioResult(
                            id=media_id,
                            score=1.0,  # Điểm số đơn giản
                            match=segment["text"],
                            start=segment["start"],
                            end=segment["end"],
                        )
                    )

        return {"results": found_matches[: query.top_k]}

    def unified_search(self, query: SearchQuery) -> Dict:
        """Thực hiện tìm kiếm hợp nhất trên cả hình ảnh và âm thanh."""
        print(f"🚀 Performing unified search for: '{query.text}'")

        # 1. Tìm kiếm trên từng modality (lấy nhiều hơn top_k để fusion)
        visual_results_data = self.search_visual(SearchQuery(text=query.text, top_k=50))
        audio_results_data = self.search_audio(SearchQuery(text=query.text, top_k=50))

        # 2. Hợp nhất kết quả bằng Reciprocal Rank Fusion
        fused_scores = self._fuse_results_rrf(
            visual_results_data["results"], audio_results_data["results"]
        )

        # 3. Sắp xếp và chuẩn bị kết quả cuối cùng
        final_results = []
        sorted_media = sorted(
            fused_scores.items(), key=lambda item: item[1]["score"], reverse=True
        )

        for media_id, data in sorted_media[: query.top_k]:
            # Lấy thumbnail hiệu quả bằng O(1) lookup
            thumbnail_path = self.media_id_to_thumbnail_path.get(media_id)
            image_base64 = (
                ImageProcessor.image_to_base64(thumbnail_path)
                if thumbnail_path
                else None
            )

            final_results.append(
                UnifiedResult(
                    id=media_id,
                    score=data["score"],
                    reason=sorted(data["reason"]),  # Sắp xếp để output nhất quán
                    image=image_base64,
                )
            )

        return {"results": final_results}

    def _fuse_results_rrf(
        self, visual_results: List[Dict], audio_results: List[Dict], k: int = 60
    ) -> Dict:
        """Hợp nhất kết quả bằng Reciprocal Rank Fusion (RRF)."""
        fused_scores = {}

        # Xử lý kết quả hình ảnh
        for rank, res in enumerate(visual_results):
            media_id = res["id"]
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": []}

            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[media_id]["score"] += rrf_score
            if "visual" not in fused_scores[media_id]["reason"]:
                fused_scores[media_id]["reason"].append("visual")

        # Xử lý kết quả âm thanh
        for rank, res in enumerate(audio_results):
            media_id = res["id"]
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": []}

            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[media_id]["score"] += rrf_score
            if "audio" not in fused_scores[media_id]["reason"]:
                fused_scores[media_id]["reason"].append("audio")

        return fused_scores

    def run(self, host: str, port: int):
        """Chạy API server."""
        import uvicorn

        print("🚀 Starting API server...")
        print(f"Server running at: http://{host}:{port}")
        print(f"Access API docs at: http://{host}:{port}/docs")
        uvicorn.run(self.app, host=host, port=port)
