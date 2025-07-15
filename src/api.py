import os
import json
from typing import List, Optional, Dict
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.image_processor import ImageProcessor

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class APIServer:
    """FastAPI server for multimodal search"""

    def __init__(
        self,
        index_file: str = "output/faiss_visual.index",
        mapping_file: str = "output/index_to_path.json",
        transcript_dir: str = "output/transcripts",
        model_id: str = "openai/clip-vit-base-patch32",
    ):
        """
        Initialize API server

        Args:
            index_file: Path to FAISS index file
            mapping_file: Path to index mapping JSON file
            transcript_dir: Directory containing transcript files
            model_id: CLIP model identifier
        """
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.transcript_dir = transcript_dir  # Lưu lại đường dẫn

        # Initialize processors
        print("Initializing image processor...")
        self.image_processor = ImageProcessor(model_id)

        # Load FAISS index and mapping
        print("Loading FAISS index and mapping...")
        self.index = faiss.read_index(index_file)
        with open(mapping_file, "r") as f:
            self.index_to_path = json.load(f)
        print(f"✅ Loaded index with {self.index.ntotal} vectors")

        # Tải tất cả các transcript vào bộ nhớ
        print("Loading audio transcripts...")
        self.transcripts = self._load_transcripts()
        print(f"✅ Loaded {len(self.transcripts)} transcripts.")

        # Initialize FastAPI app
        self.app = self._create_app()

    # Hàm để tải transcript
    def _load_transcripts(self) -> Dict[str, List[Dict]]:
        """Load all JSON transcripts from the output directory."""
        transcripts = {}
        if not os.path.exists(self.transcript_dir):
            print(f"⚠️ Transcript directory not found: {self.transcript_dir}")
            return transcripts

        for filename in os.listdir(self.transcript_dir):
            if filename.endswith(".json"):
                # Lấy tên file media gốc (vd: video999.json -> video999)
                media_name = os.path.splitext(filename)[0]

                # Tìm đường dẫn file media đầy đủ trong mapping
                # Điều này giả định file media (mp4) cũng có trong data
                # và đã được index (ít nhất 1 frame) trong faiss_visual.index
                original_path = None
                for path in self.index_to_path.values():
                    if os.path.splitext(os.path.basename(path))[0] == media_name:
                        original_path = os.path.basename(path)
                        break

                if original_path:
                    try:
                        with open(
                            os.path.join(self.transcript_dir, filename),
                            "r",
                            encoding="utf-8",
                        ) as f:
                            transcripts[original_path] = json.load(f)
                    except Exception as e:
                        print(f"Error loading transcript {filename}: {e}")

        return transcripts

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
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
            version="1.1.0",  # CẬP NHẬT: Tăng phiên bản
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
        """Add API routes to FastAPI app"""

        # Pydantic models
        class SearchQuery(BaseModel):
            text: str = Field(
                ...,
                description="Natural language description",
                example="red car on street",
            )
            top_k: int = Field(
                default=10, ge=1, le=50, description="Number of results (1-50)"
            )

        class VisualResult(BaseModel):
            id: str = Field(description="Image/Video identifier")
            score: float = Field(description="Similarity score")
            timestamp: Optional[float] = Field(
                description="Timestamp in video if applicable"
            )
            image: Optional[str] = Field(description="Base64-encoded image data")

        class AudioResult(BaseModel):
            id: str = Field(description="Video/Audio identifier")
            score: float = Field(description="Match score")
            match: str = Field(description="The matched text segment")
            start: float = Field(description="Start time of the segment")
            end: float = Field(description="End time of the segment")

        class UnifiedResult(BaseModel):
            id: str = Field(description="Video/Media identifier")
            score: float = Field(description="Final fused score")
            reason: List[str] = Field(
                description="Reasons for the match (visual, audio)"
            )
            image: Optional[str] = Field(description="Base64-encoded thumbnail")

        class VisualSearchResponse(BaseModel):
            results: List[VisualResult]

        class AudioSearchResponse(BaseModel):
            results: List[AudioResult]

        class UnifiedSearchResponse(BaseModel):
            results: List[UnifiedResult]

        # Routes
        @app.get("/")
        def root():
            """API health check and basic info"""
            return {
                "message": "AI Multimodal Search API is running!",
                "status": "ok",
                "version": "1.1.0",
                "indexed_visuals": len(self.index_to_path),
                "indexed_audios": len(self.transcripts),
                "model_info": self.image_processor.get_model_info(),
            }

        # ... (endpoint /stats và /image/{filename} giữ nguyên) ...

        @app.post("/search_visual", response_model=VisualSearchResponse)
        def search_visual(query: SearchQuery):
            """Search for images/video frames using natural language"""
            return self._search_visual(query.text, query.top_k)

        # Endpoint tìm kiếm audio
        @app.post("/search_audio", response_model=AudioSearchResponse)
        def search_audio(query: SearchQuery):
            """Search for spoken text in video transcripts"""
            return self._search_audio(query.text, query.top_k)

        # Endpoint tìm kiếm hợp nhất
        @app.post("/unified_search", response_model=UnifiedSearchResponse)
        def unified_search(query: SearchQuery):
            """Perform a unified search across visual and audio modalities"""
            print(f"🚀 Performing unified search for: '{query.text}'")

            # 1. Thực hiện tìm kiếm trên từng modality (lấy nhiều hơn top_k để fusion)
            visual_results_data = self._search_visual(query.text, top_k=50)
            audio_results_data = self._search_audio(query.text, top_k=50)

            # 2. Hợp nhất kết quả bằng RRF
            fused_scores = self._fuse_results_rrf(
                visual_results_data["results"], audio_results_data["results"]
            )

            # 3. Chuẩn bị kết quả cuối cùng
            final_results = []
            sorted_media = sorted(
                fused_scores.items(), key=lambda item: item[1]["score"], reverse=True
            )

            for media_id, data in sorted_media[: query.top_k]:
                # Lấy ảnh thumbnail cho kết quả
                image_base64 = None
                for index, path in self.index_to_path.items():
                    if os.path.basename(path) == media_id:
                        image_base64 = ImageProcessor.image_to_base64(path)
                        break

                final_results.append(
                    UnifiedResult(
                        id=media_id,
                        score=data["score"],
                        reason=data["reason"],
                        image=image_base64,
                    )
                )

            return {"results": final_results}

    # Đổi tên _search_images thành _search_visual để nhất quán
    def _search_visual(self, text: str, top_k: int = 10) -> dict:
        """Internal method to search for images/frames"""
        print(f"🔍 Visual search for: '{text}' (top_k={top_k})")
        try:
            query_vector = self.image_processor.text_to_embedding(text)
            distances, indices = self.index.search(query_vector, top_k)

            results = []
            for i in range(len(indices[0])):
                result_index = str(indices[0][i])
                # mapping của bạn lưu key là string {"0": "path"}
                result_path = self.index_to_path.get(result_index)

                if result_path:
                    filename = os.path.basename(result_path)
                    score = float(distances[0][i])

                    # Giả định mapping lưu thông tin timestamp nếu là video
                    # Trong low_implementation.md, bạn sẽ cần cập nhật indexer_visual.py
                    # để lưu `{"video_path": path, "timestamp_sec": sec}`
                    # Ở đây ta tạm bỏ qua timestamp cho đơn giản

                    result_data = {"id": filename, "score": score, "timestamp": None}
                    image_base64 = ImageProcessor.image_to_base64(result_path)
                    if image_base64:
                        result_data["image"] = image_base64
                    results.append(result_data)

            return {"results": results}
        except Exception as e:
            print(f"❌ Error in visual search: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    #  Hàm tìm kiếm audio
    def _search_audio(self, text: str, top_k: int = 10) -> dict:
        """Internal method to search through transcripts (naive implementation)."""
        print(f"🗣️ Audio search for: '{text}' (top_k={top_k})")

        # Đây là cách tìm kiếm tuần tự, chậm với dữ liệu lớn.
        # Giải pháp tốt hơn là dùng Elasticsearch/OpenSearch.

        found_matches = []
        query_lower = text.lower()

        for media_id, segments in self.transcripts.items():
            for segment in segments:
                if query_lower in segment["text"].lower():
                    # Gán điểm đơn giản, mỗi lần tìm thấy là 1 điểm
                    found_matches.append(
                        {
                            "id": media_id,
                            "score": 1.0,
                            "match": segment["text"],
                            "start": segment["start"],
                            "end": segment["end"],
                        }
                    )

        # Vì cách tính điểm đơn giản, ta chỉ trả về top_k kết quả đầu tiên tìm thấy
        return {"results": found_matches[:top_k]}

    # Hàm hợp nhất kết quả
    def _fuse_results_rrf(
        self, visual_results: List, audio_results: List, k: int = 60
    ) -> Dict:
        """Fuse results from different modalities using Reciprocal Rank Fusion (RRF)."""
        fused_scores = {}

        # Process visual results
        for rank, res in enumerate(visual_results):
            media_id = res["id"]
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": []}

            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[media_id]["score"] += rrf_score
            if "visual" not in fused_scores[media_id]["reason"]:
                fused_scores[media_id]["reason"].append("visual")

        # Process audio results
        for rank, res in enumerate(audio_results):
            media_id = res["id"]
            if media_id not in fused_scores:
                fused_scores[media_id] = {"score": 0.0, "reason": []}

            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[media_id]["score"] += rrf_score
            if "audio" not in fused_scores[media_id]["reason"]:
                fused_scores[media_id]["reason"].append("audio")

        return fused_scores

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the API server"""
        import uvicorn

        print("🚀 Starting API server...")
        print(f"Server running at: http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Main execution
if __name__ == "__main__":
    # Configuration
    INDEX_FILE = "output/faiss_visual.index"
    MAPPING_FILE = "output/index_to_path.json"
    TRANSCRIPT_DIR = "output/transcripts"
    MODEL_ID = "openai/clip-vit-base-patch32"

    server = APIServer(
        index_file=INDEX_FILE,
        mapping_file=MAPPING_FILE,
        transcript_dir=TRANSCRIPT_DIR,
        model_id=MODEL_ID,
    )
    server.run()
