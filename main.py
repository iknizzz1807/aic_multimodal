import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.api_router import router as api_router
import config


def create_app() -> FastAPI:
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
        version="2.2.0-refactored",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Gắn router vào ứng dụng
    app.include_router(api_router)

    return app


app = create_app()


def run_server():
    """Chạy API server."""
    print("🚀 Starting API server...")
    print(f"Server running at: http://{config.API_HOST}:{config.API_PORT}")
    print(f"Access API docs at: http://{config.API_HOST}:{config.API_PORT}/docs")
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
    )


# File `main.py` sẽ gọi hàm này thay vì class
# Hoặc có thể chạy trực tiếp file này
if __name__ == "__main__":
    run_server()
