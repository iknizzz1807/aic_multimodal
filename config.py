import os

# --- Thư mục ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Cấu hình Model ---
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
WHISPER_MODEL_ID = "base"
CLAP_MODEL_ID = ""  # Not used anymore, architecture is defined in AudioEventProcessor
CLIP_EMBEDDING_DIM = 512  # Kích thước vector của model 'openai/clip-vit-base-patch32'

# CLAP embedding dimension for HTSAT-base model
CLAP_EMBEDDING_DIM = 512  # HTSAT-base model embedding dimension

# Thêm model cho bước Re-rank (quan trọng cho độ chính xác)
CROSS_ENCODER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Số lượng processing worker
PROCESSING_WORKERS = 2  # <-- Thay đổi tuỳ theo độ trâu của máy =))

# --- Cấu hình DB ---
# Milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
VISUAL_COLLECTION_NAME = "visual_media_v1"
AUDIO_EVENT_COLLECTION_NAME = "audio_events_v1"

# Elasticsearch
ES_HOST = "localhost"
ES_PORT = 9200
TRANSCRIPT_INDEX_NAME = "media_transcripts_v1"


# Chọn phương thức trích xuất keyframe: 'interval' (nhanh) hoặc 'scenedetect' (chậm, chất lượng cao)
KEYFRAME_EXTRACTION_METHOD = "interval"

# --- Cấu hình Processing ---
VIDEO_FRAME_EXTRACTION_INTERVAL = 1.0  # Dùng cho phương thức 'interval'
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
MEDIA_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".m4a")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

# --- Cấu hình API Server ---
API_HOST = "127.0.0.1"
API_PORT = 8000
