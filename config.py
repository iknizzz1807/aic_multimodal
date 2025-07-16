import os

# --- Thư mục ---
# Thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục chứa dữ liệu đầu vào (video, ảnh)
DATA_DIR = os.path.join(BASE_DIR, "data")

# Thư mục chứa các file output
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Thư mục chứa index của FAISS và file mapping
INDEX_DIR = os.path.join(OUTPUT_DIR, "index")

# Thư mục chứa các file transcript đã được bóc tách
TRANSCRIPT_DIR = os.path.join(OUTPUT_DIR, "transcripts")

# Thư mục chứa các khung hình trích xuất từ video
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")


# --- Tên File ---
# File index FAISS cho hình ảnh và video frame
VISUAL_INDEX_FILE = os.path.join(INDEX_DIR, "faiss_visual.index")

# File mapping từ index của FAISS tới thông tin media (thay thế cho index_to_path.json)
MEDIA_DATA_MAPPING_FILE = os.path.join(INDEX_DIR, "index_to_media_data.json")


# --- Cấu hình Model ---
# Model CLIP để xử lý hình ảnh và văn bản
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Model Whisper để bóc tách audio
WHISPER_MODEL_ID = "base"  # 'tiny', 'base', 'small', 'medium', 'large'


# --- Cấu hình Indexer ---
# Loại index FAISS ('flat_ip', 'flat_l2')
# 'flat_ip' (Inner Product) tốt cho vector đã chuẩn hóa của CLIP
FAISS_INDEX_TYPE = "flat_ip"

# Khoảng thời gian (giây) giữa các lần trích xuất frame từ video
VIDEO_FRAME_EXTRACTION_INTERVAL = 1.0  # Trích xuất 1 frame mỗi giây

# Các định dạng file ảnh được chấp nhận
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Các định dạng file video/audio được chấp nhận
MEDIA_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".m4a")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


# --- Cấu hình API Server ---
API_HOST = "127.0.0.1"
API_PORT = 8000
