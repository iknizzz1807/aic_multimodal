import os
import torch

# Biến global này sẽ được khởi tạo MỘT LẦN cho mỗi worker
worker_data = {}


def init_worker():
    """Hàm khởi tạo cho mỗi worker process."""
    worker_pid = os.getpid()
    print(f"[Worker PID: {worker_pid}] Initializing...")

    # Import các module cần thiết bên trong worker
    from core.vision import VisionProcessor
    from core.audio import AudioProcessor
    from core.video import VideoProcessor
    from database.milvus_connector import MilvusConnector
    from database.es_connector import ElasticsearchConnector
    import config

    # Tải model và kết nối DB MỘT LẦN
    # Mỗi worker sẽ có kết nối riêng để tránh xung đột
    milvus_conn = MilvusConnector()
    milvus_conn.setup_visual_collection()  # Quan trọng: Lấy hoặc tạo collection

    worker_data["milvus_conn"] = milvus_conn
    worker_data["es_conn"] = ElasticsearchConnector()
    worker_data["vision_processor"] = VisionProcessor(model_id=config.CLIP_MODEL_ID)
    worker_data["audio_processor"] = AudioProcessor(model_id=config.WHISPER_MODEL_ID)
    worker_data["video_processor"] = VideoProcessor(
        frame_interval=config.VIDEO_FRAME_EXTRACTION_INTERVAL
    )
    worker_data["config"] = config

    print(f"✅ [Worker PID: {worker_pid}] Initialization complete.")


def _process_visual(file_path: str, media_id: str):
    """Xử lý phần hình ảnh của một tệp (ảnh hoặc video)."""
    vp = worker_data["vision_processor"]
    vidp = worker_data["video_processor"]
    milvus_conn = worker_data["milvus_conn"]
    config = worker_data["config"]
    worker_pid = os.getpid()

    visual_batch = []
    try:
        if file_path.lower().endswith(config.IMAGE_EXTENSIONS):
            embedding = vp.image_to_embedding(file_path)
            if embedding is not None:
                visual_batch.append(
                    {"media_id": media_id, "timestamp": 0.0, "vector": embedding[0]}
                )
        elif file_path.lower().endswith(config.VIDEO_EXTENSIONS):
            for frame in vidp.extract_frames_in_memory(file_path):
                embedding = vp.numpy_array_to_embedding(frame["frame_rgb"])
                if embedding is not None:
                    visual_batch.append(
                        {
                            "media_id": media_id,
                            "timestamp": frame["timestamp"],
                            "vector": embedding[0],
                        }
                    )

        if visual_batch:
            milvus_conn.insert(visual_batch)
            print(
                f"🖼️  [Worker PID: {worker_pid}] Inserted {len(visual_batch)} visual vectors for {media_id}"
            )

    except Exception as e:
        print(
            f"❌ [Worker PID: {worker_pid}] Visual processing error for {media_id}: {e}"
        )
        import traceback

        traceback.print_exc()


def _process_audio(file_path: str, media_id: str):
    """Xử lý phần âm thanh của một tệp (video hoặc audio)."""
    ap = worker_data["audio_processor"]
    es_conn = worker_data["es_conn"]
    worker_pid = os.getpid()

    try:
        # Kiểm tra xem tệp có luồng âm thanh không trước khi xử lý
        if not ap.has_audio_stream(file_path):
            print(
                f"📝 [Worker PID: {worker_pid}] No audio stream detected in {media_id}"
            )
            return

        result = ap.transcribe(file_path)
        if result and result.get("segments"):
            docs = [
                {
                    "media_id": media_id,
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                }
                for s in result["segments"]
            ]
            es_conn.bulk_insert(docs)
            print(
                f"🗣️  [Worker PID: {worker_pid}] Inserted {len(docs)} transcript segments for {media_id}"
            )
        else:
            print(f"📝 [Worker PID: {worker_pid}] No speech detected in {media_id}")
    except Exception as e:
        print(
            f"❌ [Worker PID: {worker_pid}] Audio processing error for {media_id}: {e}"
        )


def process_single_media_file(file_path: str):
    """Hàm chính để xử lý một tệp media, điều hướng đến hàm xử lý phù hợp."""
    media_id = os.path.basename(file_path)
    worker_pid = os.getpid()
    config = worker_data["config"]

    print(f"▶️  [Worker PID: {worker_pid}] Processing: {media_id}")

    file_ext = os.path.splitext(file_path)[1].lower()

    # --- Phân loại và xử lý ---
    if file_ext in config.IMAGE_EXTENSIONS:
        _process_visual(file_path, media_id)

    elif file_ext in config.VIDEO_EXTENSIONS:
        _process_visual(file_path, media_id)
        _process_audio(file_path, media_id)

    elif file_ext in config.MEDIA_EXTENSIONS:  # Các tệp audio khác
        _process_audio(file_path, media_id)

    print(f"✅ [Worker PID: {worker_pid}] Finished processing {media_id}\n")
