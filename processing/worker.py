import os
import torch
import traceback

# Biến global này sẽ được khởi tạo MỘT LẦN cho mỗi worker
worker_data = {}


def init_worker(lock, log_file_path):
    """Hàm khởi tạo cho mỗi worker process."""
    worker_pid = os.getpid()
    print(f"[Worker PID: {worker_pid}] Initializing...")

    # Import các module cần thiết bên trong worker
    from core.vision import VisionProcessor
    from core.audio import AudioProcessor, AudioEventProcessor
    from core.video import VideoProcessor
    from database.milvus_connector import MilvusConnector
    from database.es_connector import ElasticsearchConnector
    import config

    # Tải model và kết nối DB MỘT LẦN
    milvus_conn = MilvusConnector()
    milvus_conn.setup_visual_collection()
    milvus_conn.setup_audio_event_collection()  # Thêm setup cho collection mới

    worker_data["milvus_conn"] = milvus_conn
    worker_data["es_conn"] = ElasticsearchConnector()
    worker_data["vision_processor"] = VisionProcessor(model_id=config.CLIP_MODEL_ID)
    worker_data["audio_processor"] = AudioProcessor(model_id=config.WHISPER_MODEL_ID)
    worker_data["audio_event_processor"] = AudioEventProcessor(
        model_id=config.CLAP_MODEL_ID
    )  # Thêm processor mới
    worker_data["video_processor"] = VideoProcessor(
        frame_interval=config.VIDEO_FRAME_EXTRACTION_INTERVAL,
        extraction_method=config.KEYFRAME_EXTRACTION_METHOD,
    )
    worker_data["config"] = config
    # Lưu lock và đường dẫn tệp log để các hàm khác có thể sử dụng
    worker_data["lock"] = lock
    worker_data["log_file_path"] = log_file_path

    print(f"✅ [Worker PID: {worker_pid}] Initialization complete.")


def _process_visual(file_path: str, media_id: str):
    """Xử lý phần hình ảnh của một tệp (ảnh hoặc video)."""
    vp = worker_data["vision_processor"]
    vidp = worker_data["video_processor"]
    milvus_conn = worker_data["milvus_conn"]
    config = worker_data["config"]
    worker_pid = os.getpid()

    visual_batch = []
    # Đã có try-except ở hàm cha (process_single_media_file)
    if file_path.lower().endswith(config.IMAGE_EXTENSIONS):
        embedding = vp.image_to_embedding(file_path)
        if embedding is not None:
            visual_batch.append(
                {"media_id": media_id, "timestamp": 0.0, "vector": embedding[0]}
            )
    elif file_path.lower().endswith(config.VIDEO_EXTENSIONS):
        # Sử dụng phương thức trích xuất keyframe thông minh mới
        for frame in vidp.extract_keyframes(file_path):
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


def _process_audio(file_path: str, media_id: str):
    """Xử lý phần âm thanh của một tệp (video hoặc audio)."""
    ap = worker_data["audio_processor"]
    es_conn = worker_data["es_conn"]
    worker_pid = os.getpid()

    # Đã có try-except ở hàm cha (process_single_media_file)
    if not ap.has_audio_stream(file_path):
        print(f"📝 [Worker PID: {worker_pid}] No audio stream detected in {media_id}")
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


def _process_audio_events(file_path: str, media_id: str):
    """Xử lý sự kiện âm thanh của một tệp bằng CLAP."""
    aep = worker_data["audio_event_processor"]
    milvus_conn = worker_data["milvus_conn"]
    worker_pid = os.getpid()
    ap = worker_data["audio_processor"]  # Dùng lại để kiểm tra audio stream

    if not ap.has_audio_stream(file_path):
        return  # Bỏ qua nếu không có audio

    audio_event_batch = []
    for event in aep.extract_event_embeddings(file_path):
        event["media_id"] = media_id
        audio_event_batch.append(event)

    if audio_event_batch:
        milvus_conn.insert_audio_events(audio_event_batch)
        print(
            f"🔊 [Worker PID: {worker_pid}] Inserted {len(audio_event_batch)} audio event vectors for {media_id}"
        )


def process_single_media_file(file_path: str):
    """Hàm chính để xử lý một tệp media, điều hướng và ghi log khi thành công."""
    media_id = os.path.basename(file_path)
    worker_pid = os.getpid()
    config = worker_data["config"]
    lock = worker_data["lock"]
    log_file_path = worker_data["log_file_path"]

    print(f"▶️  [Worker PID: {worker_pid}] Processing: {media_id}")

    try:
        file_ext = os.path.splitext(file_path)[1].lower()

        # --- Phân loại và xử lý ---
        if file_ext in config.IMAGE_EXTENSIONS:
            _process_visual(file_path, media_id)
        elif file_ext in config.VIDEO_EXTENSIONS:
            # Thực hiện cả ba tác vụ cho video
            _process_visual(file_path, media_id)
            _process_audio(file_path, media_id)
            _process_audio_events(file_path, media_id)
        elif file_ext in config.MEDIA_EXTENSIONS:  # Chỉ các tệp audio
            # Thực hiện cả hai tác vụ audio
            _process_audio(file_path, media_id)
            _process_audio_events(file_path, media_id)

        # Ghi vào file log KHI VÀ CHỈ KHI tất cả xử lý cho file này thành công
        # Sử dụng lock để đảm bảo không có 2 worker cùng ghi file một lúc (thread-safe)
        with lock:
            with open(log_file_path, "a") as f:
                f.write(f"{media_id}\n")

        print(f"✅ [Worker PID: {worker_pid}] Finished and logged {media_id}\n")

    except Exception as e:
        # Nếu có bất kỳ lỗi nào xảy ra, ghi nhận nó và KHÔNG ghi vào log thành công.
        # Tệp này sẽ được thử lại trong lần chạy tiếp theo.
        print(
            f"🔥🔥🔥 [Worker PID: {worker_pid}] FAILED to process {media_id}. It will be retried on next run."
        )
        print(f"Error details: {e}")
        traceback.print_exc()
