import os
import multiprocessing
import time
from pymilvus import utility

import config

# Bỏ các import không cần thiết ở đây, chúng sẽ được xử lý trong worker
from processing.worker import (
    process_single_media_file,
    init_worker,
)

# Đường dẫn đến tệp log, đặt cùng cấp với file script này
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "processed_files.log")


def get_processed_files():
    """Đọc danh sách các tệp đã xử lý từ tệp log."""
    if not os.path.exists(LOG_FILE_PATH):
        return set()
    try:
        with open(LOG_FILE_PATH, "r") as f:
            # Dùng .strip() để loại bỏ các khoảng trắng hoặc dòng trống
            return set(line.strip() for line in f if line.strip())
    except IOError as e:
        print(f"⚠️  Warning: Could not read log file {LOG_FILE_PATH}: {e}")
        return set()


def main():
    print("=" * 60)
    print("🚀 STARTING NEW BATCH PROCESSING PIPELINE (Resumable) 🚀")
    print("=" * 60)

    # 1. Chỉ cần setup DB một lần từ main process
    print("--- STEP 1: Setting up Databases ---")
    try:
        from database.milvus_connector import MilvusConnector
        from database.es_connector import ElasticsearchConnector

        MilvusConnector().setup_visual_collection()
        ElasticsearchConnector().setup_transcript_index()
        print("✅ Databases are ready.\n")
    except Exception as e:
        print(f"❌ Critical error during DB setup: {e}")
        return

    # 2. Quét thư mục data và lọc ra các tệp chưa xử lý
    print("--- STEP 2: Scanning for Media Files ---")
    processed_files = get_processed_files()
    if processed_files:
        print(f"Found {len(processed_files)} previously processed files to skip.")

    all_files_in_dir = [
        f
        for f in os.listdir(config.DATA_DIR)
        if f.lower().endswith(config.MEDIA_EXTENSIONS + config.IMAGE_EXTENSIONS)
    ]

    files_to_process_paths = [
        os.path.join(config.DATA_DIR, f)
        for f in all_files_in_dir
        if f not in processed_files  # Lọc bỏ các tệp đã có trong log
    ]

    if not files_to_process_paths:
        print("✅ No new media files to process. Exiting.")
        return
    print(f"Found {len(files_to_process_paths)} new files to process.\n")

    # 3. Sử dụng multiprocessing.Pool với initializer và Lock
    num_processes = config.PROCESSING_WORKERS
    print(f"--- STEP 3: Starting Parallel Processing with {num_processes} workers ---")
    start_time = time.time()

    # Tạo một Lock để ghi file log an toàn từ nhiều process
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    # Sử dụng initializer để mỗi worker chỉ load model một lần
    # và truyền Lock vào cho mỗi worker process
    init_args = (lock, LOG_FILE_PATH)
    with multiprocessing.Pool(
        processes=num_processes, initializer=init_worker, initargs=init_args
    ) as pool:
        pool.map(process_single_media_file, files_to_process_paths)

    end_time = time.time()
    print("\n--- STEP 4: Finalizing ---")

    # Flush Milvus collection để đảm bảo tất cả dữ liệu được ghi
    try:
        print("Flushing Milvus collection to ensure data persistence...")
        # (MilvusConnector được thiết kế để xử lý việc này một cách an toàn)
        milvus_conn = MilvusConnector()
        collection = milvus_conn.get_collection()

        if collection:
            collection.flush()
            print("✅ Milvus flush complete.")
        else:
            print("⚠️ Warning: Could not get Milvus collection to flush.")

    except Exception as e:
        # Giữ lại để bắt các lỗi khác có thể xảy ra
        print(f"⚠️ Warning: Could not flush Milvus collection: {e}")

    print("\n" + "=" * 60)
    print("🎉🎉🎉 ALL BATCH PROCESSING COMPLETE! 🎉🎉🎉")
    print(f"Total processing time for this run: {end_time - start_time:.2f} seconds.")
    print("Your data is now indexed in Milvus and Elasticsearch.")
    print("=" * 60)
