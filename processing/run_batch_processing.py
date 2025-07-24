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


def main():
    print("=" * 60)
    print("🚀 STARTING NEW BATCH PROCESSING PIPELINE 🚀")
    print("=" * 60)

    # 1. Chỉ cần setup DB một lần từ main process
    # Worker sẽ tự kết nối, không cần tạo instance ở đây nữa
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

    # 2. Quét thư mục data
    print("--- STEP 2: Scanning for Media Files ---")
    all_files = [
        os.path.join(config.DATA_DIR, f)
        for f in os.listdir(config.DATA_DIR)
        if f.lower().endswith(config.MEDIA_EXTENSIONS + config.IMAGE_EXTENSIONS)
    ]
    if not all_files:
        print("❌ No media files found. Exiting.")
        return
    print(f"Found {len(all_files)} files to process.\n")

    # 3. Sử dụng multiprocessing.Pool với initializer
    # num_processes = multiprocessing.cpu_count()
    num_processes = 2  # Giới hạn số lượng worker để tránh quá tải GPU
    print(f"--- STEP 3: Starting Parallel Processing with {num_processes} workers ---")
    start_time = time.time()

    # Sử dụng initializer để mỗi worker chỉ load model một lần!
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker) as pool:
        pool.map(process_single_media_file, all_files)

    end_time = time.time()
    print("\n--- STEP 4: Finalizing ---")

    # Flush Milvus collection để đảm bảo tất cả dữ liệu được ghi
    try:
        print("Flushing Milvus collection to ensure data persistence...")
        utility.flush([config.VISUAL_COLLECTION_NAME])
        print("✅ Milvus flush complete.")
    except Exception as e:
        print(f"⚠️ Warning: Could not flush Milvus collection: {e}")

    print("\n" + "=" * 60)
    print("🎉🎉🎉 ALL BATCH PROCESSING COMPLETE! 🎉🎉🎉")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")
    print("Your data is now indexed in Milvus and Elasticsearch.")
    print("=" * 60)


if __name__ == "__main__":
    # Đặt start_method là 'spawn' để tương thích tốt hơn với CUDA trên một số hệ thống
    multiprocessing.set_start_method("spawn", force=True)
    main()
