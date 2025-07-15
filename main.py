import os
import sys
from src.api import APIServer
from src import config


def main():
    """
    Điểm khởi đầu chính của ứng dụng.
    Kiểm tra các file index và khởi động API server.
    """
    print("--- AI Multimodal Search Server ---")

    # Kiểm tra xem các file index cần thiết có tồn tại không
    required_files = [config.VISUAL_INDEX_FILE, config.MEDIA_DATA_MAPPING_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("\n" + "=" * 60)
        print("❌ CRITICAL ERROR: Index files not found!")
        print("The following required files are missing:")
        for f in missing_files:
            print(f"  - {f}")

        print("\n👉 Please build the index first by running:")
        print("   python build_index.py")
        print("=" * 60)
        sys.exit(1)  # Thoát chương trình

    print("\n✅ All required index files found. Initializing server...")

    # Khởi tạo và chạy server
    try:
        server = APIServer()
        server.run(host=config.API_HOST, port=config.API_PORT)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during server startup: {e}")
        print("Please check your configuration and the error logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
