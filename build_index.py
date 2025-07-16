import os
from visual_indexer import VisualIndexer
from audio_extractor import AudioIndexer
import config


def main():
    """
    Chạy toàn bộ quá trình sau:
    1. Bóc tách audio từ video/audio files.
    2. Index hình ảnh tĩnh và các frame trích xuất từ video.
    """
    print("=" * 60)
    print("🚀 STARTING MULTIMODAL INDEXING PROCESS 🚀")
    print("=" * 60)

    # Đảm bảo các thư mục cần thiết tồn tại
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    os.makedirs(config.TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(config.FRAME_DIR, exist_ok=True)

    # --- Bước 1: Audio Indexing ---
    print("\n" + "-" * 20 + " STEP 1: AUDIO TRANSCRIPTION " + "-" * 20)
    audio_indexer = AudioIndexer(model_id=config.WHISPER_MODEL_ID)
    audio_indexer.transcribe_media_from_directory(
        media_directory=config.DATA_DIR, output_directory=config.TRANSCRIPT_DIR
    )
    print("-" * 60)
    print("✅ Audio transcription completed.")

    # --- Bước 2: Visual Indexing ---
    print(
        "\n" + "-" * 20 + " STEP 2: VISUAL INDEXING (Images & Video Frames) " + "-" * 20
    )
    visual_indexer = VisualIndexer(model_id=config.CLIP_MODEL_ID)
    visual_indexer.create_index_from_directory(
        data_directory=config.DATA_DIR,
        output_index_file=config.VISUAL_INDEX_FILE,
        output_mapping_file=config.MEDIA_DATA_MAPPING_FILE,
        index_type=config.FAISS_INDEX_TYPE,
    )
    print("-" * 60)
    print("✅ Visual indexing completed.")

    print("\n" + "=" * 60)
    print("🎉🎉🎉 ALL INDEXING PROCESSES COMPLETED! 🎉🎉🎉")
    print("You can now start the API server by running:")
    print("   python -m src.api")
    print("=" * 60)


if __name__ == "__main__":
    main()
