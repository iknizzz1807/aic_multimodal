import os
import json
import whisper
import torch
import subprocess
import json

from src import config


class AudioIndexer:
    def __init__(self, model_id: str = config.WHISPER_MODEL_ID):
        """
        Initialize audio indexer with a Whisper model.

        Args:
            model_id: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                      'base' is a good starting point.
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Whisper model: {model_id}")
        print(f"Device: {self.device}")

        self.model = whisper.load_model(self.model_id, device=self.device)
        print("✅ Whisper model loaded")

    def _has_audio_stream(self, video_path: str) -> bool:
        """
        Kiểm tra xem file media có luồng âm thanh hay không bằng ffprobe.
        Yêu cầu ffprobe (một phần của ffmpeg) phải được cài đặt và có trong PATH.
        """
        try:
            command = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            # Chạy lệnh và lấy output
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            # Nếu có output (vd: 'audio') thì có luồng âm thanh
            return len(result.stdout.strip()) > 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            # CalledProcessError: ffprobe chạy nhưng trả về lỗi (vd: file không có luồng audio)
            # FileNotFoundError: không tìm thấy lệnh ffprobe
            return False

    def transcribe_media_from_directory(
        self,
        media_directory: str = config.DATA_DIR,
        output_directory: str = config.TRANSCRIPT_DIR,
        media_extensions: tuple = config.MEDIA_EXTENSIONS,
    ):
        """
        Bóc tách audio từ tất cả các file media trong thư mục.
        CẬP NHẬT: Xử lý nhẹ nhàng các video không có âm thanh.
        """
        print(f"🔍 Transcribing media from directory: {media_directory}")
        os.makedirs(output_directory, exist_ok=True)

        media_files = [
            os.path.join(media_directory, f)
            for f in os.listdir(media_directory)
            if f.lower().endswith(media_extensions)
        ]

        if not media_files:
            print("🔵 No media files found in directory.")
            return

        print(f"📁 Found {len(media_files)} media files to process.")
        total = len(media_files)
        transcribed_count = 0
        skipped_count = 0

        for i, path in enumerate(media_files):
            print(
                f"   Processing {i+1}/{total}: {os.path.basename(path)}",
                end="",
                flush=True,
            )

            # CẬP NHẬT: Kiểm tra xem file json đã tồn tại chưa để bỏ qua
            base_filename = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(output_directory, f"{base_filename}.json")
            if os.path.exists(output_path):
                print("... ⏩ Already transcribed, skipping.")
                transcribed_count += 1
                continue

            # CẬP NHẬT: Kiểm tra luồng audio
            if path.lower().endswith(
                config.VIDEO_EXTENSIONS
            ) and not self._has_audio_stream(path):
                print("... 🔇 No audio stream found, skipping.")
                skipped_count += 1
                continue

            try:
                # Thực hiện bóc tách
                result = self.model.transcribe(path, fp16=torch.cuda.is_available())

                # Lưu kết quả
                with open(output_path, "w", encoding="utf-8") as f:
                    # Chỉ lưu nếu có text, tránh tạo file rỗng
                    if result["text"].strip():
                        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
                        print(f" -> ✅ Saved transcript.")
                        transcribed_count += 1
                    else:
                        print(" -> 📝 No speech detected, skipping file creation.")
                        skipped_count += 1

            except Exception as e:
                # In lỗi gọn gàng hơn
                error_message = str(e)
                if "Failed to load audio" in error_message:
                    print(
                        " -> ❌ Error: Failed to load audio (likely corrupt or unsupported format)."
                    )
                else:
                    print(f" -> ❌ An unexpected error occurred: {e}")
                skipped_count += 1
                continue

        print("-" * 50)
        print("🎉 Transcription process completed!")
        print(f"   - Transcribed/Already Existed: {transcribed_count}")
        print(f"   - Skipped (No audio/Error): {skipped_count}")


# Simple CLI - run this script to generate transcripts
if __name__ == "__main__":
    print("🚀 Starting Audio Transcription Builder")
    print("=" * 50)

    # Configuration
    DATA_DIR = "data"
    OUTPUT_DIR = "output/transcripts"
    MODEL_ID = "base"  # Use 'base' for speed, 'medium' for better accuracy

    print(f"📁 Media directory: {DATA_DIR}")
    print(f"📄 Output directory: {OUTPUT_DIR}")
    print(f"🤖 Whisper model: {MODEL_ID}")
    print("=" * 50)

    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory '{DATA_DIR}' not found!")
        print(f"Please create the directory and add your video/audio files there.")
        exit(1)

    # Create and run indexer
    audio_indexer = AudioIndexer(model_id=MODEL_ID)
    audio_indexer.transcribe_media_from_directory(DATA_DIR, OUTPUT_DIR)
