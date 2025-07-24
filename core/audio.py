import os
import subprocess
import torch
import whisper


class AudioProcessor:
    """Handles audio transcription using Whisper."""

    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Audio model (Whisper): {model_id} on {self.device}")
        self.model = whisper.load_model(model_id, device=self.device)
        print("✅ AudioProcessor initialized.")

    def transcribe(self, file_path: str) -> dict:
        """Transcribes an audio or video file."""
        try:
            # fp16=True nếu có CUDA để tăng tốc
            result = self.model.transcribe(file_path, fp16=torch.cuda.is_available())
            return result
        except Exception as e:
            print(
                f"Error during audio transcription for {os.path.basename(file_path)}: {e}"
            )
            return {}

    @staticmethod
    def has_audio_stream(video_path: str) -> bool:
        """Checks if a media file has an audio stream using ffprobe."""
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
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return len(result.stdout.strip()) > 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                f"⚠️  ffprobe not found or failed for {os.path.basename(video_path)}. Assuming no audio."
            )
            return False
