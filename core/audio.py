import os
import subprocess
import torch
import whisper
from typing import Iterator, Dict, Any
import numpy as np

# Thêm import ffmpeg, yêu cầu pip install ffmpeg-python
try:
    import ffmpeg
except ImportError:
    print("-------------------------------------------------------------------")
    print("!!! LỖI QUAN TRỌNG !!!")
    print("Thư viện 'ffmpeg-python' chưa được cài đặt.")
    print("Vui lòng chạy lệnh: pip install ffmpeg-python")
    print("-------------------------------------------------------------------")
    ffmpeg = None

# Import thư viện CLAP
from laion_clap import CLAP_Module

# Import các hàm tiền xử lý của CLAP để đảm bảo tương thích
from laion_clap.training.data import int16_to_float32, float32_to_int16


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


class AudioEventProcessor:
    """
    Handles audio event detection and embedding using CLAP.
    Tối ưu hóa để xử lý audio theo luồng (streaming) và theo lô (batching)
    để đạt hiệu năng cao và sử dụng bộ nhớ hiệu quả.
    """

    def __init__(self, model_id: str, enable_fusion=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Audio Event model (CLAP) on {self.device}")

        self.model = CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-tiny",
            tmodel="roberta",
        ).to(self.device)

        self.model.load_ckpt()
        print("✅ AudioEventProcessor initialized.")
        if ffmpeg is None:
            raise ImportError(
                "ffmpeg-python is required for AudioEventProcessor to work."
            )

    def get_text_embedding(self, text: str) -> list:
        """Creates an embedding for a text description."""
        text_data = [text]
        with torch.no_grad():
            text_embedding = self.model.get_text_embedding(text_data, use_tensor=False)
        return text_embedding[0]

    def _process_chunk_batch(
        self, chunks: list, start_times: list, chunk_duration: int
    ):
        """Hàm trợ giúp để xử lý một lô (batch) các chunk audio."""
        with torch.no_grad():
            # Đưa cả batch vào model một lần để tận dụng GPU
            audio_embeds = self.model.get_audio_embedding_from_data(
                x=chunks, use_tensor=False
            )

        if audio_embeds is not None:
            for i, embed in enumerate(audio_embeds):
                start = float(start_times[i])
                yield {
                    "start": start,
                    "end": start + chunk_duration,
                    "vector": embed,
                }

    def extract_event_embeddings(
        self, file_path: str, chunk_duration: int = 5, batch_size: int = 16
    ) -> Iterator[Dict[str, Any]]:
        """
        Trích xuất embedding sự kiện âm thanh bằng cách stream audio qua FFmpeg.
        Phương pháp này hiệu quả hơn nhiều về bộ nhớ và tốc độ so với việc tải toàn bộ file.
        """
        sample_rate = 48000  # Tần số mẫu yêu cầu của CLAP
        chunk_samples = sample_rate * chunk_duration
        bytes_per_sample = 2  # 16-bit PCM (s16le) có 2 bytes/mẫu

        try:
            # 1. Thiết lập tiến trình FFmpeg để đọc và chuyển đổi audio
            # -ac 1: Chuyển thành mono
            # -ar 48000: Resample về 48kHz
            # -f s16le: Định dạng output là 16-bit signed little-endian PCM
            # pipe:stdout=True: Đưa output ra pipe để Python đọc
            process = (
                ffmpeg.input(file_path)
                .output(
                    "pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate
                )
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            chunk_buffer = []
            start_time_buffer = []
            total_chunks_processed = 0

            # 2. Vòng lặp đọc dữ liệu từ stream của FFmpeg
            while True:
                # Đọc đủ số byte cho một chunk
                in_bytes = process.stdout.read(chunk_samples * bytes_per_sample)
                if not in_bytes:
                    break

                # Chuyển đổi byte thành mảng numpy float32 và chuẩn hóa về [-1, 1]
                audio_chunk = (
                    np.frombuffer(in_bytes, np.int16).astype(np.float32) / 32768.0
                )

                # Áp dụng tiền xử lý giống hệt CLAP gốc
                audio_chunk = int16_to_float32(float32_to_int16(audio_chunk))

                # Thêm chunk vào buffer để xử lý theo lô
                current_time = total_chunks_processed * chunk_duration
                chunk_buffer.append(audio_chunk)
                start_time_buffer.append(current_time)
                total_chunks_processed += 1

                # Khi buffer đầy, xử lý lô và xóa buffer
                if len(chunk_buffer) >= batch_size:
                    yield from self._process_chunk_batch(
                        chunk_buffer, start_time_buffer, chunk_duration
                    )
                    chunk_buffer.clear()
                    start_time_buffer.clear()

            # 3. Xử lý các chunk còn lại trong buffer sau khi vòng lặp kết thúc
            if chunk_buffer:
                yield from self._process_chunk_batch(
                    chunk_buffer, start_time_buffer, chunk_duration
                )

            # Đợi tiến trình ffmpeg kết thúc và kiểm tra lỗi
            process.wait()

        except Exception as e:
            print(
                f"❌ Error during audio event extraction for {os.path.basename(file_path)}: {e}"
            )
            # Nếu có lỗi, thử đọc thông báo từ stderr của ffmpeg để debug
            if "process" in locals():
                err_output = process.stderr.read().decode(errors="ignore")
                if err_output:
                    print(f"   FFmpeg stderr: {err_output.strip()}")
