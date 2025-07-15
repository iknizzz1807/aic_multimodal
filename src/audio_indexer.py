import os
import json
import whisper
import torch


class AudioIndexer:
    """Handles audio transcription using Whisper"""

    def __init__(self, model_id: str = "base"):
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

    def transcribe_media_from_directory(
        self,
        media_directory: str,
        output_directory: str = "output/transcripts",
        media_extensions: tuple = (".mp4", ".mp3", ".wav", ".m4a"),
    ):
        """
        Transcribe all media files in a directory.

        Args:
            media_directory: Directory containing media files.
            output_directory: Directory to save transcript JSON files.
            media_extensions: Tuple of valid media file extensions.
        """
        print(f"🔍 Transcribing media from directory: {media_directory}")
        os.makedirs(output_directory, exist_ok=True)

        media_files = [
            os.path.join(media_directory, f)
            for f in os.listdir(media_directory)
            if f.lower().endswith(media_extensions)
        ]

        if not media_files:
            print("❌ No media files found in directory.")
            return

        print(f"📁 Found {len(media_files)} media files to process.")
        total = len(media_files)

        for i, path in enumerate(media_files):
            try:
                print(f"   Processing {i+1}/{total}: {os.path.basename(path)}")

                # Perform transcription
                result = self.model.transcribe(path, fp16=torch.cuda.is_available())

                # Get the original filename without extension
                base_filename = os.path.splitext(os.path.basename(path))[0]
                output_path = os.path.join(output_directory, f"{base_filename}.json")

                # Save the segments which include text and timestamps
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result["segments"], f, ensure_ascii=False, indent=2)

                print(f"   ✅ Saved transcript to: {output_path}")

            except Exception as e:
                print(f"   ❌ Error processing {path}: {e}")
                continue

        print("🎉 Transcription process completed!")


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
