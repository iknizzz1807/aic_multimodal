"""
AI Multimodal Search API
Main entry point for the application
"""

from src.api import APIServer


def main():
    """Main function to start the API server"""
    # Config
    INDEX_FILE = "output/faiss_visual.index"
    MAPPING_FILE = "output/index_to_path.json"
    TRANSCRIPT_DIR = "output/transcripts"
    MODEL_ID = "openai/clip-vit-base-patch32"

    server = APIServer(
        index_file=INDEX_FILE,
        mapping_file=MAPPING_FILE,
        transcript_dir=TRANSCRIPT_DIR,
        model_id=MODEL_ID,
    )

    server.run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
