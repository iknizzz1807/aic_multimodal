import os
import json
from typing import List, Dict, Optional, Tuple
import faiss
import numpy as np
from src.image_processor import ImageProcessor


class ImageIndexer:
    """Handles FAISS index creation and management for images"""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        """
        Initialize image indexer

        Args:
            model_id: CLIP model identifier
        """
        self.model_id = model_id
        self.image_processor = ImageProcessor(model_id)
        self.index = None
        self.index_to_path = {}

        print("✅ Image indexer initialized")

    def create_index_from_directory(
        self,
        image_directory: str,
        output_index_file: str = "output/faiss_visual.index",
        output_mapping_file: str = "output/index_to_path.json",
        index_type: str = "flat_ip",
    ) -> bool:
        """
        Create FAISS index from images in directory

        Args:
            image_directory: Directory containing images
            output_index_file: Path to save FAISS index
            output_mapping_file: Path to save index mapping
            index_type: Type of FAISS index ('flat_ip', 'flat_l2', 'ivf')

        Returns:
            True if successful, False otherwise
        """
        print(f"🔍 Creating index from directory: {image_directory}")

        # Get all image files
        image_paths = self.image_processor.get_image_files(image_directory)
        if not image_paths:
            print("❌ No images found in directory")
            return False

        print(f"📁 Found {len(image_paths)} images")

        # Process images to embeddings
        embeddings_dict = self.batch_process_images(image_paths)
        if not embeddings_dict:
            print("❌ No embeddings generated")
            return False

        # Create FAISS index
        success = self.create_faiss_index(
            embeddings_dict, output_index_file, output_mapping_file, index_type
        )

        if success:
            print(f"✅ Index created successfully!")
            print(f"📄 Index file: {output_index_file}")
            print(f"📄 Mapping file: {output_mapping_file}")

        return success

    def batch_process_images(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Process multiple images to embeddings with progress tracking

        Args:
            image_paths: List of image file paths

        Returns:
            Dictionary mapping image paths to embeddings
        """
        embeddings = {}
        total = len(image_paths)

        print(f"🖼️ Processing {total} images...")

        for i, path in enumerate(image_paths):
            try:
                print(f"   Processing {i+1}/{total}: {os.path.basename(path)}")
                embedding = self.image_processor.image_to_embedding(path)

                if embedding is not None:
                    embeddings[path] = embedding
                else:
                    print(f"   ⚠️ Failed to process: {path}")

            except Exception as e:
                print(f"   ❌ Error processing {path}: {e}")
                continue

        print(f"✅ Successfully processed {len(embeddings)}/{total} images")
        return embeddings

    def create_faiss_index(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        output_index_file: str,
        output_mapping_file: str,
        index_type: str = "flat_ip",
    ) -> bool:
        """
        Create FAISS index from embeddings dictionary

        Args:
            embeddings_dict: Dictionary mapping paths to embeddings
            output_index_file: Path to save FAISS index
            output_mapping_file: Path to save index mapping
            index_type: Type of FAISS index

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_index_file), exist_ok=True)

            # Convert embeddings to matrix
            paths = list(embeddings_dict.keys())
            embeddings_list = [embeddings_dict[path] for path in paths]
            vector_matrix = np.vstack(embeddings_list)

            print(f"📊 Creating FAISS index with {vector_matrix.shape[0]} vectors")
            print(f"📊 Vector dimension: {vector_matrix.shape[1]}")

            # Create FAISS index based on type
            d = vector_matrix.shape[1]

            if index_type == "flat_ip":
                # Inner Product (good for normalized vectors)
                index = faiss.IndexFlatIP(d)
            elif index_type == "flat_l2":
                # L2 distance
                index = faiss.IndexFlatL2(d)
            elif index_type == "ivf":
                # IVF for larger datasets
                nlist = min(100, vector_matrix.shape[0] // 10)
                quantizer = faiss.IndexFlatIP(d)
                index = faiss.IndexIVFFlat(
                    quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
                )
                index.train(vector_matrix)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            # Add vectors to index
            index.add(vector_matrix)

            # Create index mapping
            index_to_path = {i: path for i, path in enumerate(paths)}

            # Save index and mapping
            faiss.write_index(index, output_index_file)

            with open(output_mapping_file, "w") as f:
                json.dump(index_to_path, f, indent=2)

            # Store in instance
            self.index = index
            self.index_to_path = index_to_path

            return True

        except Exception as e:
            print(f"❌ Error creating FAISS index: {e}")
            return False

    def load_index(
        self,
        index_file: str = "output/faiss_visual.index",
        mapping_file: str = "output/index_to_path.json",
    ) -> bool:
        """
        Load existing FAISS index and mapping

        Args:
            index_file: Path to FAISS index file
            mapping_file: Path to mapping JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"📂 Loading FAISS index from: {index_file}")
            self.index = faiss.read_index(index_file)

            print(f"📂 Loading mapping from: {mapping_file}")
            with open(mapping_file, "r") as f:
                self.index_to_path = json.load(f)

            print(f"✅ Loaded index with {self.index.ntotal} vectors")
            print(f"✅ Loaded mapping with {len(self.index_to_path)} entries")

            return True

        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False

    def search_similar_images(
        self, query_text: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar images using text query

        Args:
            query_text: Text description to search for
            top_k: Number of results to return

        Returns:
            List of tuples (image_path, similarity_score)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")

        # Convert text to embedding
        query_vector = self.image_processor.text_to_embedding(query_text)

        # Search in FAISS
        distances, indices = self.index.search(query_vector, top_k)

        # Convert results
        results = []
        for i in range(top_k):
            if i < len(indices[0]):
                result_index = indices[0][i]
                result_path = self.index_to_path.get(str(result_index))
                if result_path:
                    similarity_score = float(distances[0][i])
                    results.append((result_path, similarity_score))

        return results

    def get_index_stats(self) -> Dict[str, any]:
        """Get statistics about the current index"""
        if self.index is None:
            return {"status": "No index loaded"}

        return {
            "total_vectors": self.index.ntotal,
            "vector_dimension": self.index.d,
            "index_type": type(self.index).__name__,
            "total_mappings": len(self.index_to_path),
            "model_info": self.image_processor.get_model_info(),
        }

    def add_images_to_index(
        self, new_image_paths: List[str], update_files: bool = True
    ) -> bool:
        """
        Add new images to existing index

        Args:
            new_image_paths: List of new image paths
            update_files: Whether to update saved index files

        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")

        print(f"➕ Adding {len(new_image_paths)} new images to index...")

        # Process new images
        new_embeddings = self.batch_process_images(new_image_paths)
        if not new_embeddings:
            print("❌ No new embeddings generated")
            return False

        # Add to index
        current_size = self.index.ntotal
        new_vectors = np.vstack(
            [new_embeddings[path] for path in new_embeddings.keys()]
        )
        self.index.add(new_vectors)

        # Update mapping
        for i, path in enumerate(new_embeddings.keys()):
            self.index_to_path[str(current_size + i)] = path

        print(f"✅ Added {len(new_embeddings)} new images to index")

        # Update files if requested
        if update_files:
            faiss.write_index(self.index, "output/faiss_visual.index")
            with open("output/index_to_path.json", "w") as f:
                json.dump(self.index_to_path, f, indent=2)
            print("💾 Updated index files")

        return True


# Simple CLI - just run the script!
if __name__ == "__main__":
    print("🚀 Starting Image Index Builder")
    print("=" * 50)

    # Default configuration
    DATA_DIR = "data"
    INDEX_FILE = "output/faiss_visual.index"
    MAPPING_FILE = "output/index_to_path.json"
    MODEL_ID = "openai/clip-vit-base-patch32"
    INDEX_TYPE = "flat_ip"

    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📄 Index file: {INDEX_FILE}")
    print(f"📄 Mapping file: {MAPPING_FILE}")
    print(f"🤖 Model: {MODEL_ID}")
    print(f"📊 Index type: {INDEX_TYPE}")
    print("=" * 50)

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory '{DATA_DIR}' not found!")
        print(f"Please create the directory and add your images there.")
        exit(1)

    # Create indexer and build index
    indexer = ImageIndexer(MODEL_ID)
    success = indexer.create_index_from_directory(
        DATA_DIR, INDEX_FILE, MAPPING_FILE, INDEX_TYPE
    )

    if success:
        print("\n" + "=" * 50)
        print("🎉 Index creation completed successfully!")
        print("=" * 50)

        stats = indexer.get_index_stats()
        print("📊 Index Statistics:")
        for key, value in stats.items():
            print(f"   • {key}: {value}")

        print("\n🚀 You can now start the API server:")
        print("   python main.py")
        print("   python api_server.py")

    else:
        print("\n" + "=" * 50)
        print("❌ Index creation failed!")
        print("=" * 50)
        print("Please check the errors above and try again.")
