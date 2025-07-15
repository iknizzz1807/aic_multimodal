import os
import json
from typing import List, Dict
import faiss
import numpy as np
from src.image_processor import ImageProcessor
from src.video_indexer import VideoIndexer
from src import config


class VisualIndexer:
    """
    Xử lý việc tạo và quản lý index FAISS cho cả ảnh tĩnh và các khung hình từ video.
    """

    def __init__(self, model_id: str = config.CLIP_MODEL_ID):
        self.model_id = model_id
        self.image_processor = ImageProcessor(model_id)
        self.video_indexer = VideoIndexer()  # MỚI: Khởi tạo video indexer
        self.index = None
        self.index_to_media_data = {}
        print("✅ Visual indexer initialized")

    def create_index_from_directory(
        self,
        data_directory: str = config.DATA_DIR,
        output_index_file: str = config.VISUAL_INDEX_FILE,
        output_mapping_file: str = config.MEDIA_DATA_MAPPING_FILE,
        index_type: str = config.FAISS_INDEX_TYPE,
    ) -> bool:
        """
        Tạo index FAISS từ ảnh và video trong một thư mục.
        """
        print(f"🔍 Creating visual index from directory: {data_directory}")

        # 1. Lấy danh sách ảnh tĩnh
        static_images = self.image_processor.get_image_files(data_directory)
        static_image_data = [
            {
                "type": "image",
                "path": path,
                "media_id": os.path.basename(path),
                "timestamp": None,
            }
            for path in static_images
        ]
        print(f"🖼️ Found {len(static_image_data)} static images.")

        # 2. Trích xuất frame từ video
        video_frame_data = self.video_indexer.extract_frames_from_directory(
            data_directory
        )
        print(f"🎞️ Found {len(video_frame_data)} video frames.")

        # 3. Kết hợp cả hai
        all_visual_data = static_image_data + video_frame_data
        if not all_visual_data:
            print("❌ No visual data (images or video frames) found.")
            return False

        print(f"📊 Total visual items to index: {len(all_visual_data)}")

        # 4. Tạo embedding cho tất cả
        embeddings_dict = self.batch_process_visuals(all_visual_data)
        if not embeddings_dict:
            print("❌ No embeddings generated.")
            return False

        # 5. Tạo index FAISS và file mapping mới
        success = self.create_faiss_index(
            embeddings_dict, output_index_file, output_mapping_file, index_type
        )
        if success:
            print(f"✅ Index created successfully!")
            print(f"   💾 Index file: {output_index_file}")
            print(f"   💾 Mapping file: {output_mapping_file}")
        return success

    def batch_process_visuals(self, visual_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Tạo embedding cho một list các item hình ảnh."""
        embeddings = {}
        total = len(visual_data)
        print(f"🤖 Processing {total} visual items to generate embeddings...")

        for i, data in enumerate(visual_data):
            path = data["path"]
            try:
                # print(f"   Processing {i+1}/{total}: {os.path.basename(path)}")
                embedding = self.image_processor.image_to_embedding(path)
                if embedding is not None:
                    # Dùng đường dẫn làm key tạm thời
                    embeddings[path] = embedding
                else:
                    print(f"   ⚠️ Failed to process: {path}")
            except Exception as e:
                print(f"   ❌ Error processing {path}: {e}")

        print(f"✅ Successfully processed {len(embeddings)}/{total} items.")
        return embeddings

    def create_faiss_index(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        output_index_file: str,
        output_mapping_file: str,
        index_type: str,
    ) -> bool:
        """
        Tạo index FAISS và file mapping với cấu trúc dữ liệu mới.
        """
        try:
            # Tạo thư mục output nếu chưa có
            os.makedirs(os.path.dirname(output_index_file), exist_ok=True)

            # Lấy lại visual_data từ key của embeddings_dict để đảm bảo thứ tự
            all_visual_data = (
                static_image_data + video_frame_data
            )  # Tái sử dụng từ hàm gọi

            # Lọc ra những item đã được tạo embedding thành công
            successful_data = [
                d for d in all_visual_data if d["path"] in embeddings_dict
            ]

            paths = [d["path"] for d in successful_data]
            vector_matrix = np.vstack([embeddings_dict[path] for path in paths])

            d = vector_matrix.shape[1]
            print(
                f"📊 Creating FAISS index with {vector_matrix.shape[0]} vectors of dimension {d}."
            )

            if index_type == "flat_ip":
                index = faiss.IndexFlatIP(d)
            elif index_type == "flat_l2":
                index = faiss.IndexFlatL2(d)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            index.add(vector_matrix)
            faiss.write_index(index, output_index_file)

            # Tạo mapping mới
            index_to_media_data = {
                str(i): {
                    "type": data["type"],
                    "path": data["path"],
                    "media_id": data["media_id"],
                    "timestamp": data["timestamp"],
                }
                for i, data in enumerate(successful_data)
            }

            with open(output_mapping_file, "w", encoding="utf-8") as f:
                json.dump(index_to_media_data, f, indent=2, ensure_ascii=False)

            self.index = index
            self.index_to_media_data = index_to_media_data
            return True

        except Exception as e:
            print(f"❌ Error creating FAISS index: {e}")
            return False


# ----- Các hàm khác như load_index, get_index_stats, ... giữ nguyên hoặc cập nhật nhỏ -----
# Tôi sẽ lược bỏ chúng ở đây để cho gọn, bạn có thể tự cập nhật chúng để dùng
# self.index_to_media_data thay cho self.index_to_path
