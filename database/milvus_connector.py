from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import os
import config


class MilvusConnector:
    def __init__(self):
        self.collection = None
        try:
            # Sử dụng alias để tránh xung đột kết nối trong môi trường đa tiến trình
            connections.connect(
                alias=f"worker_{os.getpid()}",
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
            )
            print(f"✅ [PID: {os.getpid()}] Connected to Milvus.")
        except Exception as e:
            print(f"❌ [PID: {os.getpid()}] Failed to connect to Milvus: {e}")
            raise

    def setup_visual_collection(self):
        collection_name = config.VISUAL_COLLECTION_NAME
        if utility.has_collection(collection_name, using=f"worker_{os.getpid()}"):
            print(f"Collection '{collection_name}' already exists. Loading...")
            self.collection = Collection(collection_name, using=f"worker_{os.getpid()}")
            self.collection.load()
            print("✅ Collection loaded.")
            return

        print(f"Creating collection: {collection_name}")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="media_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="timestamp", dtype=DataType.FLOAT),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=config.CLIP_EMBEDDING_DIM,
            ),
        ]
        schema = CollectionSchema(fields, "Visual media embeddings collection")
        self.collection = Collection(
            collection_name, schema, using=f"worker_{os.getpid()}"
        )

        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
        print(f"✅ Collection '{collection_name}' created and index built.")
        self.collection.load()

    def insert(self, data):
        if not data:
            return
        # Đảm bảo collection đã được load trước khi chèn
        if self.collection is None:
            print("❌ Milvus collection is not initialized. Cannot insert.")
            return

        print(f"Inserting {len(data)} vectors into Milvus via [PID: {os.getpid()}]...")
        self.collection.insert(data)

    def get_collection(self):
        # Đảm bảo collection được khởi tạo nếu chưa có
        if self.collection is None:
            self.setup_visual_collection()
        return self.collection
