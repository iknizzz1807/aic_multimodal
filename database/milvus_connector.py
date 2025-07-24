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

    def setup_audio_event_collection(self):
        collection_name = config.AUDIO_EVENT_COLLECTION_NAME
        if utility.has_collection(collection_name, using=f"worker_{os.getpid()}"):
            print(f"Collection '{collection_name}' already exists. Loading...")
            self.audio_event_collection = Collection(
                collection_name, using=f"worker_{os.getpid()}"
            )
            self.audio_event_collection.load()
            print("✅ Audio event collection loaded.")
            return

        print(f"Creating collection: {collection_name}")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="media_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="start", dtype=DataType.FLOAT),
            FieldSchema(name="end", dtype=DataType.FLOAT),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=config.CLAP_EMBEDDING_DIM,  # Use the new config constant
            ),
        ]
        schema = CollectionSchema(fields, "Audio event embeddings collection (CLAP)")
        self.audio_event_collection = Collection(
            collection_name, schema, using=f"worker_{os.getpid()}"
        )

        index_params = {
            "metric_type": "IP",  # Inner Product is equivalent to Cosine Similarity on normalized vectors
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self.audio_event_collection.create_index(
            field_name="vector", index_params=index_params
        )
        print(f"✅ Collection '{collection_name}' created and index built.")
        self.audio_event_collection.load()

    def insert(self, data):
        if not data:
            return
        # Đảm bảo collection đã được load trước khi chèn
        if self.collection is None:
            print("❌ Milvus collection is not initialized. Cannot insert.")
            return

        print(f"Inserting {len(data)} vectors into Milvus via [PID: {os.getpid()}]...")
        self.collection.insert(data)

    def insert_audio_events(self, data):
        if not data:
            return
        if (
            not hasattr(self, "audio_event_collection")
            or self.audio_event_collection is None
        ):
            print("❌ Milvus audio event collection is not initialized. Cannot insert.")
            return

        print(
            f"Inserting {len(data)} audio event vectors into Milvus via [PID: {os.getpid()}]..."
        )
        self.audio_event_collection.insert(data)

    def get_collection(self, name=config.VISUAL_COLLECTION_NAME):
        if name == config.VISUAL_COLLECTION_NAME:
            if self.collection is None:
                self.setup_visual_collection()
            return self.collection
        elif name == config.AUDIO_EVENT_COLLECTION_NAME:
            if (
                not hasattr(self, "audio_event_collection")
                or self.audio_event_collection is None
            ):
                self.setup_audio_event_collection()
            return self.audio_event_collection
        raise ValueError(f"Unknown collection name: {name}")
