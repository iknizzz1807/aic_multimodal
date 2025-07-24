from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import config


class ElasticsearchConnector:
    def __init__(self):
        try:
            self.es = Elasticsearch(f"http://{config.ES_HOST}:{config.ES_PORT}")
            if not self.es.ping():
                raise ConnectionError("Could not connect to Elasticsearch")
            print("✅ Connected to Elasticsearch.")
        except Exception as e:
            print(f"❌ Failed to connect to Elasticsearch: {e}")
            raise

    def setup_transcript_index(self):
        index_name = config.TRANSCRIPT_INDEX_NAME
        if self.es.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists.")
            return

        print(f"Creating index: {index_name}")
        mapping = {
            "properties": {
                "media_id": {"type": "keyword"},
                "start": {"type": "float"},
                "end": {"type": "float"},
                "text": {"type": "text", "analyzer": "standard"},
            }
        }
        self.es.indices.create(index=index_name, mappings=mapping)
        print(f"✅ Index '{index_name}' created.")

    def bulk_insert(self, documents):
        if not documents:
            return
        print(f"Bulk inserting {len(documents)} documents into Elasticsearch...")

        actions = [
            {
                "_index": config.TRANSCRIPT_INDEX_NAME,
                "_source": doc,
            }
            for doc in documents
        ]
        success, failed = bulk(self.es, actions)
        print(f"  - Success: {success}, Failed: {failed}")
        if failed:
            print(f"  - Some documents failed to index: {failed}")
