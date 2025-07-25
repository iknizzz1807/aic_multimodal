### **high_implementation_revised.md**

### **Sơ đồ kiến trúc tổng thể (Đã điều chỉnh cho cuộc thi)**

```
+-----------------------------------------------------------------------------------+
| Automated Testing System (Single Request) |
+---------------------------------+-------------------------------------------------+
| (HTTP Request)
+---------------------------------v-------------------------------------------------+
| UNIFIED SEARCH API SERVICE (Python - FastAPI) |
| - Tải tất cả model cần thiết khi khởi động (CLIP, Cross-Encoder, etc.) |
| - Endpoint 1: /search (Logic chính) |
|   1. Async Recall: Gọi song song Milvus & Elasticsearch để lấy ứng viên. |
|   2. Fusion: Kết hợp kết quả bằng RRF. |
|   3. Rerank: Dùng Cross-Encoder để xếp hạng lại top ứng viên. |
|   4. Response: Trả về kết quả cuối cùng. |
| - Endpoint 2: /analyze (Tính năng mở rộng nếu có) |
+-----------------+-------------------+-------------------------------------------+
| (Async DB Calls)| (Async DB Calls)  | (Model Inference)
+------v----------+ +-----v-----------+ +-----v-------------------------------------+
| VECTOR DB | | TEXT SEARCH | | LOCAL AI MODELS |
| (Milvus) | | ENGINE | | (PyTorch / ONNX Runtime) |
| - Visual Vectors| | (Elasticsearch)| | - CLIP (cho query) |
| - Audio Events | | - Transcripts | | - Cross-Encoder (cho reranking) |
+-----------------+ +-----------------+ +-----------------------------------------+
```

---

### **Kiến Trúc End-Product Chi Tiết**

### **Phần 1 (Không đổi về mục tiêu, nhưng bổ sung chi tiết): Pipeline Xử Lý Dữ Liệu Nâng Cao**

**Mục tiêu:** Thực hiện **một lần duy nhất (offline)** để biến đổi dữ liệu thô thành các chỉ mục (index) thông minh và giàu thông tin. Đây là giai đoạn quan trọng nhất để tối ưu độ chính xác.

**Triết lý thiết kế:**

- **Song song hóa tối đa:** Sử dụng `multiprocessing` để xử lý song song các file media trên tất cả các core CPU.
- **Trích xuất thông tin sâu (Rich Feature Extraction):** Không chỉ tạo vector, mà còn trích xuất các metadata hữu ích có thể được dùng để lọc hoặc xếp hạng sau này.
- **Lưu trữ chuyên dụng:** Sử dụng Milvus cho dữ liệu vector và Elasticsearch cho dữ liệu văn bản.

#### **Luồng xử lý (`run_batch_processing.py`):**

1.  **Script Chính:** Quét thư mục `data`, tạo một `multiprocessing.Pool` và phân phát các đường dẫn file cho các hàm worker.
2.  **Hàm Worker `process_single_media_file(file_path)` (Trái tim của pipeline):**
    - **Khởi tạo:** Tải các model cần thiết (CLIP, Whisper, có thể thêm YOLO/SceneDetector/CLAP) và kết nối tới DB.
    - **Xử lý Visual:**
      - **Trích xuất Keyframe thông minh:** Thay vì trích xuất đều đặn, có thể dùng các thư viện như `PySceneDetect` để chỉ lấy những frame đại diện cho sự thay đổi trong cảnh, giảm nhiễu và số lượng vector.
      - **Tạo Vector đa mức (Multi-level Embedding):**
        - **Vector toàn cảnh:** Dùng CLIP để tạo vector cho cả khung hình.
        - **(Nâng cao) Vector đối tượng:** Dùng một model phát hiện đối tượng (như YOLO) để cắt ra các đối tượng chính (người, xe,...) và tạo vector riêng cho từng đối tượng bằng CLIP.
      - **Tính toán Metadata chất lượng:** Với mỗi frame, tính toán các chỉ số như độ nét (Laplacian variance), độ sáng.
      - **Lưu vào Milvus:** Đẩy một bản ghi đa trường vào collection `visual_collection`: `{media_id, timestamp, vector, quality_score, object_type (nếu có)}`.
    - **Xử lý Audio:**
      - **Bóc tách lời thoại (ASR):** Dùng Whisper để lấy transcript có dấu thời gian.
      - **Lưu vào Elasticsearch:** Đẩy các segment vào index `transcripts_index`: `{media_id, start, end, text}`.
      - **(Nâng cao) Nhận dạng sự kiện âm thanh:** Dùng một model như CLAP để tạo vector cho các đoạn âm thanh không phải lời nói. Lưu vào một collection riêng trong Milvus, ví dụ `audio_events_collection`: `{media_id, start, end, event_vector}`.

---

### **Phần 2 (Hoàn toàn mới, thay thế Giai đoạn 2 & 3 cũ): API Service Tìm kiếm Hợp nhất và Tái xếp hạng**

**Mục tiêu:** Thiết kế một API service duy nhất, hiệu năng cao (sử dụng `asyncio` của Python) để thực hiện toàn bộ luồng tìm kiếm, từ thu hồi ứng viên đến tái xếp hạng và trả về kết quả cuối cùng.

**Triết lý thiết kế:**

- **Tối ưu cho một request:** Toàn bộ logic nằm trong một tiến trình ứng dụng, loại bỏ độ trễ giao tiếp mạng giữa các service.
- **Bất đồng bộ I/O-bound:** Sử dụng `async/await` để thực hiện các tác vụ chờ đợi (query DB) một cách song song, giảm thiểu thời gian chờ.
- **Kiến trúc Recall-then-Rank:** Tách biệt rõ ràng 2 bước trong cùng một hàm:
  1.  **Recall (Thu hồi):** Lấy nhanh một lượng lớn ứng viên tiềm năng từ DB.
  2.  **Rank (Xếp hạng):** Dùng model AI "đắt tiền" hơn để chấm điểm lại một lượng nhỏ ứng viên, đảm bảo độ chính xác cho top đầu.

#### **Luồng xử lý của Endpoint `POST /search` (Mã giả - Python FastAPI):**

```python
# main_api.py

from fastapi import FastAPI
from services.search_service import SearchService, SearchQuery, UnifiedSearchResponse

app = FastAPI()
# Tải tất cả model và kết nối DB một lần duy nhất khi server khởi động
search_service = SearchService()

@app.post("/search", response_model=UnifiedSearchResponse)
async def unified_search(query: SearchQuery):
    """
    Thực hiện tìm kiếm hợp nhất, bất đồng bộ và có tái xếp hạng.
    """
    return await search_service.perform_unified_search(query.text, query.top_k)

# services/search_service.py

class SearchService:
    def __init__(self):
        # Tải tất cả tài nguyên:
        self.milvus_conn = ...
        self.es_conn = ...
        self.clip_model = ...
        # Tải model "hạng nặng" cho re-ranking
        self.cross_encoder = ...

    async def _recall_visual(self, query_vector, k):
        # Gọi bất đồng bộ tới Milvus để tìm kiếm vector
        # return await self.milvus_conn.search(...)
        pass

    async def _recall_text(self, text, k):
        # Gọi bất đồng bộ tới Elasticsearch để tìm kiếm text
        # return await self.es_conn.search(...)
        pass

    async def perform_unified_search(self, text_query: str, top_k: int) -> list:
        # --- BƯỚC 1: RECALL (Bất đồng bộ) ---
        query_vector = self.clip_model.encode(text_query)

        # Tạo các task để chạy song song
        visual_recall_task = self._recall_visual(query_vector, k=100)
        text_recall_task = self._recall_text(text_query, k=100)

        # Chạy và đợi kết quả từ cả hai nguồn
        visual_candidates, text_candidates = await asyncio.gather(
            visual_recall_task,
            text_recall_task
        )

        # --- BƯỚC 2: FUSION ---
        # Dùng RRF để kết hợp `visual_candidates` và `text_candidates`
        # thành một danh sách `fused_candidates` (top 100).
        fused_candidates = self._fuse_results_rrf(visual_candidates, text_candidates)

        # --- BƯỚC 3: RE-RANK (Phần quyết định độ chính xác) ---
        # Lấy thông tin chi tiết (ảnh/text) cho top 100 ứng viên
        rerank_data = self._prepare_data_for_reranking(fused_candidates)

        # Dùng model Cross-Encoder để chấm điểm lại độ tương quan
        # giữa query và từng ứng viên. Đây là một tác vụ CPU-bound.
        # Có thể chạy trong ThreadPoolExecutor để không block main event loop.
        rerank_scores = await asyncio.to_thread(
            self.cross_encoder.predict, rerank_data
        )

        # --- BƯỚC 4: FINAL RANKING & RESPONSE ---
        # Kết hợp điểm fusion ban đầu với điểm từ cross-encoder
        final_results = self._combine_scores(fused_candidates, rerank_scores)

        # Sắp xếp lần cuối và trả về top_k kết quả tốt nhất
        return sorted(final_results, key=lambda x: x.score, reverse=True)[:top_k]

```

---

### **Phần 3 (Trước là Giai đoạn 4): Phân Tích Chuyên Sâu (Tính năng mở rộng)**

**Mục tiêu:** Cung cấp các khả năng tương tác sâu hơn, nếu thời gian cho phép.

**Triết lý thiết kế:** Tận dụng các model đã tải sẵn hoặc tải thêm các model lớn (LVLM) để thực hiện các tác vụ phân tích trên một media cụ thể.

#### **Các tính năng có thể có:**

1.  **Endpoint `POST /analyze`:**
    - **Visual Question Answering (VQA):** Nhận `media_id`, `timestamp` và một `question`. Service sẽ lấy frame tương ứng, đưa vào một model LVLM (như LLaVA) cùng với câu hỏi để trả về câu trả lời.
    - **Summarization:** Nhận một `media_id`, lấy toàn bộ transcript từ Elasticsearch và dùng một LLM để tóm tắt nội dung.

---

### **Kết Luận Toàn Bộ Kiến Trúc (Đã điều chỉnh)**

Kiến trúc này tập trung vào những gì quan trọng nhất cho cuộc thi:

1.  **Giai đoạn Offline (Phần 1):** Xây dựng một pipeline xử lý dữ liệu **siêu kỹ lưỡng** để tạo ra các index chất lượng cao nhất có thể. Đây là nền tảng của độ chính xác.
2.  **Giai đoạn Online (Phần 2):** Xây dựng một API Service **tinh gọn**, sử dụng `asyncio` để tối đa hóa tốc độ phản hồi cho một request.
3.  **Lõi cạnh tranh (Phần 2 - Re-rank):** Dồn tài nguyên phát triển vào thuật toán **re-ranking**, vì đây là yếu tố quyết định sự khác biệt về chất lượng của các kết quả trả về.

Kiến trúc này loại bỏ sự phức tạp không cần thiết của microservices và Go, giúp bạn tập trung thời gian và công sức vào việc cải thiện mô hình và thuật toán cốt lõi.
