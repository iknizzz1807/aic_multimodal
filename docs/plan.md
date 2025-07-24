Kiến trúc mới sẽ tập trung vào 3 phần chính:

1.  **Pipeline xử lý offline nâng cao.**
2.  **API Service hợp nhất, bất đồng bộ.**
3.  **Logic tìm kiếm và tái xếp hạng chuyên sâu.**

---

### **Tổng quan Kế hoạch Triển khai (Đã điều chỉnh cho cuộc thi)**

Kế hoạch này tập trung vào việc xây dựng một hệ thống tìm kiếm đa phương thức hiệu năng cao, tối ưu cho tốc độ phản hồi và độ chính xác của một request duy nhất.

---

### **Giai đoạn 1: Xây dựng Pipeline Xử lý Dữ liệu Nâng cao**

**Mục tiêu:** Thực hiện một lần duy nhất (offline) để biến đổi dữ liệu thô thành các chỉ mục (index) thông minh, giàu thông tin, sẵn sàng cho việc truy vấn tốc độ cao. Đây là nền tảng quan trọng nhất cho độ chính xác.

#### **Các bước thực hiện:**

1.  **Thiết lập Cơ sở dữ liệu:**

    - Sử dụng Docker để khởi chạy **Milvus** (Vector DB) và **Elasticsearch** (Text Search Engine).
    - Trong code, định nghĩa schema cho các collection/index:
      - Milvus `visual_collection`: Lưu vector hình ảnh, có các trường `media_id`, `timestamp`, `vector`, và các trường metadata nâng cao như `quality_score`.
      - **(Nâng cao)** Milvus `audio_events_collection`: Lưu vector sự kiện âm thanh.
      - Elasticsearch `transcripts_index`: Lưu lời thoại đã bóc tách, có các trường `media_id`, `start`, `end`, `text`.

2.  **Xây dựng Script Xử lý Song song (`processing/run_batch_processing.py`):**

    - Sử dụng `multiprocessing.Pool` để phân phát công việc xử lý từng file media cho các tiến trình con, tận dụng tối đa CPU.

3.  **Triển khai Worker Xử lý Sâu (`processing/worker.py`):**
    - Mỗi worker là một tiến trình độc lập, tự tải model và kết nối DB.
    - **Logic Visual Nâng cao:**
      - Sử dụng `PySceneDetect` để trích xuất các keyframe quan trọng thay vì trích xuất đều đặn.
      - Tính toán các đặc trưng chất lượng (độ nét, độ sáng) cho mỗi frame.
      - Tạo vector embedding bằng CLIP.
      - **Ghi trực tiếp** vector và tất cả metadata vào **Milvus**.
    - **Logic Audio Nâng cao:**
      - Sử dụng Whisper để bóc tách lời thoại.
      - **Ghi trực tiếp** các segment lời thoại vào **Elasticsearch** thông qua `bulk` API.
      - **(Tùy chọn)** Sử dụng model CLAP để tạo vector cho các sự kiện âm thanh và ghi vào Milvus.

**Kết quả Giai đoạn 1:** Toàn bộ dữ liệu được xử lý và nằm sẵn trong các DB chuyên dụng. Các file `.faiss`, `.json` và các script xử lý tuần tự được loại bỏ hoàn toàn.

---

### **Giai đoạn 2: Xây dựng API Service Hợp nhất (Thay thế Giai đoạn 2 & 3 cũ)**

**Mục tiêu:** Xây dựng một API Service duy nhất bằng FastAPI, có khả năng xử lý toàn bộ luồng tìm kiếm từ đầu đến cuối một cách hiệu quả, sử dụng lập trình bất đồng bộ.

#### **Các bước thực hiện:**

1.  **Tái cấu trúc `services/search_service.py`:**

    - Chuyển đổi các hàm tìm kiếm (`perform_visual_search`, `perform_audio_search`) thành các hàm `async`.
    - Trong `__init__`, tải tất cả các model cần thiết cho cả việc **recall** (CLIP) và **rerank** (Cross-Encoder) một lần duy nhất khi server khởi động.

2.  **Triển khai Tìm kiếm Bất đồng bộ trong `perform_unified_search`:**

    - Hàm này sẽ là một hàm `async`.
    - **Bước Recall:**
      - Tạo vector cho query của người dùng.
      - Tạo các coroutine task để gọi đến Milvus và Elasticsearch.
      - Sử dụng `asyncio.gather` để thực thi các task này **cùng một lúc**.
    - **Bước Fusion:**
      - Sau khi có kết quả từ các nguồn, sử dụng thuật toán RRF (`_fuse_results_rrf`) để kết hợp chúng thành một danh sách ứng viên (ví dụ: top 100).

3.  **Tích hợp Logic Re-ranking (Phần quan trọng nhất):**

    - Từ danh sách top 100 ứng viên, chuẩn bị dữ liệu (cặp `query` và `media_item`).
    - Sử dụng model **Cross-Encoder** đã tải sẵn để chấm điểm lại độ tương quan cho 100 ứng viên này. Để tránh block event loop, tác vụ tính toán này có thể được chạy trong một luồng riêng bằng `asyncio.to_thread`.
    - Kết hợp điểm RRF ban đầu và điểm từ Cross-Encoder để ra điểm số cuối cùng.
    - Sắp xếp lại lần cuối và trả về top K kết quả.

4.  **Cập nhật `api/api_router.py` và `main_api.py`:**
    - Đảm bảo endpoint `/search` là một hàm `async` và `await` kết quả từ `search_service`.

**Kết quả Giai đoạn 2:** Có một API duy nhất, hiệu năng cao, tối ưu cho việc xử lý một request, với luồng "Recall -> Fusion -> Rank" rõ ràng.

---

### **Giai đoạn 3: Phân Tích Chuyên Sâu và Tối ưu hóa (Trước là Giai đoạn 4)**

**Mục tiêu:** Thêm các tính năng AI nâng cao và tinh chỉnh hệ thống để đạt độ chính xác tối đa.

#### **Các bước thực hiện:**

1.  **Tạo Endpoint Phân tích `/analyze` (Tùy chọn):**

    - Tạo một endpoint mới trong `api_router.py`.
    - Dựa trên yêu cầu (`task: "qa"` hoặc `task: "summarize"`), gọi đến các hàm xử lý tương ứng trong `SearchService`.
    - Các hàm này sẽ sử dụng các model "hạng nặng" khác như LVLM (LLaVA) hoặc LLM chuyên tóm tắt.

2.  **Tinh chỉnh và Tối ưu hóa:**
    - **Trọng số Fusion:** Tinh chỉnh hằng số `k` trong RRF hoặc thêm trọng số cho từng modality (`visual_score`, `text_score`).
    - **Trọng số Re-ranking:** Tinh chỉnh trọng số khi kết hợp điểm RRF và điểm Cross-Encoder (`final_score = w1 * rrf_score + w2 * cross_encoder_score`).
    - **Tối ưu Model:** Sử dụng các kỹ thuật như ONNX, quantization để tăng tốc độ inference của các model AI.
    - **Tối ưu Truy vấn DB:** Tinh chỉnh các tham số index của Milvus (`nlist`, `efSearch`) và xây dựng các câu query Elasticsearch phức tạp hơn (sử dụng `bool`, `should` để kết hợp nhiều điều kiện).

---

---

### **FILE ĐÃ CHỈNH SỬA: `high_implementation_revised.md`**

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

### **Phần 1: Pipeline Xử Lý Dữ Liệu Nâng cao (Offline)**

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

### **Phần 2: API Service Tìm kiếm Hợp nhất và Tái xếp hạng (Online)**

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

### **Phần 3: Phân Tích Chuyên Sâu và Tối ưu hóa (Online)**

**Mục tiêu:** Cung cấp các khả năng tương tác sâu hơn và tinh chỉnh hệ thống để đạt hiệu năng tối đa.

#### **Các hoạt động:**

1.  **Endpoint Phân tích `/analyze` (Tùy chọn):**

    - Tạo một endpoint mới để thực hiện các tác vụ như Visual Question Answering (dùng LVLM) hoặc tóm tắt video (dùng LLM). Đây là tính năng nâng cao để thể hiện sự vượt trội, nếu thời gian cho phép.

2.  **Tinh chỉnh và Tối ưu hóa (Quan trọng):**
    - **Trọng số:** Tinh chỉnh các trọng số trong thuật toán fusion (RRF) và re-ranking để phản ánh đúng tầm quan trọng của từng loại điểm số.
    - **Tối ưu Model:** Sử dụng các kỹ thuật như ONNX, `torch.compile`, quantization để giảm độ trễ của các model AI.
    - **Tối ưu Truy vấn DB:** Tinh chỉnh các tham số index của Milvus (`nlist`, `efSearch`) và xây dựng các câu query Elasticsearch phức tạp hơn để tăng độ chính xác của bước recall.
