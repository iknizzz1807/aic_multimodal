### Sơ đồ kiến trúc tổng thể

+-----------------------------------------------------------------------------------+
| User Interface (Web App) / Automated Testing System |
+---------------------------------+-------------------------------------------------+
| (HTTP/WebSocket Requests)
+---------------------------------v-------------------------------------------------+
| API GATEWAY & ORCHESTRATOR (Go) - "Bộ Não Trung Tâm" |
| - Quản lý kết nối, Caching (Redis), Điều phối tác vụ bất đồng bộ |
| - Module 1: LLM-Powered Query Orchestrator (Phân rã yêu cầu -> Kế hoạch) |
| - Module 2: Fusion & Re-ranking Engine (Kết hợp & Tái xếp hạng) |
+------+----------------+-------------------+------------------+-------------------+
| (gRPC Calls) | (gRPC Calls) | (gRPC Calls) | (gRPC Calls) |
+------v---------+ +----v-------------+ +---v--------------+ +---v-----------------+
| VISUAL SERVICE | | TEXT SERVICE | | AUDIO SERVICE | | QA & ANALYSIS |
| (Python/ONNX) | | (Python) | | (Python/ONNX) | | SERVICE (Python) |
+----------------+ +------------------+ +------------------+ +-------------------+
| (Queries) | (Queries) | (Queries) | (Media + Question)
+------v---------+ +----v-------------+ +---v--------------+ +---v-----------------+
| VECTOR DB | | TEXT SEARCH | | VECTOR DB | | LLM / LVLM |
| (Milvus/FAISS) | | ENGINE | | (Milvus/FAISS) | | PROVIDER (API) |
| - Visual Vectors | | (Elasticsearch) | | - Audio Vectors | | (GPT-4V, LLaVA) |
+----------------+ +------------------+ +------------------+ +-------------------+

### **Kiến Trúc End-Product Chi Tiết (Phần 1/4): Pipeline Xử Lý Dữ Liệu Bất Đồng Bộ**

**Mục tiêu của Phần 1:** Thiết kế một hệ thống **offline** mạnh mẽ, có khả năng mở rộng và tự động, để biến đổi dữ liệu multimedia thô (video, ảnh) thành các "tài sản số" có cấu trúc, được lập chỉ mục và sẵn sàng để truy vấn với tốc độ cực nhanh.

**Triết lý thiết kế:** Thay vì một script lớn chạy tuần tự, chúng ta sử dụng kiến trúc **Microservices** giao tiếp qua **Message Queue** (như RabbitMQ/Kafka). Điều này giúp:

- **Song song hóa:** Nhiều video có thể được xử lý cùng lúc.
- **Khả năng phục hồi:** Nếu một worker bị lỗi, tin nhắn không bị mất và có thể được xử lý lại.
- **Khả năng mở rộng:** Có thể dễ dàng tăng số lượng worker cho các tác vụ nặng.

---

#### **Sơ Đồ Luồng Xử Lý Dữ Liệu Offline (Mã Giả)**

```pseudocode
// ===================================================================
// I. CÁC THÀNH PHẦN CHÍNH CỦA PIPELINE
// ===================================================================

// Message Queues (Hàng đợi tin nhắn) được định nghĩa:
// - "queue.media.new": Chứa tin nhắn về các file media mới cần xử lý.
// - "queue.process.visual": Tác vụ xử lý hình ảnh cho một file.
// - "queue.process.audio": Tác vụ xử lý âm thanh cho một file.
// - "queue.index.visual": Dữ liệu hình ảnh đã xử lý, sẵn sàng để index.
// - "queue.index.text": Dữ liệu transcript đã xử lý, sẵn sàng để index.
// - "queue.index.audio_event": Dữ liệu sự kiện âm thanh đã xử lý, sẵn sàng để index.

// ===================================================================
// II. CÁC WORKER TRONG PIPELINE
// ===================================================================

// --- Worker 1: Ingestion & Dispatcher (Go hoặc Python) ---
// Nhiệm vụ: Theo dõi dữ liệu mới và phân phát công việc.
MODULE IngestionAndDispatcher:

    FUNCTION WatchNewFilesAndDispatch():
        WHILE TRUE:
            new_files = ListenToNewFilesIn(CONFIG.DATA_FOLDER)
            FOR EACH file_path IN new_files:
                media_id = GenerateUniqueID(file_path) // Tạo ID duy nhất cho file

                // Phân rã công việc và đẩy vào các hàng đợi chuyên biệt
                message_body = { "media_id": media_id, "path": file_path }
                SendMessageToQueue("queue.process.visual", message_body)
                SendMessageToQueue("queue.process.audio", message_body)

                Log(f"Dispatched tasks for media_id: {media_id}")
            Sleep(30_seconds)

// --- Worker 2: Visual Processor (Python) ---
// Nhiệm vụ: "Tiêu hóa" phần hình ảnh của media.
MODULE VisualProcessorWorker:

    // Tải các mô hình đã được tối ưu hóa (vd: qua ONNX/TensorRT)
    MODELS.CLIP = LoadOptimizedModel("clip-vit-L-14/ONNX")
    MODELS.YOLO = LoadOptimizedModel("yolov8/ONNX")
    MODELS.SceneDetector = LoadSceneDetectorModel()

    FUNCTION ProcessVisualMessage(message):
        media_id = message.media_id
        path = message.path

        // 1. Trích xuất Keyframes thông minh bằng Scene Detection
        keyframes = MODELS.SceneDetector.ExtractKeyframes(path)

        // 2. Xử lý từng keyframe để tạo biểu diễn đa mức
        processed_visual_data = NewList()
        FOR EACH frame IN keyframes:
            // 2a. Tính toán đặc trưng chất lượng (để dùng khi re-ranking)
            quality_features = {
                "sharpness": CalculateLaplacianVariance(frame.image),
                "brightness": CalculateBrightness(frame.image)
            }

            // 2b. Vector Toàn Cảnh (Global Vector)
            global_vector = MODELS.CLIP.GetImageVector(frame.image)

            // 2c. Vector Đối Tượng (Object-level Vectors)
            object_vectors = NewList()
            detected_objects = MODELS.YOLO.Detect(frame.image)
            FOR EACH obj IN detected_objects:
                cropped_image = Crop(frame.image, obj.bounding_box)
                object_vector = MODELS.CLIP.GetImageVector(cropped_image)
                object_vectors.Add({
                    "label": obj.label,
                    "confidence": obj.confidence,
                    "vector": object_vector
                })

            // 2d. Gom gói dữ liệu của frame
            processed_visual_data.Add({
                "timestamp": frame.timestamp,
                "quality": quality_features,
                "global_vector": global_vector,
                "object_vectors": object_vectors
            })

        // 3. Gửi kết quả cuối cùng cho worker Indexing
        SendMessageToQueue("queue.index.visual", {
            "media_id": media_id,
            "visual_data": processed_visual_data
        })

// --- Worker 3: Audio Processor (Python) ---
// Nhiệm vụ: "Tiêu hóa" phần âm thanh của media.
MODULE AudioProcessorWorker:
    // Sử dụng model ASR đã được fine-tune cho tiếng Việt
    MODELS.ASR_Vietnamese = LoadFineTunedModel("whisper-large-v3-vietnamese")
    MODELS.CLAP = LoadOptimizedModel("clap-model/ONNX") // Model cho sự kiện âm thanh

    FUNCTION ProcessAudioMessage(message):
        media_id = message.media_id
        path = message.path

        // Chạy song song 2 tác vụ nặng để tiết kiệm thời gian
        // 1. Nhận dạng giọng nói (ASR)
        TASK asr_task = RunInBackground(MODELS.ASR_Vietnamese.TranscribeWithTimestamps, path)

        // 2. Nhận dạng sự kiện âm thanh
        TASK event_task = RunInBackground(ExtractAudioEvents, path, MODELS.CLAP)

        // Đợi kết quả
        transcript_data = WaitFor(asr_task)
        audio_event_data = WaitFor(event_task)

        // Gửi kết quả cho các worker Indexing tương ứng
        SendMessageToQueue("queue.index.text", { "media_id": media_id, "data": transcript_data })
        SendMessageToQueue("queue.index.audio_event", { "media_id": media_id, "data": audio_event_data })

// --- Worker 4, 5, 6: Indexing Workers (Python) ---
// Các worker này chỉ có nhiệm vụ ghi dữ liệu đã xử lý vào các DB chuyên dụng.
MODULE VisualIndexer:
    DB.VECTOR_DB = ConnectToVectorDB_Milvus()

    FUNCTION IndexVisualData(message):
        // Ghi dữ liệu vào Vector DB với cấu trúc phức tạp
        // Mỗi frame là một entry, có vector toàn cảnh, và metadata chứa các vector đối tượng
        DB.VECTOR_DB.BatchInsert("visual_collection", message.visual_data, message.media_id)

MODULE TextIndexer:
    DB.ELASTICSEARCH = ConnectToElasticsearch()

    FUNCTION IndexTextData(message):
        // Chuẩn bị document để ghi vào Elasticsearch
        document = { "media_id": message.media_id, "segments": message.data }
        DB.ELASTICSEARCH.IndexDocument("transcripts_index", document)

MODULE AudioEventIndexer:
    DB.VECTOR_DB = ConnectToVectorDB_Milvus() // Có thể dùng chung DB nhưng khác collection

    FUNCTION IndexAudioEventData(message):
        DB.VECTOR_DB.BatchInsert("audio_events_collection", message.data, message.media_id)

```

#### **Tổng kết luồng hoạt động của Phần 1:**

1.  Một file video mới được thêm vào thư mục `data`.
2.  `IngestionAndDispatcher` phát hiện và gửi 2 tin nhắn vào `queue.process.visual` và `queue.process.audio`.
3.  `VisualProcessorWorker` nhận tin nhắn, trích xuất keyframes, tạo vector toàn cảnh, vector đối tượng, tính toán điểm chất lượng và gửi một gói dữ liệu lớn vào `queue.index.visual`.
4.  Đồng thời, `AudioProcessorWorker` nhận tin nhắn, chạy ASR và nhận dạng sự kiện âm thanh, rồi gửi 2 gói dữ liệu vào `queue.index.text` và `queue.index.audio_event`.
5.  Ba `IndexingWorker` cuối cùng nhận dữ liệu đã được xử lý sạch sẽ và chỉ việc ghi chúng vào các cơ sở dữ liệu chuyên dụng (Milvus, Elasticsearch).

Kết thúc Phần 1, toàn bộ kho dữ liệu thô đã được "số hóa" và sẵn sàng cho các truy vấn trong **Phần 2: Các Microservices Tìm Kiếm Chuyên Biệt**.

### **Kiến Trúc End-Product Chi Tiết (Phần 2/4): Các Microservices Tìm Kiếm Chuyên Biệt**

**Mục tiêu của Phần 2:** Thiết kế các dịch vụ backend độc lập, hiệu năng cao, mỗi dịch vụ chịu trách nhiệm cho một phương thức tìm kiếm (visual, text, audio event). Chúng sẽ giao tiếp nội bộ qua gRPC để đạt được độ trễ thấp nhất.

**Triết lý thiết kế:**

- **Single Responsibility Principle:** Mỗi service chỉ làm một việc và làm thật tốt. `VisualSearchService` không biết gì về ASR, và ngược lại.
- **Tối ưu hóa chuyên sâu:** Mỗi service sử dụng cơ sở dữ liệu và mô hình phù hợp nhất cho nhiệm vụ của nó (Vector DB cho tìm kiếm tương đồng, Text Search Engine cho tìm kiếm văn bản).
- **Giao tiếp hiệu quả:** Sử dụng gRPC và Protocol Buffers để định nghĩa các request/response một cách chặt chẽ và truyền dữ liệu nhị phân hiệu quả.

---

#### **Sơ Đồ Các Microservices (Mã Giả)**

```pseudocode
// ===================================================================
// I. ĐỊNH NGHĨA GIAO THỨC CHUNG (gRPC .proto file)
// ===================================================================

// File: search.proto
// Định nghĩa các cấu trúc dữ liệu và service signatures

message SearchRequest {
    string query_text = 1;      // Truy vấn bằng văn bản
    bytes query_image = 2;      // Truy vấn bằng hình ảnh (bytes của ảnh)
    int32 top_k = 3;            // Số lượng kết quả cần lấy
}

message MediaResult {
    string media_id = 1;        // ID của media gốc
    string media_path = 2;      // Đường dẫn đến file
    float timestamp = 3;        // Dấu thời gian (nếu là video frame/audio segment)
    float score = 4;            // Điểm số từ lần tìm kiếm ban đầu
    map<string, float> metadata = 5; // Các thông tin phụ như quality_score, v.v.
}

message SearchResponse {
    repeated MediaResult results = 1;
}

// Định nghĩa các service và các hàm RPC của chúng
service VisualSearch {
    rpc Search(SearchRequest) returns (SearchResponse);
}

service TextSearch {
    rpc Search(SearchRequest) returns (SearchResponse);
}

service AudioEventSearch {
    rpc Search(SearchRequest) returns (SearchResponse);
}


// ===================================================================
// II. TRIỂN KHAI CÁC MICROSERVICES (Python)
// ===================================================================

// --- Service 1: Visual Search Service ---
// Nhiệm vụ: Tìm kiếm hình ảnh/khung hình dựa trên sự tương đồng về nội dung.
MODULE VisualSearchService:
    // Tải tài nguyên khi khởi động
    MODELS.CLIP = LoadOptimizedModel("clip-vit-L-14/ONNX")
    DB.VECTOR_DB = ConnectToVectorDB_Milvus("visual_collection")

    // Triển khai hàm RPC "Search"
    RPC FUNCTION Search(request: SearchRequest) -> SearchResponse:
        // 1. Vector hóa truy vấn của người dùng
        IF request.has_text_query:
            query_vector = MODELS.CLIP.GetTextVector(request.text_query)
        ELSE IF request.has_image_query:
            query_vector = MODELS.CLIP.GetImageVector(request.query_image)
        ELSE:
            RAISE Error("Truy vấn không hợp lệ")

        // 2. Xây dựng truy vấn tìm kiếm vector nâng cao
        // Tận dụng các tính năng của Vector DB để tăng tốc và độ chính xác
        search_params = {
            "metric_type": "IP", // Inner Product, tương đương Cosine Similarity với vector đã chuẩn hóa
            "params": { "efSearch": 128 } // Tham số cho index HNSW
        }

        // Pre-filtering: Lọc trước khi tìm kiếm vector, cực kỳ hiệu quả
        // Chỉ tìm kiếm trên các frame có chất lượng tốt
        filter_expression = "quality.sharpness > 0.6"

        // 3. Thực hiện tìm kiếm
        // Tìm kiếm trên các vector toàn cảnh (global vectors)
        raw_results = DB.VECTOR_DB.Search(
            query_vectors=[query_vector],
            limit=request.top_k,
            params=search_params,
            filter=filter_expression
        )
        // Lưu ý: Có thể có logic tìm kiếm trên object_vectors và kết hợp ở đây nếu cần.

        // 4. Định dạng lại kết quả và trả về
        response = FormatResultsToProto(raw_results)
        RETURN response

// --- Service 2: Text Search Service ---
// Nhiệm vụ: Tìm kiếm trong các transcript đã được nhận dạng.
MODULE TextSearchService:
    // Tải tài nguyên khi khởi động
    DB.ELASTICSEARCH = ConnectToElasticsearch("transcripts_index")

    // Triển khai hàm RPC "Search"
    RPC FUNCTION Search(request: SearchRequest) -> SearchResponse:
        // 1. Xây dựng truy vấn Elasticsearch phức tạp
        // Không chỉ là tìm kiếm chuỗi đơn thuần, mà tận dụng sức mạnh của ES
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        { "match_phrase": { "segments.text": request.text_query } } // Tìm chính xác cụm từ
                    ],
                    "should": [
                        { "match": { "segments.text": { "query": request.text_query, "fuzziness": "AUTO" } } } // Hỗ trợ lỗi chính tả
                    ]
                }
            },
            "highlight": { "fields": { "segments.text": {} } } // Đánh dấu từ khóa khớp
        }

        // 2. Thực hiện tìm kiếm
        raw_results = DB.ELASTICSEARCH.Search(query=es_query, size=request.top_k)

        // 3. Định dạng lại kết quả
        // Trích xuất media_id, đoạn text khớp, timestamp, và điểm số từ ES
        response = FormatResultsToProto(raw_results)
        RETURN response

// --- Service 3: Audio Event Search Service ---
// Nhiệm vụ: Tìm kiếm các sự kiện âm thanh (tiếng vỗ tay, tiếng chó sủa,...).
MODULE AudioEventSearchService:
    // Tải tài nguyên khi khởi động
    MODELS.CLAP = LoadOptimizedModel("clap-model/ONNX")
    DB.VECTOR_DB = ConnectToVectorDB_Milvus("audio_events_collection")

    // Triển khai hàm RPC "Search"
    RPC FUNCTION Search(request: SearchRequest) -> SearchResponse:
        // 1. Vector hóa truy vấn text (ví dụ: "dog barking") bằng text encoder của CLAP
        query_vector = MODELS.CLAP.GetTextVector(request.text_query)

        // 2. Thực hiện tìm kiếm trong Vector DB
        search_params = { "metric_type": "COSINE", "params": { "efSearch": 128 } }
        raw_results = DB.VECTOR_DB.Search(
            query_vectors=[query_vector],
            limit=request.top_k,
            params=search_params
        )

        // 3. Định dạng lại kết quả và trả về
        response = FormatResultsToProto(raw_results)
        RETURN response
```

#### **Tổng kết luồng hoạt động của Phần 2:**

- Khi hệ thống khởi động, mỗi microservice sẽ tải các mô hình và kết nối cơ sở dữ liệu cần thiết vào RAM, sẵn sàng phục vụ.
- Các service này sẽ "lắng nghe" các lệnh gọi gRPC từ một nơi khác (chính là API Gateway sẽ được mô tả ở Phần 3).
- Chúng thực hiện một nhiệm vụ duy nhất: **thu hồi (recall)**. Tức là, chúng có trách nhiệm tìm ra một danh sách các ứng viên **tiềm năng** từ hàng triệu bản ghi một cách nhanh nhất có thể.
- Kết quả trả về từ các service này là "nguyên liệu thô" cho bước xử lý tiếp theo. Chúng đã được xếp hạng sơ bộ, nhưng chưa phải là kết quả cuối cùng cho người dùng.

Kết thúc Phần 2, chúng ta đã có một bộ "chuyên gia" mạnh mẽ. Bước tiếp theo trong **Phần 3: Bộ Não Điều Phối và API Gateway** sẽ là cách chúng ta chỉ huy các chuyên gia này để tạo ra một bản giao hưởng hoàn chỉnh.

Phần 3 này mô tả thành phần quan trọng nhất của hệ thống online: **API Gateway và Bộ Điều Phối (Orchestrator)**. Đây là nơi phép màu thực sự xảy ra, biến các yêu cầu mơ hồ của người dùng thành các kết quả chính xác và có liên quan.

---

### **Kiến Trúc End-Product Chi Tiết (Phần 3/4): Bộ Não Điều Phối và API Gateway**

**Mục tiêu của Phần 3:** Thiết kế một service trung tâm (viết bằng Go để có hiệu năng I/O cao) có 3 trách nhiệm chính:

1.  **Tiếp nhận và Hiểu (Receive & Understand):** Là điểm cuối (endpoint) duy nhất cho người dùng, và sử dụng một Large Language Model (LLM) để phân tích ý định phức tạp của người dùng.
2.  **Điều phối và Thực thi (Orchestrate & Execute):** Gọi song song các microservices chuyên biệt (đã xây dựng ở Phần 2) dựa trên kế hoạch được LLM đề ra.
3.  **Tổng hợp và Trả lời (Synthesize & Respond):** Nhận lại các kết quả "thô", thực hiện một thuật toán kết hợp (fusion) thông minh, và trả về một câu trả lời cuối cùng cho người dùng.

**Triết lý thiết kế:**

- **Tách biệt vai trò:** Logic AI nặng (tính toán vector) nằm trong các services Python. Logic điều phối, xử lý mạng, và kết hợp dữ liệu nằm trong Go Gateway, tận dụng thế mạnh của từng ngôn ngữ.
- **Thông minh hóa truy vấn:** Không chỉ tìm kiếm từ khóa, mà dùng LLM để phân rã một câu ngôn ngữ tự nhiên thành các tác vụ cụ thể (Tool/Function Calling).
- **Bất đồng bộ toàn diện:** Mọi lệnh gọi đến các services backend đều được thực hiện song song để giảm thiểu thời gian chờ đợi của người dùng.

---

#### **Sơ Đồ API Gateway và Bộ Điều Phối (Mã Giả - Viết bằng Go)**

```pseudocode
// ===================================================================
// I. CẤU TRÚC VÀ CÁC THÀNH PHẦN CỦA GATEWAY (Go)
// ===================================================================

// --- Service: API_Gateway ---
PACKAGE main

// Khai báo các client gRPC để giao tiếp với các services Python
gRPC_CLIENTS {
    VisualSearcher: ConnectTo_gRPC_Service("visual-search-service:50051"),
    TextSearcher:   ConnectTo_gRPC_Service("text-search-service:50052"),
    AudioSearcher:  ConnectTo_gRPC_Service("audio-event-search-service:50053")
    // (Tương lai có thể thêm QA_Service_Client)
}

// Khởi tạo các module cốt lõi của Gateway
MODULES {
    QueryOrchestrator: NewQueryOrchestrator(LLM_API_KEY),
    FusionEngine:      NewFusionEngine(RRF_K_CONSTANT = 60),
    Cache:             NewRedisCache("redis:6379")
}

// ===================================================================
// II. CÁC MODULE LOGIC BÊN TRONG GATEWAY
// ===================================================================

// --- Module 1: Query Orchestrator - Bộ não phân tích yêu cầu ---
MODULE QueryOrchestrator:

    FUNCTION DecomposeQueryToExecutionPlan(user_query: string) -> ExecutionPlan:
        // 1. Kiểm tra cache trước để tránh gọi LLM không cần thiết
        cached_plan = MODULES.Cache.Get("plan:" + hash(user_query))
        IF cached_plan IS NOT NULL:
            RETURN cached_plan

        // 2. Định nghĩa các "công cụ" mà LLM có thể sử dụng
        available_tools = [
            { "name": "search_visual", "description": "Tìm kiếm hình ảnh hoặc video dựa trên mô tả cảnh vật, hành động, đối tượng." },
            { "name": "search_spoken_text", "description": "Tìm kiếm video dựa trên nội dung lời nói, trích dẫn chính xác." },
            { "name": "search_audio_event", "description": "Tìm kiếm âm thanh không phải lời nói như tiếng vỗ tay, nhạc, tiếng chó sủa." }
        ]

        // 3. Xây dựng prompt theo kỹ thuật Tool/Function Calling
        prompt = BuildFunctionCallingPrompt(user_query, available_tools)

        // 4. Gọi LLM API (GPT-4, Llama 3, Gemini,...)
        llm_response_json = Call_LLM_API(prompt)

        // 5. Phân tích JSON trả về để tạo ra kế hoạch thực thi
        execution_plan = ParseLLMResponseToPlan(llm_response_json)

        // 6. Lưu kế hoạch vào cache
        MODULES.Cache.Set("plan:" + hash(user_query), execution_plan, expiry_time=1_hour)

        RETURN execution_plan

// --- Module 2: Fusion Engine - Bộ máy kết hợp kết quả ---
MODULE FusionEngine:

    FUNCTION FuseRankedLists(ranked_lists: map[string, SearchResponse]) -> list[FinalResult]:
        final_scores = NewMap<string, float64>() // Map: media_id -> final_rrf_score

        // Lặp qua kết quả từ mỗi loại tìm kiếm (visual, text, audio)
        FOR EACH modality_name, response IN ranked_lists:
            FOR rank, result IN enumerate(response.results):
                media_id = result.media_id

                // Áp dụng công thức Reciprocal Rank Fusion (RRF)
                rrf_score = 1.0 / (CONFIG.RRF_K_CONSTANT + (rank + 1))

                // (Tối ưu hóa) Có thể áp dụng trọng số ở đây nếu cần
                // weighted_score = rrf_score * GetWeightForModality(modality_name)

                final_scores[media_id] = final_scores.Get(media_id, 0.0) + rrf_score

        // Sắp xếp map theo điểm số để có danh sách cuối cùng
        sorted_results = SortMapByValue(final_scores, descending=true)

        RETURN sorted_results

// ===================================================================
// III. HTTP ENDPOINT CHÍNH (Tiếp nhận request từ người dùng)
// ===================================================================

HTTP_ENDPOINT POST "/search" (request: http.Request):
    // 1. Phân tích request body từ người dùng
    user_query = ParseJSON(request.body).query
    top_k = ParseJSON(request.body).top_k

    // 2. Dùng Orchestrator để lấy kế hoạch thực thi
    plan = MODULES.QueryOrchestrator.DecomposeQueryToExecutionPlan(user_query)

    // 3. Thực thi kế hoạch - Gọi các gRPC services song song
    // Sử dụng goroutines và channels của Go để xử lý bất đồng bộ

    // Channel để thu thập kết quả
    results_channel = make(chan map[string, SearchResponse])

    // WaitGroup để đợi tất cả các goroutine hoàn thành
    var wg sync.WaitGroup

    FOR EACH tool_call IN plan.tool_calls:
        wg.Add(1)
        GO FUNCTION (call): // Chạy trong một goroutine riêng
            DEFER wg.Done()

            var response SearchResponse
            CASE call.name:
                WHEN "search_visual":
                    response = gRPC_CLIENTS.VisualSearcher.Search(query_text=call.query)
                WHEN "search_spoken_text":
                    response = gRPC_CLIENTS.TextSearcher.Search(query_text=call.query)
                WHEN "search_audio_event":
                    response = gRPC_CLIENTS.AudioSearcher.Search(query_text=call.query)

            results_channel <- map[string, SearchResponse>{call.name: response}
        (tool_call) // Truyền tool_call vào goroutine

    // Chạy một goroutine để đóng channel sau khi tất cả các task hoàn thành
    GO FUNCTION():
        wg.Wait()
        close(results_channel)
    ()

    // 4. Thu thập tất cả các kết quả từ các services
    all_search_results = NewMap<string, SearchResponse>()
    FOR result_map IN <-results_channel:
        FOR key, value IN result_map:
            all_search_results[key] = value

    // 5. Dùng Fusion Engine để kết hợp và xếp hạng
    fused_results = MODULES.FusionEngine.FuseRankedLists(all_search_results)

    // 6. Trả về top K kết quả cuối cùng cho người dùng
    final_response_body = fused_results.Slice(0, top_k)
    RETURN HTTP_RESPONSE(status=200, body=final_response_body)

```

#### **Tổng kết luồng hoạt động của Phần 3:**

1.  Một request HTTP chứa câu truy vấn tự nhiên của người dùng (ví dụ: "video tôi ở bãi biển nói về AI") được gửi đến API Gateway (Go).
2.  Gateway dùng `QueryOrchestrator` để gửi câu truy vấn này cho một LLM.
3.  LLM phân tích và trả về một "kế hoạch" dạng JSON, ví dụ: `[ {name: "search_visual", query: "beach scene"}, {name: "search_spoken_text", query: "AI"} ]`.
4.  Gateway đọc kế hoạch này và tạo ra 2 goroutine: một để gọi `VisualSearchService`, một để gọi `TextSearchService` qua gRPC. Cả hai chạy **cùng một lúc**.
5.  Gateway chờ cả hai service trả về kết quả (đã được xếp hạng sơ bộ).
6.  Gateway đưa 2 danh sách kết quả này vào `FusionEngine`. Engine này dùng thuật toán RRF để tính một điểm số tổng hợp duy nhất cho mỗi media, ưu tiên các media xuất hiện ở top đầu của cả hai danh sách.
7.  Cuối cùng, Gateway trả về một danh sách kết quả duy nhất, đã được xếp hạng thông minh, cho người dùng.

Kết thúc Phần 3, chúng ta đã có một hệ thống tìm kiếm đa phương thức hoàn chỉnh. Phần cuối cùng, **Phần 4: Tầng Tái Xếp Hạng Nâng Cao và Phân Tích Chuyên Sâu**, sẽ bổ sung "vũ khí tối thượng" để hệ thống thực sự vượt trội.

### **Kiến Trúc End-Product Chi Tiết (Phần 4/4): Tầng Tái Xếp Hạng Nâng Cao và Phân Tích Chuyên Sâu**

**Mục tiêu của Phần 4:** Thiết kế các module nâng cao hoạt động sau khi đã có kết quả tìm kiếm sơ bộ. Các module này sẽ:

1.  **Tái Xếp Hạng (Re-ranking):** Sắp xếp lại danh sách ứng viên (ví dụ top 100) từ bước Fusion, sử dụng các mô hình AI "đắt đỏ" hơn nhưng chính xác hơn để đảm bảo các kết quả ở top đầu là tốt nhất có thể.
2.  **Phân tích & Hỏi-Đáp (Analysis & QA):** Cung cấp khả năng tương tác sâu với một kết quả cụ thể, ví dụ như hỏi đáp về nội dung ảnh hoặc tóm tắt video.

**Triết lý thiết kế:**

- **Kiến trúc Recall-then-Rank:** Tách biệt rõ ràng giữa việc **thu hồi (Recall)** nhanh và rộng (Phần 2-3) và việc **xếp hạng (Rank)** chậm và sâu (Phần 4). Điều này giữ cho hệ thống vừa nhanh vừa chính xác.
- **Tương tác thông minh:** Tận dụng tối đa khả năng của các mô hình LLM/LVLM để tạo ra một trải nghiệm "trợ lý ảo" thực thụ, không chỉ là một công cụ tìm kiếm.
- **Tối ưu hóa chi phí:** Các mô hình tốn kém nhất (LVLM, Cross-Encoder) chỉ được gọi khi thực sự cần thiết và trên một tập dữ liệu rất nhỏ.

---

#### **Sơ Đồ Tầng Nâng Cao và Tích Hợp vào Luồng Chính (Mã Giả)**

```pseudocode
// ===================================================================
// I. CÁC MODULE VÀ SERVICES NÂNG CAO
// ===================================================================

// --- Service 4: QA & Analysis Service (Python, gRPC Server) ---
// Service này được thiết kế để xử lý các tác vụ phân tích sâu, tốn nhiều tài nguyên.
SERVICE AnalysisService:
    // Tải các mô hình "hạng nặng"
    MODELS.LVLM_QA = Load_LLaVA_or_GPT4V_Model()  // Model để hỏi đáp về ảnh/video
    MODELS.CROSS_ENCODER = Load_CrossEncoder_Model() // Model để tính độ tương đồng chính xác cao
    MODELS.SUMMARIZER_LLM = Load_LLM_Model("summarization-tuned-llama3") // Model chuyên tóm tắt

    // RPC 1: Tính điểm tương đồng chính xác cao cho việc re-ranking
    RPC FUNCTION RerankWithCrossEncoder(request: RerankRequest) -> RerankResponse:
        // request chứa: (text_query, list_of_media_items)
        scores = NewMap<string, float>()
        FOR EACH item IN request.media_items:
            // Cross-Encoder xử lý cả text và ảnh cùng lúc, cho điểm số rất chính xác
            score = MODELS.CROSS_ENCODER.Predict(text=request.text_query, image=item.thumbnail_bytes)
            scores[item.media_id] = score
        RETURN { "scores": scores }

    // RPC 2: Trả lời câu hỏi về một media cụ thể
    RPC FUNCTION AnswerQuestionOnMedia(request: QA_Request) -> QA_Response:
        // request chứa: (media_id, question)
        frame_data = GetFrameForMedia(request.media_id)
        answer = MODELS.LVLM_QA.Ask(image=frame_data, question=request.question)
        RETURN { "answer": answer }

    // RPC 3: Tóm tắt nội dung video
    RPC FUNCTION SummarizeVideo(request: SummarizeRequest) -> SummarizeResponse:
        // request chứa: (media_id)
        transcript = GetTranscriptForMedia(request.media_id)
        summary = MODELS.SUMMARIZER_LLM.Summarize(transcript)
        RETURN { "summary": summary }

// ===================================================================
// II. CẬP NHẬT LOGIC CỦA API GATEWAY (Go)
// ===================================================================

MODULE API_Gateway_Updated:

    // --- Module 2 được nâng cấp: Fusion & Re-ranking Engine ---
    MODULE AdvancedFusionEngine:

        FUNCTION FuseAndRerank(ranked_lists: map[string, SearchResponse], original_query: string) -> list[FinalResult]:
            // 1. Giai đoạn Fusion (Nhanh) - Giữ nguyên như Phần 3
            fused_list = ReciprocalRankFusion(ranked_lists)
            top_100_candidates = fused_list.Slice(0, 100)

            // --- 2. Giai đoạn Re-ranking (Mới & Sâu) ---
            // 2a. Chuẩn bị request cho Rerank service
            rerank_request = PrepareRerankRequest(original_query, top_100_candidates)

            // 2b. Gọi Cross-Encoder Service để lấy điểm số chính xác hơn
            rerank_response = gRPC_CLIENTS.AnalysisService.RerankWithCrossEncoder(rerank_request)
            cross_encoder_scores = rerank_response.scores

            // 2c. Kết hợp tất cả các điểm để ra điểm cuối cùng
            final_scores = NewMap<string, float>()
            FOR EACH candidate IN top_100_candidates:
                // Trọng số có thể được học bằng một mô hình Learning-to-Rank nhỏ
                // W1, W2, W3 là các trọng số được tinh chỉnh
                final_score = (W1 * candidate.fusion_score) +
                              (W2 * candidate.quality_score) +
                              (W3 * cross_encoder_scores[candidate.media_id])
                final_scores[candidate.media_id] = final_score

            // 3. Sắp xếp lại lần cuối cùng dựa trên điểm số tổng hợp
            final_list = SortMapByValue(final_scores, descending=true)

            RETURN final_list

    // --- Cập nhật Endpoint POST "/search" ---
    // Luồng thực thi từ 1-4 giữ nguyên như Phần 3
    // Bước 5 được cập nhật để gọi Engine nâng cao

    ENDPOINT POST "/search" (request):
        // ... (Các bước 1-4: Decompose, Execute, Collect) ...
        search_results = WaitForAll(search_tasks)

        // --- Bước 5 (Cập nhật) ---
        // Gọi AdvancedFusionEngine thay vì engine cũ
        final_results = MODULES.AdvancedFusionEngine.FuseAndRerank(search_results, user_query)

        // --- Bước 6 (Giữ nguyên) ---
        RETURN HTTP_RESPONSE(200, body=final_results.Slice(0, top_k))

    // --- Endpoint mới cho Tương tác Chuyên sâu ---
    ENDPOINT POST "/analyze" (request):
        // Request body chứa: { "task": "qa" | "summarize", "media_id": "...", "question": "..." }
        task_type = request.body.task

        CASE task_type:
            WHEN "qa":
                // Gọi gRPC đến Analysis Service
                response = gRPC_CLIENTS.AnalysisService.AnswerQuestionOnMedia(
                    media_id=request.body.media_id,
                    question=request.body.question
                )
                RETURN HTTP_RESPONSE(200, body=response)

            WHEN "summarize":
                // Gọi gRPC đến Analysis Service
                response = gRPC_CLIENTS.AnalysisService.SummarizeVideo(
                    media_id=request.body.media_id
                )
                RETURN HTTP_RESPONSE(200, body=response)

            DEFAULT:
                RETURN HTTP_RESPONSE(400, body={"error": "Invalid task type"})
```

#### **Tổng kết luồng hoạt động của sản phẩm cuối cùng:**

**Luồng 1: Tìm kiếm (Search)**

1.  Người dùng nhập một truy vấn phức tạp.
2.  **API Gateway (Go)** nhận request.
3.  **Orchestrator** dùng LLM để phân rã truy vấn thành một kế hoạch thực thi.
4.  Gateway gọi song song các **services chuyên biệt** (Visual, Text, Audio) qua gRPC để thu hồi (recall) một lượng lớn ứng viên (ví dụ, 3 danh sách top 50).
5.  **AdvancedFusionEngine** bên trong Gateway thực hiện:
    a. **Fusion:** Dùng RRF để hợp nhất 3 danh sách thành một danh sách top 100 ứng viên.
    b. **Re-ranking:** Gửi danh sách top 100 này đến **Analysis Service (Python)**. Service này dùng một mô hình **Cross-Encoder** mạnh mẽ để chấm điểm lại 100 ứng viên này một cách chính xác.
    c. Gateway nhận lại 100 điểm số mới, kết hợp chúng với điểm fusion và điểm chất lượng ban đầu để ra điểm số cuối cùng.
6.  Gateway trả về top 10 kết quả đã được tái xếp hạng cho người dùng.

**Luồng 2: Phân tích (Analyze)**

1.  Người dùng đã có kết quả tìm kiếm, họ click vào một video và muốn "Hỏi đáp về video này".
2.  Giao diện người dùng gửi request đến endpoint mới: `POST /analyze`.
    Body: `{ "task": "qa", "media_id": "video_123", "question": "Người mặc áo đỏ đang nói gì?" }`
3.  **API Gateway** nhận request và thấy `task` là `qa`.
4.  Gateway gọi trực tiếp đến `AnalysisService`, hàm `AnswerQuestionOnMedia`.
5.  **Analysis Service** lấy frame của video, đưa frame và câu hỏi vào mô hình LVLM (LLaVA/GPT-4V).
6.  LVLM trả về câu trả lời. Service gửi câu trả lời này ngược lại cho Gateway.
7.  Gateway trả câu trả lời về cho giao diện người dùng hiển thị.

---

### **Kết Luận Toàn Bộ Kiến Trúc**

Bằng cách chia thành 4 phần, chúng ta đã xây dựng một hệ thống hoàn chỉnh từ A-Z:

- **Phần 1** đảm bảo dữ liệu đầu vào luôn được xử lý một cách nhất quán, tự động và hiệu quả.
- **Phần 2** tạo ra các "chuyên gia" tìm kiếm tốc độ cao, mỗi người giỏi một lĩnh vực.
- **Phần 3** giới thiệu "nhạc trưởng" LLM để chỉ huy các chuyên gia, và một "bộ hòa âm" (Fusion) để kết hợp các màn trình diễn riêng lẻ.
- **Phần 4** bổ sung một "hội đồng thẩm định" (Re-ranking) để chọn ra những màn trình diễn xuất sắc nhất và cung cấp khả năng "phỏng vấn chuyên sâu" (QA) với từng nghệ sĩ.

Đây là một kiến trúc hiện đại, có khả năng mở rộng, và đáp ứng đầy đủ các yêu cầu khắt khe nhất của cuộc thi, từ xử lý dữ liệu lớn, tốc độ truy vấn, độ chính xác cao, đến việc tích hợp AI tạo sinh và tương tác thông minh.
