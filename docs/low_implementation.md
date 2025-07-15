### Xây dựng cơ bản

Mục tiêu của chúng ta là xây dựng một hệ thống có 3 khả năng cốt lõi:

1.  **Tìm kiếm hình ảnh** bằng văn bản.
2.  **Tìm kiếm video** bằng nội dung **hình ảnh** (visual) của nó.
3.  **Tìm kiếm video** bằng nội dung **lời nói** (spoken) trong đó.

Sau đó, chúng ta sẽ kết hợp chúng lại.

---

### **Giai Đoạn 0: Chuẩn Bị Nền Tảng (Setup)**

Trước khi viết dòng code đầu tiên, hãy chuẩn bị môi trường và dữ liệu.

1.  **Tạo Môi Trường Lập Trình:**

    - Tạo một thư mục cho dự án, ví dụ: `ai_challenge_2025`.
    - Mở terminal/command prompt trong thư mục đó.
    - Tạo một môi trường ảo Python để quản lý các thư viện: `python -m venv venv`
    - Kích hoạt môi trường ảo:
      - Windows: `venv\Scripts\activate`
      - macOS/Linux: `source venv/bin/activate`

2.  **Cài Đặt Các Thư Viện Cốt Lõi:**

    ```bash
    pip install torch transformers Pillow
    pip install faiss-cpu # Bắt đầu với bản CPU cho đơn giản
    pip install opencv-python
    pip install openai-whisper # Dùng cho nhận dạng giọng nói
    pip install fastapi uvicorn[standard] # Dùng để tạo API
    ```

3.  **Chuẩn Bị Dữ Liệu:**
    - Tạo một thư mục `data` trong dự án.
    - Giả sử bạn có một tập dữ liệu mẫu từ Ban Tổ chức, hãy bỏ tất cả các file ảnh và video vào thư mục `data`.
    - Tạo một thư mục `output` để lưu các file index được tạo ra.

---

### **Giai Đoạn 1: Xây Dựng Module Tìm Kiếm Hình Ảnh (Text-to-Image)**

Đây là module cơ bản và quan trọng nhất. Chúng ta sẽ làm 2 phần: **Offline Indexing** (xử lý dữ liệu trước) và **Online Search** (tạo API để tìm kiếm).

#### **Bước 1.1: Viết Script Xử Lý Dữ Liệu Hình Ảnh (Offline)**

Mục tiêu: Chuyển đổi tất cả hình ảnh thành vector và lưu vào một chỉ mục (index) của FAISS.

Tạo file `indexer_visual.py`:

```python
import os
import json
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- Cấu hình ---
MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_FOLDER = "data/"
OUTPUT_INDEX_FILE = "output/faiss_visual.index"
OUTPUT_MAPPING_FILE = "output/index_to_path.json"

# --- Tải mô hình CLIP ---
print("Đang tải mô hình CLIP...")
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Thu thập danh sách ảnh ---
image_paths = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
all_vectors = []
index_to_path_map = {}

# --- Trích xuất vector từ mỗi ảnh ---
print(f"Bắt đầu trích xuất vector từ {len(image_paths)} ảnh...")
for i, path in enumerate(image_paths):
    try:
        image = Image.open(path)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Chuẩn hóa vector (quan trọng cho Cosine Similarity)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        all_vectors.append(image_features.cpu().numpy())
        index_to_path_map[i] = path
        print(f"Đã xử lý ảnh {i+1}/{len(image_paths)}: {path}")
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {path}: {e}")

# --- Xây dựng và lưu chỉ mục FAISS ---
if all_vectors:
    # Chuyển đổi danh sách các vector thành một mảng NumPy 2D
    vector_matrix = np.vstack(all_vectors)

    # Lấy số chiều của vector
    d = vector_matrix.shape[1]

    # Dùng IndexFlatL2 cho bước đầu - brute-force nhưng chính xác 100%
    # Dùng Inner Product (IP) vì chúng ta đã chuẩn hóa vector (tương đương Cosine Similarity)
    index = faiss.IndexFlatIP(d)

    # Thêm các vector vào index
    index.add(vector_matrix)

    # Lưu index và mapping
    print(f"Đang lưu index vào {OUTPUT_INDEX_FILE}...")
    faiss.write_index(index, OUTPUT_INDEX_FILE)

    print(f"Đang lưu file mapping vào {OUTPUT_MAPPING_FILE}...")
    with open(OUTPUT_MAPPING_FILE, 'w') as f:
        json.dump(index_to_path_map, f)

    print("Hoàn tất!")
else:
    print("Không tìm thấy vector nào để tạo index.")

```

**Cách chạy:** Mở terminal và chạy `python indexer_visual.py`. Script này sẽ tạo ra 2 file trong thư mục `output`.

#### **Bước 1.2: Tạo API để Tìm Kiếm Hình Ảnh (Online)**

Mục tiêu: Tạo một endpoint web để người dùng có thể gửi một câu văn bản và nhận lại danh sách các ảnh phù hợp.

Tạo file `api.py`:

```python
import faiss
import json
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel

# --- Cấu hình ---
MODEL_ID = "openai/clip-vit-base-patch32"
INDEX_FILE = "output/faiss_visual.index"
MAPPING_FILE = "output/index_to_path.json"

# --- Tải các thành phần cần thiết ---
print("Đang tải mô hình, index và mapping...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
index = faiss.read_index(INDEX_FILE)
with open(MAPPING_FILE, 'r') as f:
    index_to_path = json.load(f)

# --- Khởi tạo FastAPI App ---
app = FastAPI()

class SearchQuery(BaseModel):
    text: str
    top_k: int = 5

@app.post("/search_visual")
def search_visual(query: SearchQuery):
    print(f"Nhận được truy vấn hình ảnh: '{query.text}'")

    # Chuyển truy vấn text thành vector
    inputs = processor(text=query.text, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # Chuẩn hóa vector truy vấn
    text_features /= text_features.norm(dim=-1, keepdim=True)
    query_vector = text_features.cpu().numpy()

    # Tìm kiếm trong FAISS
    distances, indices = index.search(query_vector, query.top_k)

    # Lấy ra kết quả
    results = []
    for i in range(query.top_k):
        result_index = indices[0][i]
        result_path = index_to_path.get(str(result_index))
        if result_path:
            results.append({
                "path": result_path,
                "score": float(distances[0][i])
            })

    return {"results": results}

print("API sẵn sàng hoạt động.")
```

**Cách chạy:** Mở terminal và chạy `uvicorn api:app --reload`. Bạn sẽ có một server chạy ở `http://127.0.0.1:8000`.

**Kiểm tra:** Bạn có thể dùng Postman hoặc `curl` để gửi request đến `http://127.0.0.1:8000/search_visual` với body là `{"text": "a dog on the beach"}`.

**Milestone 1:** Bạn đã có một công cụ tìm kiếm ảnh bằng văn bản hoàn chỉnh!

---

### **Giai Đoạn 2: Mở Rộng Ra Video và Âm Thanh**

#### **Bước 2.1: Xử Lý Video (Xem Video như chuỗi ảnh)**

Chúng ta sẽ mở rộng script `indexer_visual.py` để nó có thể "xem" video, trích xuất các khung hình chính và thêm vector của chúng vào cùng một index.

- **Hành động:** Mở `indexer_visual.py`.
- **Chỉnh sửa:**
  1.  Import `cv2`.
  2.  Trong vòng lặp `for`, thay vì chỉ duyệt file ảnh, hãy duyệt cả file video (`.mp4`, `.mov`, ...).
  3.  Nếu là file video, dùng `cv2.VideoCapture` để mở video.
  4.  Lặp qua video, trích xuất 1 khung hình mỗi giây.
  5.  Với mỗi khung hình, thực hiện y hệt như với một ảnh (đưa qua CLIP, lấy vector).
  6.  **Quan trọng:** File mapping bây giờ cần lưu cả `video_id` và `timestamp` của khung hình đó. Ví dụ: `index_to_path_map[i] = {"video_path": path, "timestamp_sec": current_second}`.

#### **Bước 2.2: Xử Lý Lời Nói (Speech-to-Text)**

Mục tiêu: Trích xuất toàn bộ lời nói từ video/audio và lưu lại để tìm kiếm.

Tạo file `indexer_audio.py`:

```python
import os
import json
import whisper

# --- Cấu hình ---
MEDIA_FOLDER = "data/"
OUTPUT_TRANSCRIPT_FOLDER = "output/transcripts/"

# --- Tải mô hình Whisper ---
# 'base' là mô hình nhỏ và nhanh, 'medium' hoặc 'large' chính xác hơn
print("Đang tải mô hình Whisper...")
model = whisper.load_model("base")

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_TRANSCRIPT_FOLDER, exist_ok=True)

# --- Thu thập file media ---
media_files = [os.path.join(MEDIA_FOLDER, f) for f in os.listdir(MEDIA_FOLDER) if f.endswith(('.mp4', '.mp3', '.wav', '.m4a'))]

# --- Chạy ASR cho từng file ---
print(f"Bắt đầu nhận dạng giọng nói cho {len(media_files)} file...")
for path in media_files:
    try:
        print(f"Đang xử lý: {path}")
        result = model.transcribe(path)

        # Lấy tên file gốc làm tên cho file output
        output_filename = os.path.splitext(os.path.basename(path))[0] + ".json"
        output_path = os.path.join(OUTPUT_TRANSCRIPT_FOLDER, output_filename)

        # Lưu kết quả (bao gồm cả text và timestamp)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result['segments'], f, ensure_ascii=False, indent=2)

        print(f"Đã lưu transcript vào: {output_path}")
    except Exception as e:
        print(f"Lỗi khi xử lý file {path}: {e}")

print("Hoàn tất nhận dạng giọng nói!")
```

**Cách chạy:** `python indexer_audio.py`. Thao tác này sẽ tạo ra nhiều file JSON trong `output/transcripts/`.

#### **Bước 2.3: Tạo API Tìm Kiếm Lời Nói**

- **Hành động:** Mở `api.py`.
- **Chỉnh sửa:**
  1.  Thêm một endpoint mới, ví dụ `@app.post("/search_spoken_text")`.
  2.  Trong hàm của endpoint này, hãy viết logic đơn giản:
      - Lấy `query.text`.
      - Duyệt qua tất cả các file JSON trong `output/transcripts/`.
      - Mở từng file, lặp qua các "segment" trong đó.
      - Nếu `query.text` xuất hiện trong `segment['text']` (dùng `in` và `lower()` để không phân biệt hoa thường), thì thêm `(tên_file, segment)` vào danh sách kết quả.
      - Trả về danh sách kết quả.

**Lưu ý:** Cách tìm kiếm văn bản này rất chậm. Đây là bước sẽ được tối ưu bằng các công cụ chuyên dụng như Elasticsearch ở giai đoạn sau.

---

### **Giai Đoạn 3: Kết Hợp Thành Trợ Lý Ảo Thống Nhất**

Mục tiêu: Tạo một endpoint duy nhất có thể nhận một truy vấn và tự động tìm kiếm trên cả hình ảnh và lời nói, sau đó kết hợp kết quả.

- **Hành động:** Mở `api.py`.
- **Chỉnh sửa:**
  1.  Tạo một endpoint mới: `@app.post("/unified_search")`.
  2.  Hàm này sẽ:
      a. Nhận một câu truy vấn từ người dùng.
      b. Gọi hàm `search_visual` và `search_spoken_text` (gọi nội bộ, không cần qua HTTP).
      c. Bạn sẽ có 2 danh sách kết quả: `visual_results` và `text_results`.
      d. **Logic Kết Hợp (Fusion):** Triển khai một thuật toán kết hợp đơn giản. Ví dụ: - Tạo một dictionary để lưu điểm của mỗi file media. - Duyệt qua `visual_results`: với mỗi kết quả, cộng điểm `score` của nó vào file media tương ứng. - Duyệt qua `text_results`: với mỗi kết quả, cộng một số điểm cố định (ví dụ: 1.0) cho file media tương ứng. - Sắp xếp các file media theo tổng điểm từ cao đến thấp.
      e. Trả về danh sách đã được sắp xếp cuối cùng.

---

### **Giai Đoạn 4: Giao Diện Người Dùng Cơ Bản**

Mục tiêu: Tạo một trang web đơn giản để tương tác với API.

Tạo một thư mục `frontend` và trong đó tạo file `index.html`:

```html
<!DOCTYPE html>
<html lang="vi">
  <head>
    <meta charset="UTF-8" />
    <title>Trợ lý ảo Multimedia</title>
    <style>
      /* CSS đơn giản */
      body {
        font-family: sans-serif;
      }
      #search-box {
        width: 500px;
        padding: 10px;
        font-size: 16px;
      }
      #results {
        margin-top: 20px;
      }
      .result-item {
        border: 1px solid #ccc;
        padding: 10px;
        margin-bottom: 10px;
      }
    </style>
  </head>
  <body>
    <h1>Tìm kiếm trong kho dữ liệu Multimedia</h1>
    <input type="text" id="search-box" placeholder="Nhập mô tả tìm kiếm..." />
    <button onclick="search()">Tìm kiếm</button>
    <div id="results"></div>

    <script>
      async function search() {
        const query = document.getElementById("search-box").value;
        const resultsDiv = document.getElementById("results");
        resultsDiv.innerHTML = "Đang tìm kiếm...";

        const response = await fetch("http://127.0.0.1:8000/unified_search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: query, top_k: 10 }),
        });
        const data = await response.json();

        resultsDiv.innerHTML = "";
        if (data.results && data.results.length > 0) {
          data.results.forEach((item) => {
            const div = document.createElement("div");
            div.className = "result-item";
            // Hiển thị đường dẫn file, bạn có thể làm phức tạp hơn để hiện ảnh/video
            div.innerText = `File: ${item.path} - Score: ${item.score.toFixed(
              4
            )}`;
            resultsDiv.appendChild(div);
          });
        } else {
          resultsDiv.innerHTML = "Không tìm thấy kết quả phù hợp.";
        }
      }
    </script>
  </body>
</html>
```

Bây giờ, bạn có thể mở file `index.html` này trên trình duyệt và sử dụng hệ thống của mình!

### **Lộ Trình Tiếp Theo (Sau khi đã có prototype)**

1.  **Tối ưu tìm kiếm văn bản:** Thay thế tìm kiếm file JSON thô bằng **Elasticsearch** hoặc **OpenSearch**.
2.  **Tối ưu tìm kiếm vector:** Thay `IndexFlatIP` của FAISS bằng các index hiệu quả hơn như `IndexIVFFlat` hoặc `IndexHNSW`.
3.  **Cải thiện thuật toán Fusion:** Áp dụng **Reciprocal Rank Fusion (RRF)** để có kết quả xếp hạng tốt hơn.
4.  **Thêm tìm kiếm sự kiện âm thanh:** Dùng các mô hình như PANNs, YAMNet để tạo vector cho âm thanh không phải lời nói (tiếng vỗ tay, tiếng chó sủa...).
5.  **Tích hợp LLM:** Dùng một LLM (như GPT) để phân tích truy vấn của người dùng trước khi gọi các API tìm kiếm, giúp hiểu ý định người dùng tốt hơn.

Lộ trình trên sẽ giúp bạn có một hệ thống hoàn chỉnh, chạy được trong thời gian ngắn nhất. Chúc bạn thành công

### **Sơ Đồ Kiến Trúc Hệ Thống (Mã Giả)**

```pseudocode
// ===================================================================
// GIAI ĐOẠN 0: CẤU HÌNH VÀ CÁC ĐỐI TƯỢNG CHUNG
// ===================================================================

// --- Các hằng số và cấu hình toàn cục ---
CONFIG {
    DATA_SOURCE_PATH: "/ai_challenge_2025/data/",
    OUTPUT_PATH: "/ai_challenge_2025/output/",

    VISUAL_INDEX_PATH: OUTPUT_PATH + "visual_index/",
    AUDIO_TRANSCRIPTS_PATH: OUTPUT_PATH + "audio_transcripts/",

    CLIP_MODEL_ID: "openai/clip-vit-base-patch32",
    WHISPER_MODEL_ID: "base",

    FAISS_INDEX_FILENAME: "visual.index",
    FAISS_MAPPING_FILENAME: "mapping.json",

    // Tham số cho thuật toán Fusion
    RRF_K_CONSTANT: 60
}

// --- Các đối tượng mô hình sẽ được tải một lần và tái sử dụng ---
MODELS {
    CLIP_MODEL: null,
    CLIP_PROCESSOR: null,
    WHISPER_MODEL: null,
    FAISS_VISUAL_INDEX: null,
    FAISS_VISUAL_MAPPING: null
}

// ===================================================================
// GIAI ĐOẠN 1: OFFLINE PROCESSING SCRIPTS
// ===================================================================
// Chạy một lần để chuẩn bị dữ liệu

// --- Module 1.1: Xử lý Dữ liệu Hình ảnh ---
FUNCTION process_visuals():
    // Tải mô hình CLIP nếu chưa có
    MODELS.CLIP_MODEL = LoadModel(CONFIG.CLIP_MODEL_ID)
    MODELS.CLIP_PROCESSOR = LoadProcessor(CONFIG.CLIP_MODEL_ID)

    all_media_files = ListFiles(CONFIG.DATA_SOURCE_PATH)
    all_vectors = NewList()
    mapping_data = NewMap() // Map từ index (0, 1, 2...) sang thông tin media

    current_index = 0
    FOR EACH file IN all_media_files:
        IF IsImageFile(file):
            image = ReadImage(file)
            vector = MODELS.CLIP_MODEL.GetImageVector(image)

            all_vectors.Add(vector)
            mapping_data[current_index] = { "type": "image", "path": file }
            current_index += 1

        ELSE IF IsVideoFile(file):
            video_frames = ExtractKeyframes(file, frames_per_second=1)
            FOR EACH frame IN video_frames:
                vector = MODELS.CLIP_MODEL.GetImageVector(frame.image_data)

                all_vectors.Add(vector)
                mapping_data[current_index] = { "type": "video_frame", "path": file, "timestamp": frame.timestamp }
                current_index += 1

    // Xây dựng và lưu FAISS Index
    faiss_index = CreateFaissIndex(dimension=vector.size, type="IndexFlatIP")
    faiss_index.AddAll(all_vectors)

    SaveFile(faiss_index, CONFIG.VISUAL_INDEX_PATH + CONFIG.FAISS_INDEX_FILENAME)
    SaveJSON(mapping_data, CONFIG.VISUAL_INDEX_PATH + CONFIG.FAISS_MAPPING_FILENAME)

    PRINT "Hoàn tất xử lý dữ liệu hình ảnh."

// --- Module 1.2: Xử lý Dữ liệu Lời nói ---
FUNCTION process_transcripts():
    // Tải mô hình Whisper nếu chưa có
    MODELS.WHISPER_MODEL = LoadModel(CONFIG.WHISPER_MODEL_ID)

    all_media_files = ListFiles(CONFIG.DATA_SOURCE_PATH, filter=["video", "audio"])

    FOR EACH file IN all_media_files:
        transcript_segments = MODELS.WHISPER_MODEL.Transcribe(file)

        output_filename = GetBaseName(file) + ".json"
        SaveJSON(transcript_segments, CONFIG.AUDIO_TRANSCRIPTS_PATH + output_filename)

    PRINT "Hoàn tất nhận dạng giọng nói."


// ===================================================================
// GIAI ĐOẠN 2: ONLINE SERVICES (Các module tìm kiếm chuyên biệt)
// ===================================================================
// Các hàm này sẽ được gọi bởi API chính

// --- Service 2.1: Tìm kiếm Hình ảnh ---
FUNCTION search_visual(text_query, top_k):
    // Tải mô hình và index nếu chưa có
    IF MODELS.CLIP_MODEL IS NULL:
        MODELS.CLIP_MODEL = LoadModel(CONFIG.CLIP_MODEL_ID)
        MODELS.CLIP_PROCESSOR = LoadProcessor(CONFIG.CLIP_MODEL_ID)
        MODELS.FAISS_VISUAL_INDEX = LoadFaissIndex(CONFIG.VISUAL_INDEX_PATH + CONFIG.FAISS_INDEX_FILENAME)
        MODELS.FAISS_VISUAL_MAPPING = LoadJSON(CONFIG.VISUAL_INDEX_PATH + CONFIG.FAISS_MAPPING_FILENAME)

    // Vector hóa truy vấn
    query_vector = MODELS.CLIP_MODEL.GetTextVector(text_query)

    // Tìm kiếm trong FAISS
    distances, indices = MODELS.FAISS_VISUAL_INDEX.Search(query_vector, top_k)

    // Chuyển đổi kết quả
    results = NewList()
    FOR i FROM 0 TO top_k-1:
        result_index = indices[i]
        media_info = MODELS.FAISS_VISUAL_MAPPING[result_index]
        score = distances[i]

        results.Add({ "media_info": media_info, "score": score })

    RETURN results

// --- Service 2.2: Tìm kiếm Lời nói ---
FUNCTION search_spoken_text(text_query, top_k):
    all_transcript_files = ListFiles(CONFIG.AUDIO_TRANSCRIPTS_PATH)
    media_scores = NewMap() // Map từ media_path sang điểm

    FOR EACH transcript_file IN all_transcript_files:
        transcript_data = LoadJSON(transcript_file)

        media_path = GetOriginalMediaPath(transcript_file)

        FOR EACH segment IN transcript_data:
            IF text_query.lower() IN segment.text.lower():
                // Cộng điểm đơn giản cho mỗi lần tìm thấy
                media_scores[media_path] = media_scores.Get(media_path, 0) + 1

    // Sắp xếp và trả về top_k
    sorted_media = SortMapByValue(media_scores, descending=true)

    results = NewList()
    FOR i FROM 0 TO min(top_k, sorted_media.length)-1:
        media_path = sorted_media[i].key
        score = sorted_media[i].value
        results.Add({ "media_info": {"path": media_path}, "score": score })

    RETURN results

// ===================================================================
// GIAI ĐOẠN 3: FUSION & MAIN API ENDPOINT
// ===================================================================

// --- Module 3.1: Logic Kết hợp ---
FUNCTION fuse_results(ranked_lists):
    final_scores = NewMap() // Map từ media_path duy nhất sang điểm RRF tổng

    FOR EACH list IN ranked_lists:
        FOR rank FROM 0 TO list.length-1:
            item = list[rank]
            // Tạo một ID duy nhất cho mỗi media item để tổng hợp điểm
            media_id = item.media_info.path

            rrf_score = 1 / (CONFIG.RRF_K_CONSTANT + (rank + 1))

            final_scores[media_id] = final_scores.Get(media_id, 0) + rrf_score

    // Sắp xếp kết quả cuối cùng
    sorted_final = SortMapByValue(final_scores, descending=true)

    // Chuyển đổi về định dạng chuẩn
    final_results = NewList()
    FOR EACH item in sorted_final:
        final_results.Add({ "path": item.key, "score": item.value })

    RETURN final_results

// --- Module 3.2: API Chính ---
ENDPOINT POST "/search" (request_body):
    // Phân rã request
    user_query = request_body.query
    top_k = request_body.top_k

    // --- Gọi các dịch vụ con song song (Asynchronously) ---
    // Giả sử có một cơ chế để chạy các hàm này cùng lúc
    TASK visual_task = RunInBackground(search_visual, user_query, top_k=50)
    TASK text_task = RunInBackground(search_spoken_text, user_query, top_k=50)

    // Đợi tất cả các task hoàn thành
    visual_results = WaitFor(visual_task)
    text_results = WaitFor(text_task)

    // --- Hợp nhất kết quả ---
    all_ranked_lists = [visual_results, text_results]
    fused_results = fuse_results(all_ranked_lists)

    // --- Trả về top_k kết quả cuối cùng ---
    final_top_k_results = fused_results.Slice(0, top_k)

    RETURN HTTP_RESPONSE(status=200, body=final_top_k_results)

// ===================================================================
// MAIN EXECUTION FLOW
// ===================================================================

FUNCTION main():
    // Bước 1: Chạy các script xử lý offline (chỉ chạy khi cần)
    // process_visuals()
    // process_transcripts()

    // Bước 2: Khởi động API server
    StartAPIServer(endpoint="/search")
    PRINT "Server đang lắng nghe các truy vấn..."
```
