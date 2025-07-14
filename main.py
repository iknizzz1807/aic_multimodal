import os
import base64
from io import BytesIO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faiss
import json
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

MODEL_ID = "openai/clip-vit-base-patch32"
INDEX_FILE = "output/faiss_visual.index"
MAPPING_FILE = "output/index_to_path.json"

# Load configs
print("Đang tải mô hình, index và mapping...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
index = faiss.read_index(INDEX_FILE)
with open(MAPPING_FILE, "r") as f:
    index_to_path = json.load(f)


app = FastAPI(title="AI Multimodal Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchQuery(BaseModel):
    text: str
    top_k: int = 5


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with Image.open(image_path) as img:
            max_size = (800, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            if img.mode != "RGB":
                img = img.convert("RGB")

            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_data = buffered.getvalue()

            img_base64 = base64.b64encode(img_data).decode("utf-8")
            return f"data:image/jpeg;base64,{img_base64}"

    except Exception as e:
        print(f"Error converting image {image_path}: {e}")
        return None


@app.get("/")
def root():
    return {"message": "AI Multimodal Search API is running!", "status": "ok"}


# {
#   "results": [
#     {
#       "id": "image_001.jpg",
#       "score": 0.892,
#       "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
#     },
#     {
#       "id": "image_042.jpg",
#       "score": 0.756,
#       "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
#     }
#   ]
# }
@app.post("/search_visual")
def search_visual(query: SearchQuery):
    print(f"Nhận được truy vấn hình ảnh: '{query.text}'")

    try:
        # Text query -> vector
        inputs = processor(text=query.text, return_tensors="pt").to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        # Chuẩn hóa vector truy vấn
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vector = text_features.cpu().numpy()

        # Tìm kiếm trong FAISS
        distances, indices = index.search(query_vector, query.top_k)

        # Simplified response
        results = []
        for i in range(query.top_k):
            if i < len(indices[0]):
                result_index = indices[0][i]
                result_path = index_to_path.get(str(result_index))
                if result_path:
                    filename = os.path.basename(result_path)
                    image_base64 = image_to_base64(result_path)

                    result_data = {
                        "id": filename,
                        "score": float(distances[0][i]),
                    }

                    # Only add image if conversion successful
                    if image_base64:
                        result_data["image"] = image_base64

                    results.append(result_data)

        print(f"Trả về {len(results)} kết quả")
        return {"results": results}

    except Exception as e:
        print(f"Lỗi trong search_visual: {e}")
        return {"error": str(e), "results": []}


if __name__ == "__main__":
    import uvicorn

    print("API sẵn sàng hoạt động.")
    print("Server đang chạy tại: http://127.0.0.1:8000")
    print("API docs tại: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
