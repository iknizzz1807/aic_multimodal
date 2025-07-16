import os
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

import config
from services.search_service import SearchService
from .dependencies import get_search_service
from .schemas import (
    SearchQuery,
    VisualSearchResponse,
    AudioSearchResponse,
    UnifiedSearchResponse,
)

router = APIRouter()


@router.get("/")
def root(service: SearchService = Depends(get_search_service)):
    """Kiểm tra health và thông tin cơ bản của API."""
    return service.get_server_info()


@router.post("/search_visual", response_model=VisualSearchResponse)
def search_visual(
    query: SearchQuery, service: SearchService = Depends(get_search_service)
):
    """Tìm kiếm hình ảnh/frame video dựa trên mô tả văn bản."""
    try:
        results = service.perform_visual_search(query.text, query.top_k)
        return {"results": results}
    except Exception as e:
        print(f"❌ Error in visual search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search_audio", response_model=AudioSearchResponse)
def search_audio(
    query: SearchQuery, service: SearchService = Depends(get_search_service)
):
    """Tìm kiếm trong transcript."""
    try:
        results = service.perform_audio_search(query.text, query.top_k)
        return {"results": results}
    except Exception as e:
        print(f"❌ Error in audio search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unified_search", response_model=UnifiedSearchResponse)
def unified_search(
    query: SearchQuery, service: SearchService = Depends(get_search_service)
):
    """Thực hiện tìm kiếm hợp nhất trên cả hình ảnh và âm thanh."""
    try:
        results = service.perform_unified_search(query.text, query.top_k)
        return {"results": results}
    except Exception as e:
        print(f"❌ Error in unified search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/media/{media_id}")
async def stream_media(media_id: str, request: Request):
    """
    Stream một file media (ảnh hoặc video) với hỗ trợ HTTP Range Requests.
    Điều này cho phép tua video (seeking/scrubbing) ở frontend.
    """
    file_path = os.path.join(config.DATA_DIR, media_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Media file not found")

    if not any(media_id.lower().endswith(ext) for ext in config.VIDEO_EXTENSIONS):
        return FileResponse(file_path)

    file_size = os.stat(file_path).st_size
    range_header = request.headers.get("range")
    headers = {"Content-Length": str(file_size), "Accept-Ranges": "bytes"}
    start, end = 0, file_size - 1
    status_code = 200

    if range_header:
        try:
            range_value = range_header.strip().replace("bytes=", "")
            parts = range_value.split("-")
            start = int(parts[0])
            if len(parts) > 1 and parts[1]:
                end = int(parts[1])
            if start >= file_size or end >= file_size:
                raise HTTPException(status_code=416, detail="Range Not Satisfiable")

            headers["Content-Length"] = str(end - start + 1)
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            status_code = 206
        except (ValueError, IndexError):
            raise HTTPException(status_code=400, detail="Invalid Range header")

    def iterfile():
        with open(file_path, "rb") as video_file:
            video_file.seek(start)
            remaining_bytes = end - start + 1
            chunk_size = 8192
            while remaining_bytes > 0:
                data = video_file.read(min(chunk_size, remaining_bytes))
                if not data:
                    break
                remaining_bytes -= len(data)
                yield data

    return StreamingResponse(iterfile(), status_code=status_code, headers=headers)
