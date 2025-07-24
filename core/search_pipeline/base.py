from abc import ABC, abstractmethod
from typing import Dict, Any


class PipelineStep(ABC):
    """
    Lớp cơ sở trừu tượng cho một bước trong pipeline tìm kiếm.
    Mỗi bước nhận một 'context' (từ điển), xử lý nó, và trả về context đã được cập nhật.
    """

    @abstractmethod
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi logic của bước pipeline.

        Args:
            context: Một từ điển chứa dữ liệu được truyền qua các bước.

        Returns:
            Từ điển context đã được cập nhật với kết quả của bước này.
        """
        pass
