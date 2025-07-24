# Hiểu đơn giản như này: API route sẽ luôn gọi một hàm trong service
# tương ứng để xử lý rồi trả về client
# File này là để định nghĩa các dependency cho API,
# ví dụ như các service sẽ được sử dụng trong các route
from core.search_service import SearchService

# Khởi tạo một instance duy nhất của SearchService
# Instance này sẽ được chia sẻ cho tất cả các request
search_service_instance = SearchService()


def get_search_service() -> SearchService:
    """Dependency function to get the shared SearchService instance."""
    return search_service_instance
