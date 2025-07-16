from services.search_service import SearchService

# Khởi tạo một instance duy nhất của SearchService
# Instance này sẽ được chia sẻ cho tất cả các request
search_service_instance = SearchService()


def get_search_service() -> SearchService:
    """Dependency function to get the shared SearchService instance."""
    return search_service_instance
