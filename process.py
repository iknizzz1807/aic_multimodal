import multiprocessing
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path để đảm bảo các import hoạt động chính xác
# dù chạy lệnh từ bất kỳ đâu.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import hàm main từ vị trí thực sự của nó
from processing.run_batch_processing import main

if __name__ == "__main__":
    """
    Đây là điểm vào chính để chạy pipeline xử lý dữ liệu hàng loạt.
    Nó thiết lập phương thức khởi tạo cho multiprocessing và gọi hàm xử lý chính.
    """
    # Đặt start_method là 'spawn' để tương thích tốt hơn với CUDA
    # và tránh các vấn đề tiềm ẩn trên nhiều hệ điều hành.
    # force=True đảm bảo nó được đặt ngay cả khi đã được cấu hình trước đó.
    multiprocessing.set_start_method("spawn", force=True)

    # Gọi hàm xử lý chính
    main()
