## AIC repo

#### Tạo thư mục data trong root dự án, thư mục này chứa các file đa phương tiện để truy vấn

- Chạy docker của database trước rồi hãy làm gì thì làm

```bash
docker-compose up -d
```

- Chạy index data nếu chưa chạy (xoá file log trong `/process` nếu muốn sạch, file log đồng bộ với volumn trong database của docker nên xoá trong đó luôn nếu muốn sạch nhất có thể)

```bash
python process.py
```

- Xem UI của Milvus trên cổng 8000

- Chạy API server:

```bash
python main.py
```
