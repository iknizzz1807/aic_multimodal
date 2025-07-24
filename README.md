## AIC repo

#### Tạo thư mục data trong root dự án, thư mục này chứa các file đa phương tiện để truy vấn

### Chạy `indexer`

```bash
python build_index.py
```

### Chạy ứng dụng

```bash
python main.py
```

### Phiên bản mới

- Chạy docker của database trước rồi hãy làm gì thì làm

```bash
docker-compose up -d
```

- Chạy index data nếu chưa chạy

```bash
python -m processing.run_batch_processing
```

- Xem UI của Milvus trên cổng 8000
