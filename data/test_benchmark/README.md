# Bộ dữ liệu kiểm định Golden Test (Smoke & Fire Detection)

Thư mục này dùng để chứa bộ dữ liệu kiểm định chuẩn (~3,857 ảnh) phục vụ tính năng **Kiểm định Benchmark** tự động và so sánh các phiên bản mô hình trực tiếp trên giao diện FLARE.

---

## Cấu trúc thư mục yêu cầu

Bạn chỉ cần giải nén hoặc copy tập dữ liệu kiểm thử vào các thư mục tương ứng bên dưới:

```text
data/test_benchmark/
├── images/             <-- Chứa toàn bộ các file ảnh (.jpg, .jpeg, .png)
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── labels/             <-- Chứa toàn bộ các file nhãn YOLO (.txt)
│   ├── image_0001.txt
│   ├── image_0002.txt
│   └── ...
└── dataset.yaml        <-- File cấu hình YOLO (hệ thống tự động đồng bộ)
```

*(Hệ thống cũng tự động nhận diện nếu dữ liệu nằm trong thư mục con `test/images/` và `test/labels/`)*

---

## Quy ước nhãn đối tượng (YOLO format)
Mỗi dòng trong file `.txt`: `<class_id> <x_center> <y_center> <width> <height>` (tọa độ chuẩn hóa từ 0.0 đến 1.0).
- **Lớp 0 (`khoi` hoặc `smoke`):** Khói
- **Lớp 1 (`lua` hoặc `fire`):** Lửa
- **File nhãn rỗng (0 bytes):** Mẫu âm tính không có khói và lửa (background negatives).

