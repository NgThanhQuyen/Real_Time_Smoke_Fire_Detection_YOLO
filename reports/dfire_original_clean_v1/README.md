# D-Fire gốc sau khi làm sạch kỹ thuật

Nguồn chỉ gồm `D-Fire.zip` và các tệp trong `Data splitting/` do chủ dự án cung cấp. Nhãn gốc là `0 = smoke`, `1 = fire`. ZIP nguồn không bị sửa.

## Quy tắc

- Đọc và giải mã từng ảnh JPG bằng Pillow; kiểm tra ZIP bằng CRC.
- Loại cả ảnh và nhãn nếu ảnh không giải mã được hoặc có ít nhất một box YOLO sai lớp, sai định dạng, kích thước không dương hay vượt biên ảnh. Không tự sửa box.
- Tính SHA-256 trên byte ảnh. Nếu cùng ảnh xuất hiện ở train và test, giữ bản test và loại bản train. Không lọc ảnh gần giống hoặc cùng cảnh khác byte.
- Giữ nguyên byte của mọi cặp ảnh và nhãn còn lại. Chỉ lọc tên ảnh bị loại khỏi danh sách train/test và năm fold của tác giả; không chia lại.

## Kết quả

| Tập | Tổng ảnh | Chỉ lửa | Chỉ khói | Cả hai | Nền | Ảnh chứa lửa | Ảnh chứa khói | Box lửa | Box khói |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 16.943 | 911 | 4.640 | 3.559 | 7.833 | 4.470 | 8.199 | 11.090 | 9.230 |
| Test | 4.244 | 208 | 1.174 | 857 | 2.005 | 1.065 | 2.031 | 2.740 | 2.253 |
| **Tổng** | **21.187** | **1.119** | **5.814** | **4.416** | **9.838** | **5.535** | **10.230** | **13.830** | **11.483** |

Đã loại 340 ảnh có nhãn hình học lỗi: 278 train, 62 test. Trong đó có 379 dòng box vượt biên và 18 dòng box có chiều rộng hoặc chiều cao bằng 0; một ảnh có thể có nhiều dòng lỗi. Không có ảnh giải mã lỗi, thiếu cặp nhãn hoặc trùng byte SHA-256 trong 21.527 ảnh gốc.

Validation của năm fold sau khi lọc có lần lượt 3.404, 3.368, 3.389, 3.391 và 3.391 ảnh. Mỗi fold vẫn phân hoạch đúng tập train sạch; test nằm riêng.

`image_manifest.csv` ghi SHA-256 và trạng thái của từng ảnh. `summary.json` chứa các bảng đếm và số ảnh trong từng danh sách. ZIP đầu ra là `runtime/dfire_original_clean_v1.zip` và chứa cặp ảnh/nhãn, danh sách chia tập đã lọc cùng `DATASET_INFO.md`.

Đã kiểm tra CRC của ZIP đầu ra và đối chiếu SHA-256 từng ảnh cùng byte từng nhãn được giữ với ZIP gốc. Chưa duyệt tính đúng đắn ngữ nghĩa của nhãn, ảnh gần trùng hoặc rò rỉ cùng cảnh khác byte.
