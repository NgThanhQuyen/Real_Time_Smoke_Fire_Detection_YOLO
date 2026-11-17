# Bộ dữ liệu Fire–Smoke final từ hai nguồn

Ngày hoàn thành: 08/11/2026.

## Đầu ra

- ZIP duy nhất trong `runtime/`: `fire_smoke_dfire_final.zip` (1.070.107.058 byte).
- Notebook có mã xử lý, bảng và biểu đồ đã chạy: `notebooks/data_preparation/16_build_final_two_source_dataset.ipynb`.
- Manifest đóng băng từng ảnh và nhãn: `selection_manifest.jsonl`. Manifest cũng nằm trong ZIP dưới `metadata/`.
- Ba hình báo cáo ở `figures/`.

| Tập | Chỉ lửa | Chỉ khói | Cả hai | Nền | Tổng | Ảnh có lửa | Ảnh có khói | Dòng nhãn lửa | Dòng nhãn khói |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5.965 | 3.850 | 1.514 | 2.000 | 13.329 | 7.479 | 5.364 | 11.666 | 6.746 |
| valid | 667 | 403 | 268 | 328 | 1.666 | 935 | 671 | 1.643 | 861 |
| test | 453 | 189 | 482 | 542 | 1.666 | 935 | 671 | 1.993 | 779 |
| **Tổng** | **7.085** | **4.442** | **2.264** | **2.870** | **16.661** | **9.349** | **6.706** | **15.302** | **8.386** |

Tỷ lệ tổng ảnh là 80,001% / 9,999% / 9,999%. Nguồn Fire-Smoke-Detection-Yolov11 đóng góp 10.459 ảnh; D-Fire đóng góp 6.202 ảnh. Không có nguồn Fog, Cloud, Indoor hoặc AIForMankind trong ZIP cuối. Nhãn `0 = Fire`, `1 = Smoke`; nền dùng file nhãn rỗng. Có 411/25/11 dòng polygon ở train/valid/test; các dòng này được giữ theo yêu cầu.

## Kiểm tra

- Mỗi ảnh được giải mã pixel khi ghi ZIP và so SHA-256 với nguồn; không gặp ảnh lỗi trong số đã chọn.
- 16.661 SHA-256 khác nhau trong manifest, nên không có ảnh trùng byte giữa ba tập.
- ZIP đã qua kiểm tra CRC toàn bộ và có đúng 16.661 ảnh cùng 16.661 file nhãn. Notebook còn đối chiếu 120 cặp ảnh/nhãn mẫu với manifest.
- Ảnh khác byte nhưng gần giống hoặc cùng cảnh được giữ theo quy tắc mới của chủ dự án. Kiểm tra SHA không chứng minh độc lập theo cảnh.
- README của hai tác giả nguồn không có trong ZIP final. Thay bằng `DATASET_INFO.md` do dự án viết, ghi lớp và nguồn ảnh mà không nhận quyền sở hữu ảnh gốc.

## Dọn dữ liệu trung gian

Đã xóa các thư mục dữ liệu gộp cũ, D-Fire clean v2 và audit ảnh ở `runtime/`, cùng ba ZIP gộp cũ và CSV điểm ảnh nền. Chỉ còn `fire_smoke_dfire_final.zip` là ZIP dữ liệu trong `runtime/`. Các thư mục vận hành web (`db`, `models`, `uploads`, `evidence`, `exports`, `feedback`, `rendered`, `thumbnails`, `tmp`, `ultralytics`) và file cấu hình vận hành được giữ. Hai ZIP nguồn gốc nằm ngoài repository tại `F:\6\KLTN\Smoke-Fire-Dataset\` được giữ để notebook có thể tái lập bộ final.
