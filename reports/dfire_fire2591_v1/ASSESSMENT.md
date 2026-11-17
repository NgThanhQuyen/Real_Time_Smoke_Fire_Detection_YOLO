# D-Fire train bổ sung 2.591 ảnh chỉ lửa

Ngày tạo: 11/11/2026. Đầu ra: `runtime/dfire_original_clean_split_fire2591_v1.zip`. Nền: `runtime/dfire_original_clean_split_v1.zip`. Hai ZIP nguồn là `DBA-YOLO-Dataset.zip` và `Indoor_Fire_Smoke.zip` tại `F:/6/KLTN/Smoke-Fire-Dataset/`.

Theo quyết định của chủ dự án, chỉ thêm vào train các ảnh chỉ lửa đủ điều kiện kỹ thuật từ báo cáo `reports/dba_indoor_relaxed_review/`, có cạnh dài ảnh gốc ≥640 px. Nguồn DBA-Fire 786 ảnh, Indoor 1.805 ảnh. Không loại ảnh vì giống train hoặc vì lặp trong nguồn. Danh sách val/test cố định, không bổ sung ảnh mới vào hai tập này.

Nhãn hai nguồn có lớp 0=lửa; khi thêm vào D-Fire đã đổi sang lớp 1=lửa, phù hợp `data.yaml` D-Fire (`0=smoke`, `1=fire`). Tám box Indoor nhô mép dưới một pixel do sai số làm tròn được ghim về mép hợp lệ; tám ảnh được giữ. Ảnh mới giữ byte gốc, tên file được thêm tiền tố nguồn và số thứ tự để tránh va chạm. `addition_manifest.csv` ghi cặp nguồn/đầu ra, SHA-256, kích thước, số box và số box chỉnh mép.

| Tập | Trước | Sau |
|---|---:|---:|
| Train | 12.699 | 15.290 |
| Val | 4.244 | 4.244 |
| Test | 4.244 | 4.244 |
| Tổng | 21.187 | 23.778 |

Train sau gộp có 3.277 ảnh chỉ lửa, 3.464 chỉ khói, 2.664 cả hai và 5.885 nền. Tổng box lửa mới là 3.823; tổng box lửa train dự kiến 12.069, box khói train 6.885. Số ảnh train có lửa/khói là 5.941/6.128.

Đã kiểm tra CRC toàn bộ ZIP đầu ra; 16.976 file ảnh/nhãn của val/test khớp SHA-256 với nền. Kiểm tra SHA-256 cả 2.591 ảnh mới khớp nguồn và mọi nhãn mới chỉ có class 1. `data.yaml` giữ nguyên byte; danh sách train có đúng 15.290 tên. ZIP đầu vào và hai ZIP nguồn giữ nguyên. Chưa train và chưa kiểm duyệt ngữ nghĩa toàn bộ ảnh hoặc độ đầy đủ của nhãn khói trong nguồn.

Mã tạo: `scripts/build_dfire_fire2591.py`. Notebook tái lập: `notebooks/data_preparation/26_build_dfire_fire2591.ipynb`. Kết quả kiểm tra máy đọc được: `summary.json`.
