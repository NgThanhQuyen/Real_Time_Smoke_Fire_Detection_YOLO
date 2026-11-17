# Bổ sung 500 ảnh âm tính khó cho data 2

## Đầu ra

- ZIP mới: `runtime/Smoke-Fire-Detection-Dataset-hardneg500.zip`.
- ZIP gốc: `runtime/Smoke-Fire-Detection-Dataset.zip`, giữ nguyên.
- Script tái lập: `scripts/make_hard_negative_review_sheets.py` và `scripts/build_fire_smoke_v2_hardneg500.py`.
- Notebook train riêng: `notebooks/model_training/train_yolo11n_smoke_fire_detection_hardneg500.ipynb`.

## Nguồn và cách chọn

| Nhóm | Nguồn | Ảnh thêm |
|---|---|---:|
| Mưa, nước trên kính, vệt nước | VdiAw `Rain.zip` | 180 |
| Sương, mù, hơi nước, tầm nhìn kém | VdiAw `Fog.zip` | 220 |
| Ánh sáng vàng/cam, hoàng hôn, mây sáng | D-Fire `WEB` train, nhãn rỗng | 100 |
| **Tổng** | | **500** |

VdiAw: <https://data.mendeley.com/datasets/766ygrbt8y/1>, giấy phép CC BY-NC 3.0 (chỉ dùng phi thương mại). D-Fire: <https://github.com/gaia-solutions-on-demand/DFireDataset>. Không dùng `Snow.zip` hoặc nguồn Durham không truy cập được.

Ứng viên đã được xem trên các bảng ảnh trong `review/`. Ảnh quá giống nhau, lệch chủ đề hoặc có tuyết rõ được bỏ khỏi danh sách. `review_decisions.csv` ghi ảnh được chọn và ảnh bỏ qua. `additions.csv` ghi nguồn, SHA-256 và đường dẫn đầu ra của từng ảnh thêm. Việc xem bảng ảnh không thay thế duyệt từng ảnh ở độ phân giải gốc; nếu dùng làm bộ cuối cùng hoặc triển khai thực tế, cần duyệt chi tiết thêm để phát hiện khói/lửa nhỏ bị bỏ sót trong ảnh nguồn.

## Kiểm tra sau khi đóng gói

- 500 ảnh được chọn có SHA-256 khác nhau và không trùng byte với toàn bộ 17.137 ảnh gốc.
- Toàn bộ 34.276 thành viên của ZIP gốc có nội dung SHA-256 giống ZIP mới. Không sửa hoặc xóa bất kỳ ảnh, nhãn, `data.yaml` hay tài liệu gốc nào.
- 500 ảnh mới giải mã được; 500 nhãn YOLO mới đều rỗng. ZIP mới qua kiểm tra CRC.
- Train: **14.269 ảnh** (2.000 ảnh nền); valid: **1.684 ảnh** (152 ảnh nền); test: **1.684 ảnh** (150 ảnh nền). Valid/test giữ nguyên từng byte.
- Các ảnh nguồn có thể chứa cảnh gần giống nhau nhưng khác byte; quy tắc chặn trùng ở đây là SHA-256 tuyệt đối. Không khẳng định đã loại toàn bộ ảnh cùng cảnh.

`summary.json` lưu số lượng và trạng thái kiểm tra bằng máy. Chưa huấn luyện trên ZIP mới, vì vậy chưa có kết luận về thay đổi mAP hay báo nhầm.
