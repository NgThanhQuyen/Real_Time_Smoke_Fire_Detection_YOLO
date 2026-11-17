# Chia validation từ D-Fire sạch

Nguồn: `runtime/dfire_original_clean_v1.zip`. Tập test sạch có 4.244 ảnh được giữ nguyên. Chọn đúng 4.244 ảnh từ train sạch sang validation, còn 12.699 ảnh train. Tỷ lệ hiện tại xấp xỉ 60:20:20. Nhãn gốc vẫn là `0 = smoke`, `1 = fire`.

| Tập | Ảnh | Chỉ lửa | Chỉ khói | Cả hai | Nền | Ảnh chứa lửa | Ảnh chứa khói |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 12.699 | 686 | 3.464 | 2.664 | 5.885 | 3.350 | 6.128 |
| Val | 4.244 | 225 | 1.176 | 895 | 1.948 | 1.120 | 2.071 |
| Test | 4.244 | 208 | 1.174 | 857 | 2.005 | 1.065 | 2.031 |

## Cách chia

Script `scripts/split_dfire_clean_val.py` tính dHash và pHash cho từng ảnh, nối các ảnh có cùng tiền tố nguồn, ID liền kề tối đa 2 và khoảng cách dHash ≤24, pHash ≤20 thành nhóm ứng viên. Một nhóm được gán trọn vào train, val hoặc test. Việc chọn validation dùng seed cố định `20261110`, ưu tiên đủ 4.244 ảnh và phân bố gần tỷ lệ gốc theo bốn trạng thái nhãn và ba tiền tố nguồn. Trong 11.430 nhóm ứng viên không có nhóm nào đi qua val và tập khác. Không sửa hoặc loại thêm ảnh, nhãn.

`split_manifest.csv` ghi tập cũ/mới, nguồn, trạng thái nhãn, SHA-256 và mã nhóm của từng ảnh. `summary.json` ghi số lượng, tiêu chí và giới hạn. ZIP mới `runtime/dfire_original_clean_split_v1.zip` có các thư mục `train`, `val`, `test`, `data.yaml` và ba danh sách tên ảnh. ZIP sạch trước đó được giữ để tái lập.

Đã kiểm tra 21.187 cặp ảnh/nhãn trong ZIP mới khớp byte với ZIP sạch, toàn bộ 4.244 ảnh test không đổi, số ảnh/nhãn khớp và CRC đạt.

Hash cảm nhận và ID liền kề chỉ nhận diện nhóm *ứng viên* cùng cảnh. Chưa kiểm duyệt trực quan toàn bộ các cặp xuyên split, nên không khẳng định đã loại hết rò rỉ cảnh. Nếu sau này thêm ảnh vào train, phải kiểm tra ảnh mới với cả val và test; không đổi hai tập đánh giá này để giữ khả năng so sánh.
