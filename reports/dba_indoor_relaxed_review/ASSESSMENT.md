# Sàng lọc DBA-Fire và Indoor theo quy tắc chỉ bổ sung train

Ngày kiểm tra: 11/11/2026. Nguồn D-Fire cố định: `runtime/dfire_original_clean_split_v1.zip`, train 12.699, val 4.244, test 4.244. Không sửa hai ZIP nguồn hoặc D-Fire và chưa gộp ảnh.

## Quy tắc đã áp dụng

- Đọc và giải mã toàn bộ ảnh, kiểm tra CRC ZIP, cặp ảnh/nhãn và cú pháp/hình học YOLO. Sai số tọa độ box dưới 0,001 theo kích thước chuẩn hóa được xem là sai số làm tròn; nhãn nguồn được giữ nguyên.
- Chỉ đối chiếu ứng viên với 8.488 ảnh D-Fire **val/test**, không đối chiếu để loại ảnh trong train hay ảnh khác cùng nguồn. Bỏ ảnh trùng SHA-256 hoặc có SSIM > 0,90 với một ảnh val/test. Ảnh có SSIM đúng 0,90 vẫn giữ.
- Để giảm số cặp cần giải mã, dHash/pHash chọn ứng viên (một trong hai khoảng cách Hamming <= 12), tối đa 30 cặp gần nhất theo tổng khoảng cách được so SSIM ở cỡ xám 128x128. Vì vậy đây là phép sàng lọc gần trùng, không phải bằng chứng tuyệt đối rằng không còn mọi biến thể cắt ghép, xoay hoặc đổi bố cục.
- Nhãn DBA-Fire: 0=lửa, 1=khói. Nhãn Indoor: 0=lửa, 1=khói. Nhãn D-Fire: 0=khói, 1=lửa. Khi gộp sau này phải đảo class ID của cả hai nguồn.

## Kết quả

| Nguồn | Tổng ảnh | Chỉ lửa nguồn | Chỉ khói nguồn | Cả hai nguồn | Loại do trùng >90% val/test | Loại ảnh/nhãn lỗi | Đủ điều kiện kỹ thuật |
|---|---:|---:|---:|---:|---:|---:|---:|
| DBA-Fire | 3.905 | 2.134 | 699 | 1.072 | 41 | 0 | 3.864 |
| Indoor | 5.000 | 2.165 | 2.376 | 459 | 4 | 0 | 4.996 |
| **Tổng** | **8.905** | **4.299** | **3.075** | **1.531** | **45** | **0** | **8.860** |

Sau sàng lọc, DBA-Fire còn **2.093 chỉ lửa, 699 chỉ khói, 1.072 cả hai**. Indoor còn **2.162 chỉ lửa, 2.376 chỉ khói, 458 cả hai**. Tổng cộng **4.255 ảnh chỉ lửa**, **3.075 ảnh chỉ khói**, **1.530 ảnh cả hai** đủ điều kiện kỹ thuật theo quy tắc này.

Indoor có 85 nhãn với box nhô quá mép ảnh rất nhỏ (tối đa 0,000695 chiều ảnh, xấp xỉ dưới một pixel), không có box rộng/cao bằng 0; các ảnh này được giữ theo dung sai làm tròn. Khi xuất bộ train cần kiểm tra trình train có chấp nhận tọa độ sát biên này hay không và chỉ chuẩn hóa box nếu thật sự cần.

Các số trên **không phải kiểm duyệt ngữ nghĩa**: một số ảnh vẫn có lửa thí nghiệm, khung hình liên tiếp, chữ phủ lên ảnh hoặc khói nhìn thấy nhưng có thể thiếu box. Đây là lựa chọn đúng theo yêu cầu chỉ loại ảnh lỗi và ảnh trùng val/test >90%. Trùng giữa DBA-Fire và Indoor cũng chưa được dùng để loại ảnh. Không kết luận 8.860 ảnh đều cải thiện model; cần kiểm tra nhãn và thử nghiệm train khi chủ dự án chọn cách bổ sung.

Chi tiết từng ảnh, cặp val/test giống nhất và lý do loại nằm trong `manifest.csv`; tổng kết máy đọc được nằm trong `summary.json`. Mã tái lập: `scripts/screen_dba_indoor_for_dfire.py` và `notebooks/data_preparation/23_screen_dba_indoor_relaxed_for_dfire.ipynb`.

## Độ phân giải của ảnh chỉ lửa với `imgsz=640`

Thống kê này chỉ tính **4.255 ảnh chỉ lửa đủ điều kiện kỹ thuật** phía trên, dựa trên kích thước ảnh gốc. Nếu cạnh dài từ 640 px trở lên thì bước letterbox về `imgsz=640` không cần phóng đại cả ảnh; cạnh ngắn vẫn có thể nhỏ hơn 640 do tỷ lệ khung hình. Ảnh nhỏ hơn 640 vẫn train được nhưng phải phóng to và không có thêm chi tiết thật.

| Nguồn | Chỉ lửa đủ điều kiện | Cạnh dài ≥640 | Cạnh dài <640 | Cả hai cạnh ≥640 | Đúng 640×640 |
|---|---:|---:|---:|---:|---:|
| DBA-Fire | 2.093 | 786 | 1.307 | 335 | 1 |
| Indoor | 2.162 | 1.805 | 357 | 1.559 | 935 |
| **Tổng** | **4.255** | **2.591** | **1.664** | **1.894** | **936** |

| Cạnh dài ảnh gốc | DBA-Fire | Indoor | Tổng |
|---|---:|---:|---:|
| <320 | 544 | 117 | 661 |
| 320–639 | 763 | 240 | 1.003 |
| 640–959 | 428 | 1.181 | 1.609 |
| 960–1279 | 191 | 7 | 198 |
| ≥1280 | 167 | 617 | 784 |

Độ phân giải phổ biến nhất: Indoor 640×640 (935 ảnh), 1280×720 (416 ảnh), 854×480 (170 ảnh); DBA-Fire 275×183 (103 ảnh), 1920×1080 (81 ảnh), 300×168 (70 ảnh). Như vậy ngưỡng cạnh dài ≥640 giữ 60,9% pool chỉ lửa, tương ứng 37,6% DBA-Fire và 83,5% Indoor. Nếu lọc theo cả hai cạnh ≥640 thì còn 1.894 ảnh (44,5%).

Kích thước box lửa sau khi scale cạnh dài về 640 được tính trong `fire_only_imgsz640.csv`; đây là số hình học từ nhãn nguồn, không chứng minh box đã gắn đủ hoặc ảnh nét. Mã tái lập: `scripts/summarize_fire_only_imgsz640.py` và `notebooks/data_preparation/24_summarize_fire_only_imgsz640.ipynb`.

## Dự phóng D-Fire nếu chỉ thêm ảnh chỉ lửa vào train

Đây là phép cộng từ thống kê D-Fire hiện tại (`reports/dfire_original_split_v1/summary.json`) và danh sách ứng viên, **chưa tạo bộ dữ liệu mới**. Val/test cố định ở 4.244 ảnh mỗi tập. Kịch bản chính dùng 2.591 ảnh chỉ lửa có cạnh dài ≥640 từ hai nguồn (786 DBA-Fire, 1.805 Indoor); kịch bản so sánh dùng cả 4.255 ảnh chỉ lửa đạt điều kiện kỹ thuật bất kể độ phân giải.

| Chỉ số train | D-Fire hiện tại | Thêm 2.591 ảnh ≥640 | Thêm cả 4.255 ảnh |
|---|---:|---:|---:|
| Tổng ảnh train | 12.699 | 15.290 | 16.954 |
| Chỉ lửa | 686 (5,40%) | 3.277 (21,43%) | 4.941 (29,14%) |
| Chỉ khói | 3.464 (27,28%) | 3.464 (22,66%) | 3.464 (20,43%) |
| Cả lửa và khói | 2.664 (20,98%) | 2.664 (17,42%) | 2.664 (15,71%) |
| Nền | 5.885 (46,34%) | 5.885 (38,49%) | 5.885 (34,71%) |
| Ảnh chứa lửa | 3.350 | 5.941 | 7.605 |
| Ảnh chứa khói | 6.128 | 6.128 | 6.128 |
| Tỷ lệ ảnh chứa lửa / ảnh chứa khói | 0,547 | 0,969 | 1,241 |
| Box lửa | 8.246 | 12.069 | 14.699 |
| Box khói | 6.885 | 6.885 | 6.885 |
| Tỷ lệ box lửa / box khói | 1,198 | 1,753 | 2,135 |

| Chỉ số toàn bộ dữ liệu | Hiện tại | Thêm 2.591 ảnh ≥640 | Thêm cả 4.255 ảnh |
|---|---:|---:|---:|
| Tổng ảnh | 21.187 | 23.778 | 25.442 |
| Train | 12.699 (59,94%) | 15.290 (64,30%) | 16.954 (66,64%) |
| Val | 4.244 (20,03%) | 4.244 (17,85%) | 4.244 (16,68%) |
| Test | 4.244 (20,03%) | 4.244 (17,85%) | 4.244 (16,68%) |

Số ảnh chứa lửa/khói có thể chồng lấp vì ảnh `cả hai` được tính vào mỗi phía. Số box bổ sung được cộng trực tiếp từ nhãn của 2.591 ảnh hoặc 4.255 ảnh trong `fire_only_imgsz640.csv`; chưa xác nhận nhãn ngữ nghĩa bằng duyệt từng ảnh. Ngưỡng ≥640 chỉ là lựa chọn kịch bản để thống kê, không phải quyết định lọc hoặc gộp.
