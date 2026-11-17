# Thử nghiệm data 2: chia theo nguồn và thêm ảnh nền

Nguồn ảnh dương tính: `F:\6\KLTN\Smoke-Fire-Dataset\fire and smoke detection.v2i.yolov11.zip`.
Nguồn ảnh nền: `runtime/fire_smoke_dfire_final.zip`, chỉ lấy các ảnh có nhãn rỗng trong manifest D-Fire của bản đó. ZIP đầu ra hiện được đặt tên `runtime/Smoke-Fire-Detection-Dataset.zip`. Hai ZIP nguồn được giữ nguyên.

## Cách xử lý

- Giải mã toàn bộ 15.337 ảnh nguồn bằng Pillow, kiểm tra nhãn YOLO box 5 trường, tọa độ và SHA-256. Không có ảnh giải mã lỗi hay trùng byte trong nguồn.
- Gom mọi ảnh có cùng `video_N` vào một nhóm. Với ảnh `youtube-*`, bỏ hậu tố Roboflow `.rf.<hash>` rồi gom theo tên gốc. Chia 705 nhóm theo mục tiêu 80/10/10, cân đối nhóm chỉ lửa, chỉ khói, cả hai và hai họ tên ảnh. Mỗi nhóm chỉ thuộc một tập.
- Chọn theo seed `20261109` 1.500 ảnh nền train, 150 valid, 150 test từ ZIP D-Fire final; giữ split của nguồn nền. Kiểm tra nhãn rỗng, giải mã ảnh và loại ứng viên có SHA-256 trùng ảnh đã chọn. Hai ảnh nền có sẵn của data 2 được giữ và chuyển theo nhóm `video_11` sang valid.
- Giữ nguyên từng ảnh và file nhãn dương tính; không sửa box, không tăng cường ảnh, không train model.

## Thống kê đầu ra

| Tập | Tổng ảnh | Chỉ lửa | Chỉ khói | Cả hai | Nền | Ảnh chứa lửa | Ảnh chứa khói | Box lửa | Box khói |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 13.769 | 2.819 | 3.110 | 6.340 | 1.500 | 9.159 | 9.450 | 20.250 | 16.881 |
| Valid | 1.684 | 352 | 389 | 791 | 152 | 1.143 | 1.180 | 2.758 | 2.945 |
| Test | 1.684 | 352 | 389 | 793 | 150 | 1.145 | 1.182 | 2.413 | 2.046 |
| Tổng | 17.137 | 3.523 | 3.888 | 7.924 | 1.802 | 11.447 | 11.812 | 25.421 | 21.872 |

Trước xử lý có 19 mã `video_N` và 278 tên gốc `youtube-*` đi qua nhiều split. Sau xử lý, không có nhóm tên nguồn nào đi qua nhiều split; 4.351 ảnh dương tính đã đổi split. Manifest có 17.137 dòng và 17.137 SHA-256 khác nhau. ZIP đầu ra đã qua CRC toàn bộ; mọi ảnh có file nhãn. Ảnh nền có file nhãn rỗng. Mẫu 60 ảnh nền ở `background_sample.jpg` đã được xem dạng contact sheet, gồm nhiều cảnh trời, đường, núi, mây và hoàng hôn.

## Giới hạn

Phân nhóm theo tên file xử lý dấu hiệu rò rỉ đã phát hiện, nhưng không chứng minh mọi ảnh cùng cảnh đã được nhận ra. Một video có thể xuất hiện dưới tên khác, hoặc hai khung cùng cảnh khác mã video. Ảnh nền D-Fire dùng split của bản final; bản đó chưa được chứng minh tách độc lập hoàn toàn theo cảnh. Chưa kiểm duyệt trực quan toàn bộ 1.800 ảnh nền ở kích thước gốc và chưa rà các cặp gần trùng khác byte trên toàn bộ 17.137 ảnh. Vì vậy đây là bộ **thử nghiệm train**, và chỉ số test cần được xác nhận bằng video/camera độc lập trước khi kết luận hiệu năng triển khai.

Tệp `manifest.csv` ghi đường dẫn gốc, split cũ/mới, nhóm, loại ảnh và SHA-256. `summary.json` ghi thống kê máy đọc được. Notebook `notebooks/data_preparation/17_build_fire_smoke_v2_grouped_bg.ipynb` chứa mã xử lý, bảng, biểu đồ và kết quả kiểm tra ZIP; đặt `RUN_BUILD = True` để tái lập vào một ZIP riêng. Script dòng lệnh tương ứng là `scripts/build_fire_smoke_v2_grouped_bg.py`.

Khi train YOLO11 detection, `data.yaml` của ZIP này dùng `0=Fire`, `1=Smoke`. Checkpoint ban đầu `models/yolo11n-v6/best.pt` của web có thứ tự ngược (`0=khói`, `1=lửa`), nên không dùng checkpoint đó làm điểm khởi đầu trực tiếp với nhãn hiện tại. Dùng checkpoint YOLO11n pretrained tổng quát hoặc xử lý ánh xạ lớp một cách có kiểm chứng.
