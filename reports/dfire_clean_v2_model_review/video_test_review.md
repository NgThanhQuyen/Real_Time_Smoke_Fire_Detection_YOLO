# Kiểm thử checkpoint D-Fire v2 trên sáu video

## Thiết lập

- Nguồn: `F:\6\KLTN\Smoke-Fire-Dataset\Video_test\test_1.mp4` đến `test_6.mp4`.
- Checkpoint: `dfire_clean_v2_yolo11n_best.pt` do chủ dự án cung cấp; class `0 = smoke`, `1 = fire`.
- Ultralytics cục bộ 8.4.118, CPU, `imgsz=640`, `conf=0.25`, `iou=0.7`.
- Ghép detection thành sự kiện bằng `AlertSmoother` hiện có: cửa sổ 1,5 giây, tỉ lệ kích hoạt 0,6, tỉ lệ kết thúc 0,25, thời lượng tối thiểu 0,7 giây, gộp khoảng cách tối đa 1 giây.
- Đây là đánh giá ở một cấu hình đã nêu; ứng dụng web hiện có thể dùng kích thước ảnh suy luận và IoU khác. Các video đã biết là ca kiểm thử hồi quy, không phải holdout độc lập mới.

## Kết quả

| Video | Nội dung theo kiểm tra hình ảnh | Frame có detection | Sự kiện smoke | Sự kiện fire |
|---|---|---:|---:|---:|
| `test_1` | Video dương tính | 273/288 | 2 | 1 |
| `test_2` | Video dương tính | 340/340 | 1 | 1 |
| `test_3` | Hoàng hôn, mặt trời và mây màu cam | 14/296 | 0 | 0 |
| `test_4` | Mây thấp/sương trên núi | 170/325 | 2 | 0 |
| `test_5` | Mây/sương trên núi | 35/250 | 1 | 0 |
| `test_6` | Video dương tính | 303/303 | 1 | 1 |

`test_4` tạo hai cảnh báo smoke giả từ 1,92–5,48 giây và 6,92–12,96 giây; `test_5` tạo một cảnh báo smoke giả từ 4,08–6,16 giây. Trên `test_3`, các detection khói rời rạc chưa đủ để tạo sự kiện. Ảnh minh chứng cho thấy box trên `test_4` và `test_5` phủ mây/sương, không phải khói từ đám cháy. Không thể tính precision/recall thật trên ba video dương tính vì chưa có nhãn box/frame hoặc mốc sự kiện ground truth.

Chi tiết từng sự kiện nằm trong `video_test_baseline.json`; các khung hình đại diện của ba video âm tính nằm trong `video_test_negatives.jpg`.

## Kế hoạch dữ liệu đề xuất, chưa triển khai

1. Giữ nguyên D-Fire sạch v2 và test cố định. Giữ cả sáu video này làm bộ kiểm thử hồi quy. Không trích frame từ `test_3`–`test_5` hoặc AoF test để huấn luyện.
2. Kiểm duyệt **một nguồn mới trước**: [Pyro-SDIS v2.0](https://huggingface.co/datasets/pyronear/pyro-sdis). Bản này có metadata `partner`, `camera`, `date`, box khói và 1.634 ảnh âm tính từ các báo nhầm của hệ thống. Nguồn mô tả đã giới hạn tối đa 10 ảnh mỗi chuỗi khói, nhưng vẫn cần kiểm tra tương quan cảnh. Chỉ tuyển chọn ảnh khói nhỏ/xa, khói mờ và âm tính khó có ích; không lấy toàn bộ nguồn.
3. Mỗi ứng viên phải được kiểm tra toàn khung, nhãn/box, nguồn camera và thời điểm; so ảnh trùng tuyệt đối và gần trùng với cả train/val/test D-Fire. Giới hạn vài frame đại diện mỗi sự kiện/cảnh; ảnh mơ hồ được đưa vào danh sách duyệt, không tự coi là âm tính. Tách các camera/sự kiện của nguồn mới để có validation độc lập theo nguồn, đồng thời giữ validation D-Fire hiện tại để so với baseline.
4. Sau đợt khói/âm tính, đo lại D-Fire validation, test theo nguồn và sáu video. Tiếp đó mới kiểm duyệt **một nguồn lửa riêng** để thu hẹp chênh lệch ảnh chứa lửa (2.928) và khói (4.683) trong train. [FURG Fire Dataset](https://github.com/steffensbola/furg-fire-dataset) có video với box ngọn lửa và nguồn cảnh rõ, nhưng số cảnh hữu hạn nên không dùng nhiều frame liền nhau để lấp đủ chỉ tiêu. [MIVIA LFDN](https://mivia.unisa.it/large-fire-dataset-with-negative-samples-lfdn/) là ứng viên bổ sung lửa và âm tính khó cần kiểm tra khả năng truy cập, nguồn gốc và trùng trước khi dùng.
5. Dành thêm camera/video hoàn toàn mới, chưa dùng để chọn data/model/ngưỡng, làm holdout cuối. D-Fire test và `test_3`–`test_5` nay là các benchmark đã biết lỗi.

Nguồn chính thức D-Fire xác nhận `PublicDataset` đã chứa ảnh từ HPWREN, FESB và FireNet; vì vậy không nhập lại các bản tổng hợp của chúng khi chưa kiểm tra nguồn và trùng ảnh: <https://github.com/gaia-solutions-on-demand/DFireDataset>.
