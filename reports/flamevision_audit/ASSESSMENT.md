# Đánh giá FlameVision v4 cho đợt bổ sung D-Fire

Ngày 30/10/2026. Nguồn: `F:\6\KLTN\Smoke-Fire-Dataset\FlameVision.zip`, do chủ dự án cung cấp. ZIP dài 3.954.924.416 byte, đọc được toàn bộ thành viên và CRC đạt. Đây là báo cáo khảo sát; **không có ảnh FlameVision nào được nhập train**.

## Cấu trúc và nhãn

- Classification: 8.600 PNG, chỉ có nhãn mức ảnh. Không dùng trực tiếp để train YOLO detection.
- Detection: 4.500 JPG 512×512 và 4.500 XML khớp tên, gồm train 3.150, valid 900, test 450. XML có 13.149 box, **tất cả ghi `fire`**, không có lớp `smoke`. Mỗi ảnh có ít nhất một box lửa.
- 1.327 ảnh có ít nhất một box vượt mép. Trong 1.590 box bị đánh dấu, 1.581 box chỉ vượt **một pixel** do làm tròn tọa độ 513 trên ảnh 512; đây không phải lỗi lớn và có thể cắt nếu sau này dùng. Có 17 box giữ lại dưới 80% diện tích sau cắt, cần loại/sửa riêng. Không dùng con số 1.327 như số ảnh không thể cứu.
- 4.500 ảnh có SHA-256 khác nhau. Không có trùng byte với D-Fire gốc, CQUniversity fire-only hoặc Indoor fire-only. Trùng byte bằng 0 không chứng minh khác cảnh.

## Kiểm duyệt nội dung

Đã mở 180 ảnh lấy mẫu theo tỉ lệ ba split ở `visual_page_01.jpg` đến `visual_page_06.jpg`, vẽ box gốc. Nhiều ảnh là các khung kế tiếp của một số ít đám cháy rừng nhìn từ drone; các góc nhìn, địa hình và vệt lửa lặp nhiều lần. Phần lớn cảnh có **cột khói thật rất rõ** nhưng XML chỉ gắn `fire`. Nếu coi là ảnh chỉ-lửa và để trống nhãn khói, model YOLO hai lớp sẽ học sai tín hiệu khói. Nhiều box lửa rất nhỏ hoặc khoanh vệt mờ/đất cháy khó xác nhận ở 512 px.

Gom ứng viên rất gần bằng điều kiện đồng thời dHash≤3 và pHash≤8 tạo 2.657 cụm từ 4.500 ảnh; 2.306 ảnh nằm trong cụm có ít nhất hai ảnh, cụm lớn nhất 102 ảnh. Có 2.568 cạnh gần trùng đi qua các split **của nguồn FlameVision**. Ngưỡng hash chỉ là tín hiệu duyệt, nhưng các trang ảnh xác nhận sự tập trung vào ít cảnh. Giữ nhiều khung khác nhau trong *train* vẫn được phép theo quyết định của chủ dự án; điều đó không khắc phục nhãn khói thiếu hoặc mất cân đối góc nhìn drone.

## Quyết định

**Không dùng FlameVision để đạt mốc 1.000 ảnh chỉ-lửa.** Có thể nghiên cứu riêng cho bài toán lửa rừng từ drone sau khi gắn nhãn khói và chọn khung theo sự kiện, nhưng đó không phải đợt bổ sung đang cần cho D-Fire. Không xuất bản gộp hoặc đề nghị train từ nguồn này. Danh sách kiểm kê từng ảnh ở `detection_index.csv`, mẫu ảnh ở `visual_sample.csv` và `visual_page_*.jpg`.
