# Thống kê và đề xuất chỉ dùng Fire-Smoke-Detection-Yolov11 + D-Fire

Khảo sát ngày 08/11/2026. Đây là phương án số lượng đã được triển khai; số liệu thành phẩm và kiểm tra nằm tại `reports/final_two_source_dataset/ASSESSMENT.md`.

## Bản hiện tại: đã thêm 2.000 ảnh khói D-Fire

| Tập | Chỉ lửa | Chỉ khói | Cả hai | Nền | Tổng | Ảnh có lửa | Ảnh có khói | Box lửa | Box khói |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5.965 | 3.674 | 1.514 | 1.500 | 12.653 | 7.479 | 5.188 | 11.666 | 6.552 |
| valid | 597 | 127 | 147 | 100 | 971 | 744 | 274 | 1.136 | 400 |
| test | 274 | 79 | 82 | 100 | 535 | 356 | 161 | 544 | 216 |

Tổng 14.159 ảnh; tỷ lệ train/valid/test là 89,4% / 6,9% / 3,8%. Trong 1.500 ảnh nền train có 260 ảnh Fog, 36 ảnh Cloud và 1.204 ảnh D-Fire. Để chỉ còn hai nguồn, cần bỏ 296 ảnh Fog/Cloud và thay bằng D-Fire.

## Phương án đề xuất

| Tập | Thay ảnh nền khác nguồn | Thêm chỉ khói | Thêm nền | Thêm chỉ lửa | Thêm cả hai | Tổng tăng ròng | Tổng sau |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 296 Fog/Cloud bằng 296 D-Fire | 176 | 500 | 0 | 0 | 676 | 13.329 |
| valid | 0 | 276 | 228 | 70 | 121 | 695 | 1.666 |
| test | 0 | 110 | 442 | 179 | 400 | 1.131 | 1.666 |

Tổng 16.661 ảnh, tỷ lệ xấp xỉ 80% / 10% / 10%. Tổng nhập mới từ D-Fire là 2.798 ảnh; bỏ 296 ảnh Fog/Cloud, tăng ròng 2.502 ảnh. Phương án train dùng 176 ảnh chỉ khói và 796 ảnh nền D-Fire trong pool đã rà soát (296 để thay thế, 500 để tăng thêm). Valid/test lấy từ split D-Fire sạch tương ứng, giữ các ảnh hiện tại và không chuyển ảnh giữa ba tập.

Sau phương án, số ảnh **có lửa** ở train/valid/test là 7.479/935/935; số ảnh **có khói** là 5.364/671/671. Hai chỉ số này cũng xấp xỉ 8:1:1. Tỷ lệ của từng nhóm con như ảnh chỉ lửa, ảnh vừa lửa vừa khói hoặc ảnh nền sẽ không đồng thời đúng 8:1:1 vì nguồn val/test D-Fire có số ảnh chỉ lửa hạn chế. Nhóm nền sau phương án là 2.000/328/542.

Pool D-Fire sạch val còn 70 chỉ lửa, 393 chỉ khói, 430 cả hai, 378 nền sau khi trừ ảnh trùng byte với bản hiện tại; pool test còn 215 chỉ lửa, 1.164 chỉ khói, 888 cả hai, 1.882 nền. Các quota đề xuất nằm trong pool theo số lượng, nhưng cần kiểm tra cùng cảnh với ảnh ở split khác và rà nhãn trước khi chọn. Nếu một quota thiếu ảnh sau kiểm duyệt, điều chỉnh quota trong cùng tổng split, không đưa ảnh rò rỉ vào để đủ số.

Lý do chọn 500 ảnh nền train thêm: train đã có 1.500 ảnh nền, còn vấn đề báo nhầm trên ảnh nền D-Fire; thêm ở mức vừa phải để thử tác động. Chỉ còn 176 ảnh D-Fire chỉ khói trong pool train đã rà soát. Ảnh có cả lửa và khói không thu hẹp chênh lệch ảnh có lửa so với ảnh có khói trong train. Vì vậy phương án này không hứa cân bằng tuyệt đối hai lớp.

Khi mở rộng test, kết quả mới không so trực tiếp với các run đã đánh giá trên 535 ảnh test cũ. Giữ danh sách 535 ảnh cũ làm tập đánh giá cố định bổ sung để so sánh lịch sử.
