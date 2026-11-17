# Đánh giá lần train thêm 500 ảnh âm tính khó

So sánh hai ZIP người dùng cung cấp:

- Gốc: `F:\6\KLTN\BaoCao\smoke_fire_v2_yolo11n_20261109_091632_results.zip`.
- Thêm 500 ảnh: `F:\6\KLTN\BaoCao\smoke_fire_v2_yolo11n_hardneg500_20261110_040250_results.zip`.

Hai ZIP đều qua CRC và có `best.pt`, metrics, CSV train. `run_summary.json` ghi cùng YOLO11n pretrained, 250 epoch tối đa, patience 20, batch 32, imgsz 640, seed 0, workers 4, Ultralytics 8.4.203 và Tesla T4. Data YAML cùng SHA-256. Run mới ghi 14.269 ảnh train thay cho 13.769, đúng +500 ảnh nền; valid/test cùng 1.684 ảnh và cùng thống kê nhãn. Bộ ZIP đầu vào do dự án tạo giữ valid/test nguyên byte; báo cáo run không tự lưu hash từng ảnh Kaggle Input để xác minh độc lập lần nữa.

## Chỉ số trên test

Giá trị là phần trăm. Δ là điểm phần trăm của run mới trừ run gốc. Precision/Recall của Ultralytics là các giá trị tổng hợp từ đường cong, không nhất thiết ứng với ngưỡng 0,25 của phép đếm ảnh nền.

| Nhóm | Precision gốc → mới | Recall gốc → mới | mAP50 gốc → mới | mAP50–95 gốc → mới |
|---|---:|---:|---:|---:|
| Toàn bộ | 81,82 → 81,05 (−0,77) | 66,17 → 64,09 (−2,08) | 74,39 → 72,75 (−1,64) | 56,05 → 55,01 (−1,04) |
| Fire | 79,51 → 78,46 (−1,05) | 70,41 → 69,21 (−1,20) | 77,74 → 76,35 (−1,39) | 55,48 → 54,60 (−0,88) |
| Smoke | 84,13 → 83,64 (−0,49) | 61,93 → 58,97 (−2,96) | 71,04 → 69,16 (−1,88) | 56,62 → 55,42 (−1,20) |

Validation cũng giảm: toàn bộ mAP50 65,85 → 65,03; mAP50–95 49,15 → 48,76; recall 58,90 → 57,25. Precision tăng rất nhẹ 78,26 → 78,49. Run gốc dừng ở epoch 108, best theo mAP50–95 tại epoch 88; run mới dừng ở epoch 75, best tại epoch 55. Cùng quy tắc dừng sớm, số epoch thực chạy khác nhau.

## Báo nhầm trên 150 ảnh nền test

Từ `background_test_predictions.csv`, ở confidence 0,25:

| Ngưỡng confidence | Gốc | Thêm 500 |
|---:|---:|---:|
| 0,25 | 5/150 | 10/150 |
| 0,30 | 5/150 | 9/150 |
| 0,40 | 5/150 | 8/150 |
| 0,50 | 4/150 | 4/150 |
| 0,60 | 4/150 | 1/150 |
| 0,70 | 2/150 | 0/150 |

Đây là phép tính hậu kiểm từ confidence tối đa mỗi ảnh sau khi chạy dự đoán tại 0,25; chưa đánh giá ảnh nền dưới 0,25. Run gốc có 3 box Fire và 4 box Smoke trên 5 ảnh; run mới có 4 box Fire và 8 box Smoke trên 10 ảnh. Năm ảnh báo nhầm cũ vẫn bị báo nhầm. Năm ảnh mới là `bg_dfire_bg_dfire_01700.jpg`, `bg_dfire_dfire_final_background_0074.jpg`, `..._0155.jpg`, `..._0181.jpg`, `..._0395.jpg`; bốn ảnh báo Smoke và một ảnh báo Fire. Ba ảnh mới thuộc cùng kiểu góc camera đồi/đường, giọt nước hoặc hơi mờ trên kính. Ảnh ghép `background_false_positives_comparison.jpg` cho thấy ảnh gốc và thống kê dự đoán; ảnh ghép không có box vì ZIP kết quả không lưu dự đoán có vẽ box cho cả mười ảnh này.

## Kết luận và giới hạn

Lần train này chưa cho lợi ích trên cùng valid/test: mAP và recall đều thấp hơn, số ảnh nền báo nhầm tại ngưỡng 0,25 tăng gấp đôi. Ở ngưỡng 0,60, mức báo nhầm giảm, nhưng không chọn ngưỡng triển khai từ test đã xem. Hai run có cùng recipe và seed nhưng chỉ có một lần train mỗi bộ; thay đổi số epoch do early stopping và nhiễu huấn luyện vẫn có thể ảnh hưởng. Ảnh Rain/Fog mới chủ yếu là đường phố/xe cộ, trong khi lỗi test tập trung ở camera đồi qua kính có giọt nước; đây là một giả thuyết về khác miền ảnh, chưa được chứng minh nhân quả.

Không thay checkpoint web bằng `best.pt` của run mới chỉ dựa trên báo cáo này. Bước tiếp theo hợp lý là đánh giá cả hai checkpoint trên video/camera âm tính và dương tính độc lập ở cùng ngưỡng, đồng thời tuyển ảnh âm tính gần kiểu camera đồi/giọt nước từ **nguồn khác test** nếu tiếp tục cải thiện dữ liệu. Không đưa ảnh hoặc khung cùng cảnh của test hiện tại vào train.
