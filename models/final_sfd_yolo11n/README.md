# Kết quả YOLO11n FinalSFD smoke/fire

`best.pt` là checkpoint được chọn bằng validation. `reports/run_summary.json` ghi nguồn dữ liệu, thông số train, metrics valid/test và SHA-256 checkpoint. `reports/label_fixes.csv` ghi nhãn đã sửa cục bộ. `reports/labelmap.json` ghi thứ tự lớp 0=smoke, 1=fire. Các file `evaluation_*` là biểu đồ đánh giá. Ảnh mẫu, ảnh sai, bảng mức ảnh và khảo sát NMS ở `reports/`. Không dùng metrics ảnh làm kết luận độ chính xác cảnh báo video.
