# Bộ đầu ra YOLO11n V6

Thư mục này lưu checkpoint suy luận ban đầu cùng metadata của lần train để truy vết nguồn model. Backend đọc `best.pt`, `labelmap.json` và `run_summary.json` khi đăng ký phiên bản đầu tiên.

`data.yaml` chứa đường dẫn dataset trên Kaggle; web local không cần tải dataset đó để suy luận. Không chỉnh sửa hoặc ghi đè `best.pt` khi có model mới. Admin mô hình đưa checkpoint mới vào hệ thống qua giao diện, và checkpoint ứng viên được lưu trong `runtime/models/`.

`best.pt` không được commit lên Git. Khi sao chép dự án sang máy khác, hãy đặt checkpoint vào đúng đường dẫn `models/yolo11n-v6/best.pt`.
