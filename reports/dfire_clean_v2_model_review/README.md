# Kiểm tra run YOLO11n trên D-Fire sạch v2

## Đầu vào và cách kiểm tra

- Checkpoint `dfire_clean_v2_yolo11n_best.pt` do người dùng gửi có SHA-256 `a04fee88e6b99c6dcb1bd67f9f32c9a251bb05a4cb03b21d65e22c87dda58355`, trùng báo cáo Kaggle.
- `args.yaml` xác nhận stage 2: YOLO11n từ stage 1, batch 16, kích thước 640, tối đa 100 epoch, patience 25, AdamW, learning rate đầu 0,0003, mosaic 0,5, `close_mosaic=10`.
- `results.csv` có 80 epoch. Validation mAP50–95 cao nhất 0,38867 ở epoch 55; cuối run còn 0,37738. Train box loss tiếp tục giảm còn val box loss tăng sau khoảng epoch 26, nên việc train dài hơn với cùng dữ liệu/cấu hình chưa có bằng chứng sẽ cải thiện.
- Chạy lại checkpoint trên 1.607 ảnh AoF test bằng Ultralytics 8.4.118, confidence 0,25, IoU NMS 0,7. Kaggle đã dùng Ultralytics 8.4.177; vì khác phiên bản, số liệu cục bộ chỉ dùng để phân tích lỗi, số liệu báo cáo chính là JSON Kaggle.

## Kết quả chính

| Chỉ số | Kết quả |
|---|---:|
| Test gộp mAP50–95 theo JSON Kaggle | 0,3558 |
| WEB test mAP50–95 | 0,4113 |
| AoF test mAP50–95 | 0,2188 |
| Smoke mAP50–95 trên WEB test | 0,5101 |
| Smoke mAP50–95 trên AoF test | 0,0510 |
| Ảnh nền AoF có ít nhất một detection, Kaggle | 280/971 (28,8%) |
| Ảnh nền AoF có ít nhất một detection, chạy lại tại máy | 279/971 (28,7%) |
| Ảnh nền AoF có smoke detection, chạy lại tại máy | 242/971 (24,9%) |

Trong 565 box smoke của AoF test, diện tích box trung vị là **0,649% diện tích ảnh**; 42,3% box có diện tích dưới 0,5% ảnh. Trong train WEB, box smoke trung vị là **27,12% diện tích ảnh**; chỉ 2,6% dưới 0,5%. Đây là khác biệt rất lớn về kích thước mục tiêu, phù hợp với lỗi bỏ sót khói nhỏ/xa quan sát được. Khác biệt này là bằng chứng về chênh lệch miền dữ liệu, chưa chứng minh đó là nguyên nhân duy nhất.

Ở confidence 0,25 và IoU ghép box 0,5, chạy lại cục bộ khớp 10/239 box smoke AoF nhỏ hơn 0,5% ảnh; 46/184 box từ 0,5% đến dưới 2%; 71/142 box từ 2% trở lên. Đây là phép kiểm tra ở một ngưỡng cụ thể, không thay cho AP tính trên nhiều ngưỡng confidence.

Ảnh ghép `aof_background_false_smoke.jpg` cho thấy một số phát hiện nhầm trên **sương mù/mây thấp và vệt tối gần ống kính**; `aof_smoke_missed.jpg` cho thấy box smoke thật **rất nhỏ ở đường chân trời hoặc mờ trong cảnh đêm**. Các ảnh cùng camera có tương quan thời gian, do đó 280 ảnh nền báo nhầm không tương đương 280 sự kiện độc lập. Một số khung có thể mơ hồ về nhãn; không sửa nhãn test chỉ vì model dự đoán khác.

## Hướng xử lý tiếp theo

1. Không tăng lên 200 epoch với cùng bộ data chỉ vì test AoF thấp. Run đã dừng sớm sau khi val không cải thiện; mosaic còn hoạt động vì run dừng trước epoch 90.
2. Tuyển chọn một đợt nhỏ **ảnh camera độc lập** có khói nhỏ/xa, khói mờ ban đêm, đồng thời ảnh âm tính khó gồm sương, mây thấp, hơi nước, vệt nước/bẩn trên ống kính, phản chiếu và đèn sáng. Mọi ảnh mới phải kiểm tra nguồn, box, trùng và rò rỉ với cả ba tập D-Fire. Giữ test đã khóa; không đưa khung từ AoF test vào train.
3. Duyệt ảnh ghép cùng video ngữ cảnh để phân biệt sương và khói thật trong các trường hợp mơ hồ. Đánh giá trên video âm tính cố định `test_3`, `test_4`, `test_5` khi người dùng cung cấp; báo tỉ lệ sự kiện cảnh báo sai, không chỉ tỉ lệ frame.
4. Chỉ sau khi có dữ liệu mới và validation tốt hơn mới so recipe tiếp theo. Chọn checkpoint, ngưỡng và temporal smoothing bằng validation; giữ nguyên D-Fire test để so sánh và dành một holdout độc lập mới cho kết luận cuối.

Đã xem lỗi trên AoF test để hiểu giới hạn của baseline. Vì vậy, nếu dùng chính các lỗi này để thiết kế dữ liệu mới, điểm D-Fire test của các phiên bản sau chỉ còn là **benchmark so sánh đã được nhìn trước**, không phải ước lượng cuối hoàn toàn độc lập. Cần một bộ camera/video mới, giữ kín đến sau khi chốt mô hình, để đánh giá cuối không bị ảnh hưởng bởi quá trình cải tiến.

## File kiểm chứng

- `training_curve.png`: đường mAP và box loss theo epoch.
- `aof_background_false_smoke.jpg`: 12 ảnh nền AoF có điểm smoke cao, chọn phân tán theo pHash để xem đa dạng lỗi; box đỏ là dự đoán smoke, box tím là dự đoán fire.
- `aof_smoke_missed.jpg`: 12 ảnh AoF có smoke GT nhưng không khớp detection smoke ở IoU 0,5; box xanh là GT smoke, vàng là GT fire, đỏ/tím là dự đoán.
- `predictions.jsonl`: dự đoán và GT của toàn bộ 1.607 ảnh AoF test để truy vết; `summary.json` tổng hợp số lượng.
