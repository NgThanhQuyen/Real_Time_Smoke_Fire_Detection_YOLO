# Khảo sát model hiện tại trước khi train lại

Checkpoint `C:/Users/User/Downloads/dfire_clean_v2_yolo11n_best.pt` được chạy bằng Ultralytics 8.4.118 trên CPU, `imgsz=640`, confidence 0,25, IoU NMS 0,7. Checkpoint này được train trên D-Fire sạch v2, chưa train trên các ảnh Catargiu mới. Số liệu dưới đây phục vụ phân tích lỗi, **không phải kết quả test độc lập**: 104 ảnh Catargiu đã được chọn để đưa vào train và video test đã được xem để định hướng dữ liệu.

## 104 ảnh Catargiu đã tuyển chọn

| Chỉ số | Kết quả |
|---|---:|
| Ảnh có nhãn lửa | 75 |
| Ảnh có dự đoán lửa ở conf≥0,25 | 62/75 |
| Box lửa GT khớp box dự đoán lửa IoU≥0,5 | 59/83 |
| Ảnh lửa có dự đoán khói | 5/75 |
| Ảnh âm tính khó | 29 |
| Ảnh âm tính có dự đoán sai bất kỳ | 4/29 |

[Ảnh lửa không khớp](missed_fire.jpg) cho thấy nhiều ngọn lửa nhỏ, lửa trong bếp/lò và lửa có màu/ánh sáng khác cảnh cháy lớn. [Ảnh âm tính bị báo nhầm](negative_false_alarms.jpg) gồm mây dông, mặt trời, đèn trong tuyết và tán cây sáng màu. Không diễn giải 62/75 hay 59/83 thành recall tổng quát vì mẫu được chọn có chủ đích và một số box lửa nhỏ/khó.

## Video kiểm thử

Lấy mẫu khoảng **5 frame/giây**, không xử lý toàn bộ frame. `test_1`, `test_2`, `test_6` có cháy thật theo ảnh mẫu; `test_3` hoàng hôn, `test_4` và `test_5` mây/sương là âm tính. Không có nhãn box hoặc mốc sự kiện đầy đủ cho video dương tính, nên chỉ báo số frame có dự đoán, không tính precision/recall video.

| Video | Khung mẫu | Có lửa | Có khói | Nhận xét |
|---|---:|---:|---:|---|
| `test_1` | 48 | 46 | 35 | Cháy cửa hàng, dự đoán lửa đa số khung |
| `test_2` | 57 | 53 | 57 | Cháy lớn, dự đoán khói mọi khung mẫu |
| `test_3` | 60 | 0 | **3** | Mây cam lúc hoàng hôn bị báo khói ở 5,8–6,2 s, tối đa 0,36 |
| `test_4` | 65 | 0 | **35** | Mây thấp/sương trên núi bị báo khói nhiều đoạn, tối đa 0,80 |
| `test_5` | 50 | 0 | **6** | Mây/sương thung lũng bị báo khói, tối đa 0,48 |
| `test_6` | 61 | 61 | 61 | Cháy rừng ban đêm, dự đoán cả hai lớp mọi khung mẫu |

[Các khung âm tính bị báo khói](video_false_smoke.jpg) cho thấy model khoanh đúng vùng mây/sương, tức lỗi phân biệt ngữ nghĩa, không chỉ nhiễu một vài pixel. `test_4` có các chuỗi dự đoán liên tiếp dài nhất khoảng 1,8 và 2,4 giây ở tốc độ lấy mẫu 5 fps; temporal smoothing đơn giản có thể vẫn phát cảnh báo. Các số liệu này không phải tỉ lệ sự kiện cảnh báo sau pipeline web.

## Quyết định dữ liệu trước run tiếp theo

Chưa đáng train lại toàn bộ chỉ vì 104 ảnh mới. Đợt 29 âm tính có vài mẫu mây/đèn/mặt trời nhưng chưa đủ cảnh **mây thấp, sương bám sườn núi, mây cam trên đường chân trời** từ các nguồn/video độc lập. Đợt 75 ảnh lửa chủ yếu bổ sung ngọn lửa đơn lẻ/được kiểm soát; nhiều ảnh vẫn còn khó đối với model hiện tại, nhưng chưa giải quyết chênh lệch miền khói nhỏ/xa AoF đã biết.

Ưu tiên tiếp theo: tuyển chọn ảnh âm tính độc lập đúng các kiểu mây/sương trên và ảnh dương tính có khói nhỏ/xa cùng cảnh cháy thực tế đa dạng trong nhà/ngoài trời. Kiểm tra toàn ảnh, nhãn box, cùng cảnh, trùng với train/val/test và video kiểm thử. **Không đưa frame của `test_3`–`test_5` vào train.** Sau khi có đợt bổ sung phù hợp mới chạy notebook đối chứng; đánh giá lại trên val/test cố định và video, đồng thời dành holdout mới chưa xem cho kết luận cuối.

Kết quả chi tiết: `summary.json`, `images.json`, `video_frames.json`. Mã chạy: `scripts/catargiu_model_probe.py`.
