# Kết luận xem 300 ảnh chỉ-lửa FASDD_CV

Ngày xem: 11/11/2026. Đọc `F:\6\KLTN\BaoCao\fasdd_cv_fire_candidate_review.zip` (3,86 MiB, 18 mục, CRC đạt). Hash của ba file COCO trong `summary.json` khớp báo cáo sàng lọc bước một. Notebook chọn 240 ảnh ngẫu nhiên theo split nguồn từ pool 5.898 ảnh chỉ-lửa cạnh dài ≥640 px, cộng 60 ảnh có box lửa nhỏ dưới 16 px sau khi quy cạnh dài về 640. Cả 300 ảnh mẫu đã hiển thị, không lỗi giải mã. Mẫu 60 ảnh nhỏ được chọn có chủ đích, nên không lấy tỷ lệ từ 300 ảnh này để ngoại suy trực tiếp cho toàn pool.

## Quan sát từ 15 bảng ảnh

- Rất nhiều ảnh ngọn lửa tách trên nền đen, thường giống các biến thể/khung gần nhau của ít nguồn; có cả ảnh đồ họa hoặc ngọn lửa được ghép vào bối cảnh. Ví dụ rõ ở các bảng `fire_candidates_007.jpg`, `fire_candidates_011.jpg`, `fire_candidates_012.jpg`.
- Nhiều ảnh nến, bật lửa, diêm, bếp/lò và thử nghiệm đốt có kiểm soát; video dọc có chữ phủ trên ảnh. Những cảnh này không đại diện cho đám cháy cần cảnh báo trong video thực tế. Thấy rõ ở `fire_candidates_004.jpg`, `fire_candidates_008.jpg`, `fire_candidates_013.jpg`.
- Có ảnh cháy xe, nhà, rừng và công trình thật, nhưng chúng nằm xen kẽ trong mẫu và một số chuỗi cảnh lặp xuất hiện nhiều lần, kể cả ở các split nguồn khác nhau: mô hình nhà/xe chữa cháy, nhóm lính cứu hỏa đốt thử trên nền đất, lửa nhỏ trên đường/hành lang, ngọn lửa ở cùng chồng gỗ. Đây là nhận diện bằng mắt trên bảng thu nhỏ, không phải danh sách trùng cảnh đã kiểm đủ toàn bộ 5.898 ảnh.
- Trong 60 ảnh box lửa nhỏ được chọn riêng, nhiều ảnh là bật lửa/diêm/nến hoặc ngọn lửa chỉ vài pixel trong cảnh quá xa/mờ; một số ảnh cháy thật có box nhỏ nhưng nhãn chia thành nhiều mảnh và cần duyệt ở độ phân giải gốc. Phần này chưa cho thấy pool đáng tin để bù lửa nhỏ/xa cho D-Fire.
- Có vài ảnh có thể thiếu box khói hoặc box lửa không ôm đúng vùng; ảnh ghép chưa đủ để kết luận tỷ lệ sai nhãn. Vấn đề chồng lên sai khác COCO↔YOLO và box lỗi đã nêu ở báo cáo bước một.

## Quyết định

**Dừng FASDD_CV làm nguồn bổ sung lửa cho đợt D-Fire hiện tại.** Không tải 12,74 GB về máy, không lọc tiếp hàng nghìn ảnh bằng notebook khác, không gộp/train và không lấy split nguồn làm tập đánh giá. Mẫu ngẫu nhiên 240 ảnh độ phân giải phù hợp không cho thấy mật độ cảnh cháy thật, độc lập và nhãn đủ cao để biện minh công kiểm duyệt đến ngưỡng ≥1.000 ảnh chỉ-lửa sạch đã thống nhất. Đây là đánh giá tính thực dụng của nguồn cho mục tiêu hiện tại, không khẳng định toàn bộ FASDD_CV đều kém hoặc không thể dùng cho nghiên cứu khác.

Tiếp theo có thể đánh giá pool Indoor_Fire_Smoke đã có trên máy, tập trung cảnh cháy trong nhà thật và kiểm tra ảnh gốc/box, nhưng phải giữ nguyên D-Fire val/test và không nhận hàng loạt chỉ theo nhãn nguồn. Giấy phép Kaggle FASDD_CV trong ảnh chụp là `Unknown`; chưa có cơ sở xuất bản bộ gộp từ nguồn này.
