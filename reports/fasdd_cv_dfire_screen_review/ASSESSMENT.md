# Sàng lọc FASDD_CV cho D-Fire

Ngày xem: 11/11/2026. Nguồn chứng cứ là `F:\6\KLTN\BaoCao\fasdd_cv_dfire_screen.zip` do chủ dự án chạy trên Kaggle với bản `cookiecacheqq/fasdd-cv`. Gói báo cáo 43,11 MiB, 61 mục, CRC đạt. Dataset ảnh gốc 12,74 GB không có trên máy; chưa đối chiếu ảnh FASDD với D-Fire.

## Quy mô và hình học

| Split nguồn | Tổng ảnh | Chỉ lửa | Chỉ khói | Cả hai | Nền | Ảnh có annotation lỗi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 47.660 | 6.276 | 11.681 | 10.022 | 19.600 | 81 |
| val | 31.770 | 4.183 | 7.782 | 6.696 | 13.066 | 43 |
| test | 15.884 | 2.091 | 3.902 | 3.351 | 6.533 | 7 |
| Tổng | 95.314 | 12.550 | 23.365 | 20.069 | 39.199 | 131 |

Các nhóm trên dựa vào COCO sau khi loại ảnh có annotation lỗi. 12.550 là **pool có nhãn chỉ-lửa**, không phải số ảnh thật, độc lập hoặc dùng được cho D-Fire. Trong pool này có 5.898 ảnh cạnh dài ít nhất 640 px (47,0%); 6.652 ảnh nhỏ hơn 640 px. Có 1.838 ảnh đúng 224×224. Có 3.007 ảnh vừa đạt cạnh dài 640 px vừa gần tỷ lệ 16:9. Không ép ảnh bổ sung phải đúng 16:9; đây chỉ là thống kê miền ảnh.

Trên **mọi box lửa hợp lệ**, gồm ảnh chỉ-lửa và ảnh cả hai, có 73.282 box. Khi quy cạnh dài ảnh về 640, trung vị cạnh ngắn box lửa là 43,44 px; 10.662 box (14,5%) nhỏ hơn 16 px, 28.536 (38,9%) nhỏ hơn 32 px. Đây không phải phân bố riêng cho 12.550 ảnh chỉ-lửa. So với D-Fire train đã đo trước đó (trung vị cạnh ngắn box lửa khoảng 26,8 px), nguồn FASDD có xu hướng lửa lớn hơn; cần kiểm tra riêng nhóm lửa nhỏ/xa nếu bổ sung.

## Chất lượng cấu trúc và trùng lặp

- Header đọc được cho toàn bộ 95.314 ảnh; chỉ 300 ảnh được giải mã pixel trong mẫu cố định. Chưa chứng minh mọi ảnh có thể giải mã đầy đủ. Riêng quét header mất khoảng 1.981 giây (33 phút) trên Kaggle.
- 141 annotation COCO lỗi: 138 box vượt ảnh, 3 box có cạnh không dương; ảnh bị ảnh hưởng là 131. Sáu ảnh có kích thước COCO khác header, chủ yếu liên quan EXIF xoay. Có 159 sai khác box ở mức ảnh khi đối chiếu COCO↔YOLO, 3 file YOLO có dòng lỗi cú pháp/hình học và 94 trường hợp tiền tố tên không khớp lớp sau lọc. Các con số này không tự động nói nhãn ngữ nghĩa còn lại là đúng.
- Đã băm SHA-256 toàn bộ 12.550 ảnh chỉ-lửa. Có 5 cặp trùng byte, trong đó một cặp xuyên train/val nguồn. Tên gốc chuẩn hóa đều khác nhau vì tên dạng `fire_CV<id>` được đánh số riêng; vì vậy kết quả 0 nhóm trùng stem **không chứng minh không trùng cảnh**. Ảnh ghép cho thấy ít nhất cảnh xe chữa cháy cháy lặp trong train, val và test dưới các tên khác nhau.
- Chưa kiểm tra trùng byte, tái mã hóa hoặc cùng cảnh giữa FASDD và D-Fire train/val/test. Split nguồn FASDD không được dùng làm val/test D-Fire. FASDD báo thứ tự YOLO `0=fire, 1=smoke` theo nhãn trong gói; D-Fire hiện tại `0=smoke, 1=fire`, nên nếu gộp phải đổi ID.

## Xem ảnh mẫu

Đã xem ba bảng ngẫu nhiên chỉ-lửa, mỗi bảng 24 ảnh, cùng bảng cả hai lớp và nền. Pool chỉ-lửa chứa một số cảnh cháy thật (xe, công trình, rừng), nhưng cũng nhiều nến/bật lửa, lửa nấu/đốt có kiểm soát, ảnh stock có watermark, clip phim/diễn tập, ngọn lửa tách trên nền đen và cùng cảnh ở nhiều split. Bảng cả hai lớp có nhiều cảnh cháy thật hơn, nhưng mục tiêu hiện tại là bổ sung ảnh chỉ-lửa; không coi ảnh có khói là ảnh chỉ-lửa. Mẫu 72 ảnh chưa đủ để ước lượng chính xác tỷ lệ đạt trên toàn pool.

Ảnh minh chứng: `train_fire_only.png`, `val_fire_only.png`, `test_fire_only.png`, `train_fire_and_smoke.png`, `train_background.png`, `label_issues_001.png` trong thư mục này. Các ảnh ghép là bản thu nhỏ, chưa đủ để duyệt box/khói từng ảnh ở kích thước gốc.

## Kết luận

**Tiếp tục sàng lọc có mục tiêu trên Kaggle; không tải/gộp cả 12,74 GB và không train từ FASDD nguyên bản.** Pool 5.898 ảnh chỉ-lửa có cạnh dài ít nhất 640 px đủ lớn để đáng thử tuyển chọn, nhưng ảnh mẫu có nhiều cảnh ngoài mục tiêu và rò rỉ cùng cảnh xuyên split nguồn. Bước tiếp theo là lấy mẫu phân tầng trong pool ảnh độ phân giải phù hợp, xem rõ ảnh/box, ước lượng tỷ lệ cảnh cháy thật và nhãn khói đầy đủ. Nếu tỷ lệ đạt thấp hoặc cảnh độc lập quá ít thì dừng FASDD và quay sang Indoor; nếu đủ triển vọng mới tuyển từng ảnh, so trùng/cùng cảnh với D-Fire, rồi xét quyền sử dụng. Kaggle hiển thị license `Unknown`, chưa có căn cứ để phân phối bản gộp.

Notebook bước hai `notebooks/data_preparation/21_review_fasdd_cv_fire_candidates_kaggle.ipynb` chỉ đọc COCO và khoảng 300 ảnh mẫu trên Kaggle, xuất `/kaggle/working/fasdd_cv_fire_candidate_review.zip`. Đã kiểm tra cú pháp và chạy thử end-to-end trên bộ COCO thu nhỏ; chưa chạy trên FASDD thật.
