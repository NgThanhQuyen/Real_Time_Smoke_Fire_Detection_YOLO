# Khảo sát nguồn bổ sung lửa và ảnh âm tính khó

Ngày 30/10/2026. Đây là khảo sát nguồn, chưa có ảnh nào được nhập vào D-Fire sạch v2.

## Mục tiêu đã chốt

Tập train D-Fire sạch v2 hiện có 7.810 ảnh: 405 ảnh chỉ-lửa, 2.160 ảnh chỉ-khói, 2.523 ảnh cả hai và 2.722 ảnh nền. Tập val 1.371 và test 4.249 ảnh được khóa. Đợt kế tiếp chỉ tuyển nhiều ảnh **chỉ-lửa** và ảnh **âm tính khó** như mây thấp, sương mù, hoàng hôn; không chủ động bổ sung khói. Điều kiện xuất bản đã được chủ dự án chốt là **ít nhất 1.000 ảnh chỉ-lửa và 250 ảnh âm tính khó đã duyệt**. Các khung liên tiếp trong train được giữ nếu thực sự khác nhau và có nhãn đúng; vẫn bỏ cùng khung tái mã hóa và mọi cảnh rò sang val/test. Theo dõi thêm số cảnh và mức tập trung theo video/camera. Không ép đủ số bằng ảnh lỗi hay khói thiếu nhãn.

## Nguồn lửa đã tải và kiểm tra bước đầu

- Công bố: [CQUniversity, DOI 10.25946/28747046](https://acquire.cqu.edu.au/articles/dataset/Annotated_Fire_-Smoke_Image_Dataset_for_fire_detection_Using_YOLO_/28747046/1), giấy phép ghi CC BY 4.0, 519,75 MB. Tệp đang có tại `F:\6\KLTN\Smoke-Fire-Dataset\MyData_Fire.zip`, SHA-256 `8965cd9ec15ed0c772f3f946bdaf272fd61d9a32e4b4034e611b7c4af1194abf`, CRC ZIP đạt.
- ZIP thực tế chứa 11.021 JPG, ít hơn 6 ảnh so với 11.027 công bố. `data.yaml` ghi `fire=0`, `smoke=1`; nhãn theo split nguồn: train 3.218 chỉ-lửa, 3.358 chỉ-khói, 2.249 cả hai, 1 nền; valid 386/417/284/1; test 395/415/297/0. Tổng cộng **3.999 ảnh được ghi nhãn chỉ-lửa**, nhưng đây chưa phải 3.999 ảnh đủ điều kiện.
- Trong 3.999 ảnh chỉ-lửa, chỉ có 2.411 tên gốc khác nhau theo bộ chuẩn hóa tên hiện tại sau khi bỏ hậu tố Roboflow; 1.588 ảnh là biến thể cùng tên gốc. Con số cũ 2.549 dùng bộ chuẩn hóa tên trước đó, không còn dùng. Có 21 tên gốc khớp tên ảnh D-Fire, bao gồm 9 ảnh từ test D-Fire gốc. So dHash với toàn bộ D-Fire gốc cho 467 ảnh ở ngưỡng ≤6, trong đó 62 ảnh đồng thời có pHash ≤12. Đây là **ứng viên rò rỉ**, cần xem cặp ảnh; dHash một mình không chứng minh cùng cảnh. Không thấy ảnh trùng byte trong phép so ZIP này, điều dễ xảy ra vì Roboflow đã xuất lại JPG.
- [Mẫu 60 ảnh chỉ-lửa](fire_only_sample_60.jpg) cho thấy nhiều ảnh nến/bật lửa/lửa kiểm soát, nhiều khung cùng cảnh, và một số ảnh có cột khói rõ nhưng nhãn chỉ có lửa. Vì vậy không được nhập hàng loạt theo nhãn nguồn. Toàn bộ 11.021 nhãn có hai nhãn rỗng và không có box sai định dạng/toạ độ theo phép kiểm cú pháp; điều này **không** xác nhận nhãn ngữ nghĩa đúng.
- `data.yaml` trong ZIP vẫn ghi `roboflow.license: Private`, khác giấy phép CC BY 4.0 trên trang xuất bản. Cần lưu xuất xứ/giấy phép và kiểm tra quyền trước khi công khai lại ảnh đã gộp trên Kaggle. Kết quả khảo sát và mã xử lý nội bộ vẫn có thể tiếp tục mà không công bố ảnh gộp.

### Rà bổ sung sau khi chủ dự án duyệt ngưỡng tối thiểu

Chủ dự án chỉ duyệt xuất bản gộp khi có **ít nhất 1.000 ảnh chỉ-lửa** và **ít nhất 250 ảnh âm tính khó** đạt chất lượng. Đây là điều kiện chặn, không phải hạn mức buộc phải lấp đầy. Phép lọc cũ từ 3.999 ảnh chỉ-lửa xuống 2.099 ứng viên đã loại toàn bộ ảnh dHash ≤6 với D-Fire và chỉ giữ một biến thể theo tên gốc. **Phép lọc đó quá chặt theo tiêu chí mới, không còn là danh sách chọn cuối.** Embedding ResNet18 gom 2.099 ảnh thành 1.765 cụm ở cosine >0,95 hoặc 1.537 cụm ở >0,93; các cụm chỉ là tín hiệu xem lại. Checkpoint D-Fire v2 đánh dấu 953/2.099 ảnh có box khói ở confidence ≥0,25. Ở ngưỡng sàng bảo thủ `smoke<0,15` và `fire≥0,5`, còn 688 ảnh; dự đoán này không thay thế việc xem nhãn. Kiểm tra ảnh cho thấy nhiều cảnh lốp xe, flare công nghiệp và cháy rừng lặp dưới tên khác, cũng như ảnh khói không được gắn nhãn. Chưa thể kết luận số ảnh được nhận cuối cùng.

Chứng cứ bổ sung nằm trong `fire_only_cross_dfire.csv` và `screening/` (thứ tự ảnh, ảnh ghép, embedding và kết quả model). Cần nguồn lửa thứ hai độc lập; `Indoor_Fire_Smoke.zip` đã được kiểm kê ở `reports/new_sources_audit/` và có **2.165 ảnh ghi nhãn chỉ-lửa** (class 0 là lửa; số 2.376 trước đây là chỉ-khói), nhưng cũng phải kiểm tra lại cảnh, nhãn và trùng D-Fire trước khi nhập.

Lượt rà 30/10/2026 đã lập hàng đợi trực quan 750 ảnh CQUniversity và 750 ảnh Indoor ở `reports/fire_source_overlap/fire_review_queue_1500.csv`, kèm các trang ảnh có box gốc trong `review_pages/`. Đây **không phải 1.500 ảnh đã duyệt**. 492 tên gốc CQUniversity dự phòng sau hàng đợi đầu chứa dày ảnh nến/diêm, khung cùng cảnh và ảnh nguồn chất lượng thấp; xem mẫu ở `reports/fire_source_overlap/cqu_reserve_pages/`. Ảnh bị model nghi thiếu nhãn khói xem ở `cqu_smoke_flag_pages/`; nhiều ảnh thực sự có khói rõ. Indoor có nhiều khung từ cùng camera, lửa dàn dựng và ảnh tổng hợp; không dùng nguyên hàng đợi để lấp gate. Tình trạng này là lý do tiếp tục tìm nguồn ảnh lửa khác có box để kiểm duyệt.

## Nguồn âm tính đang xét

- [SkyFinder trên Zenodo](https://zenodo.org/records/5884485) có ảnh từ 53 camera cố định, nhiều thời tiết và ánh sáng; chỉ chọn rất ít khung cách xa nhau cho mỗi camera, xem toàn ảnh để xác nhận không có lửa/khói. Không lấy nhiều khung một cảnh để làm đầy số lượng.
- [VdiAw, Mendeley Data](https://data.mendeley.com/datasets/766ygrbt8y/3) có riêng `Fog.zip` 29,2 MB, ảnh giao thông trong sương; nguồn ghi CC BY-NC 3.0. ZIP do chủ dự án cung cấp đã đọc được toàn bộ 300 ảnh, chi tiết đối chiếu ở `reports/fog_cloud_audit/`.
- [MS-FSDB](https://github.com/XiaoyiHan6/MS-FSDB) mô tả 2.731 cảnh và âm tính hoàng hôn/mây, nhưng liên kết Google Drive chính thức trả về “tệp không tồn tại”; chưa dùng. DetectiumFire trên Kaggle khoảng 103 GB, giấy phép hiển thị CC BY-NC-ND 4.0 và lớp “non-fire” có thể chứa khói, nên chưa dùng để tạo âm tính cho bộ hai lớp.

## Quy trình đề xuất trước khi xuất dữ liệu

1. Cố định bản D-Fire v2, ghi hash ảnh/nhãn và test; không thay đổi val/test. Đọc toàn bộ ảnh và nhãn nguồn, chuẩn hóa về `0=smoke`, `1=fire` chỉ trong bản chọn.
2. Gom các ảnh nguồn theo tên gốc, hash, nội dung và cảnh/video; bỏ biến thể augmentation thực chất là cùng khung và mọi ảnh/cảnh giống D-Fire val/test. Trong train có thể giữ nhiều khung thực sự khác nhau từ một sự kiện/camera; ghi số khung mỗi nhóm để tránh một chuỗi lấn át dữ liệu.
3. Kiểm duyệt ở kích thước gốc từng ứng viên chỉ-lửa: lửa thật, box đủ, không có khói thật thiếu nhãn; loại ảnh diễn tập/lửa kiểm soát quá nhiều, hình vẽ/đồ họa, box thiếu hoặc cảnh mơ hồ. Xuất manifest `accept/reject/review` và ảnh minh chứng để người dùng kiểm tra.
4. Lấy âm tính từ các nguồn camera/thời tiết độc lập, ưu tiên hình giống các lỗi mây thấp/sương/hoàng hôn đã quan sát nhưng **không lấy frame từ video kiểm thử hiện có**. Xác nhận toàn khung không có lửa hoặc khói; nhãn YOLO rỗng. Theo dõi số camera/cảnh, không chỉ số ảnh.
5. Chạy đối chiếu SHA/dHash/pHash và kiểm tra thủ công các cặp nghi ngờ với D-Fire val/test và giữa các nguồn; kiểm tra nhãn/box, kích thước, ảnh lỗi. Chỉ nhập ảnh đã duyệt vào train; giữ val/test D-Fire đúng byte. Tạo manifest nguồn và quality gate số ảnh đã duyệt, số cảnh, phân bố bối cảnh và khả năng rò rỉ còn chưa loại trừ.
6. Chỉ khi đạt đợt đủ lớn và qua quality gate mới xuất **một** bộ dữ liệu gộp và notebook train tương ứng; so với baseline trên val/test cố định và video âm tính, không dùng test để chọn ngưỡng.

Chủ dự án đã duyệt kế hoạch bổ sung và mốc 1.000/250; chưa xuất notebook/bộ gộp vì nhánh lửa chưa được chốt đủ ảnh sạch.
