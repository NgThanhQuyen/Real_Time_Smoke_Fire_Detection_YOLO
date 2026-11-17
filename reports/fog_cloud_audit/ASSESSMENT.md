# Kiểm duyệt ảnh âm tính Fog và Cloud cho D-Fire v2

Ngày 30/10/2026. **Chưa tạo bộ dữ liệu gộp hoặc khuyến nghị train.** Nhánh ảnh âm tính đã qua kiểm tra ảnh, hash và đối chiếu D-Fire; nhánh lửa vẫn chưa được duyệt đủ để vượt quality gate toàn bộ.

## Nguồn và chất lượng

| Nguồn | Ảnh gốc | Đã duyệt cho train kế tiếp | Nội dung |
|---|---:|---:|---|
| `Fog.zip` | 300 | 260 | Đường giao thông trong sương, mưa, mờ và đèn xe; nhãn XML/TXT trong nguồn là phương tiện, không phải nhãn lửa/khói. Khi dùng làm âm tính cho YOLO hai lớp, nhãn phải rỗng. |
| `AIForMankind_Cloud_processed.zip` | 1.080 | 36 | Sáu camera cố định, mỗi camera 180 khung trong khoảng ba giờ; chọn sáu thời điểm trải đều mỗi camera. Có mây thấp, mây sáng và ánh hoàng hôn. |

Hai ZIP mở được và `testzip()` không thấy CRC lỗi. Đã xem contact sheet của cả 300 ảnh Fog ở `fog_page_01.jpg` đến `fog_page_06.jpg`, 36 khung Cloud đã chọn ở `cloud_36_spaced.jpg`, và 48 ảnh bị model báo nhầm mạnh nhất ở `negative_flag_page_01.jpg` và `negative_flag_page_02.jpg`. Trong những ảnh đã xem, không thấy lửa hoặc khói cháy thực. Nhóm Cloud có cùng cảnh qua các thời điểm khác nhau; đây là các khung khác nhau chỉ được đưa vào **train**, phù hợp điều chỉnh mới của chủ dự án. Cần theo dõi riêng đóng góp của sáu camera, không gọi 36 khung là 36 cảnh độc lập.

Trong 300 ảnh Fog có 17 ảnh trùng byte dư và 266 nhóm sau khi gom các ứng viên cùng khung rất mạnh bằng SHA/dHash/pHash. Chọn một ảnh độ phân giải tốt nhất cho mỗi nhóm, bỏ sáu ảnh ngắn cạnh dưới 400 px không có bản thay thế đủ tốt. Kiểm tra lại danh sách 260 ảnh đã chọn: không còn cặp đạt đồng thời dHash≤3 và pHash≤8. Embedding ResNet18 cho 42 cặp cosine≥0,96; 41 cặp đã nằm trong nhóm hash mạnh, một cặp còn lại là hai cảnh mưa khác nhau theo hash, không tự loại chỉ vì embedding gần.

Đã so hash toàn bộ 1.380 ảnh âm tính nguồn với **5.620 ảnh val/test D-Fire v2 đang khóa**. Có một cặp Fog đạt đồng thời dHash≤6 và pHash≤12 với test, `Fog/haze-007.jpg` và `test/images/WEB10832.jpg`; [ảnh đối chiếu](locked_hash_pair.jpg) cho thấy đó là đường trong sương và đống lửa khác cảnh. Không phát hiện cặp hash khác theo ngưỡng này. Hash và kiểm tra mắt giúp giảm rủi ro, không chứng minh tuyệt đối không có cùng cảnh bị biến đổi quá mạnh.

Checkpoint D-Fire v2 được chạy ở `imgsz=640`, confidence tối thiểu 0,05 để **ưu tiên xem ảnh khó**, không dùng dự đoán thay nhãn thật. Đầu vào NumPy đã được đổi RGB→BGR đúng yêu cầu của Ultralytics; lượt chấm đầu tiên sai màu đã bị ghi đè và không dùng. Trong 300 Fog có 48 ảnh bị báo khói và 38 ảnh bị báo lửa với confidence≥0,25; trong 36 Cloud có bảy ảnh bị báo khói và một ảnh bị báo lửa ở ngưỡng đó. Những ảnh báo nhầm mạnh đã được xem riêng và là sương/mây/đèn; chúng nên được giữ vì đúng mục tiêu giảm báo nhầm.

Đối chiếu SHA-256 với toàn bộ ảnh gốc D-Fire cho kết quả **0 trùng byte**, giữa 296 ảnh được chọn cũng **0 trùng byte**. Cặp hash đáng ngờ duy nhất đã xem ở ảnh gốc và là hai cảnh khác nhau. Quyết định cuối của nhánh âm tính ở `negative_final_manifest.csv`: **296 ảnh đã duyệt để thêm vào train khi nhánh lửa cũng đạt gate; chưa nhập train.** Chỉ mục hash và đối chiếu val/test ở `negative_index.csv`; điểm model ở `negative_model_probe.csv`. Phần lớn Fog là cảnh đường có sương; sáu camera Cloud thêm mây và ánh hoàng hôn nhưng độ đa dạng bối cảnh âm tính vẫn còn giới hạn. Không đưa video `test_3`, `test_4`, `test_5` vào train. Không thay đổi val/test D-Fire.

## Việc còn thiếu để chốt bản gộp

- Chủ dự án đã chép lại `Indoor_Fire_Smoke.zip` nguyên vẹn (195.789.486 byte, 5.000 ảnh, CRC đạt). Nguồn có **2.165 ảnh ghi nhãn chỉ-lửa**, không phải 2.376; 2.376 là chỉ-khói. Đã lập chỉ mục đối chiếu val/test tại `reports/indoor_fire_audit/fire_only_cross_locked.csv`. Kiểm duyệt trực quan cho thấy nhiều camera/cảnh dàn dựng lặp và ảnh tổng hợp; không nhập hàng loạt.
- ZIP CQUniversity `MyData_Fire.zip` nay có tại `F:\6\KLTN\Smoke-Fire-Dataset\`, 11.021 ảnh, CRC đạt. Đã xem nhiều lô ảnh và box gốc. Phần dư sau hàng đợi đầu chứa nhiều diêm/nến, khung lặp và ảnh có khói thiếu nhãn; chưa đủ căn cứ chốt 1.000 ảnh lửa chất lượng.
- Chỉ khi ≥1.000 ảnh chỉ-lửa sạch và ≥250 ảnh âm tính khó sạch qua đối chiếu cuối, mới tạo **một** bản gộp train. Val/test phải giữ byte như D-Fire v2. Khi công bố lại dữ liệu, kiểm tra quyền của từng nguồn; VdiAw/Fog ghi CC BY-NC 3.0, còn ZIP CQUniversity có thông tin giấy phép không nhất quán giữa trang công bố và `data.yaml`.
