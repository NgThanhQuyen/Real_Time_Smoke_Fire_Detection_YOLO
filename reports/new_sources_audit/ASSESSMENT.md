# Đánh giá ba ZIP đề xuất cho D-Fire sạch v2

Ngày kiểm tra: 2026-10-30. Đây là kiểm kê và xem mẫu, chưa nhận ảnh vào train. Val/test D-Fire đang cố định.

## Phương pháp và giới hạn

- Đọc trực tiếp ba ZIP; giải mã toàn bộ 8.271 ảnh; kiểm tra ảnh, số box, nhãn tồn tại, SHA-256, dHash và pHash.
- Đối chiếu ứng viên với ảnh D-Fire hiện diện trong bản `dfire_clean_v2_plus_pyro_catargiu_v1` bằng dHash ≤6 và pHash ≤12. Hash chỉ tạo danh sách nghi ngờ, không tự chứng minh trùng cảnh.
- Xem contact sheet ngẫu nhiên có box và contact sheet các cặp gần val/test. Chưa kiểm duyệt nhãn từng ảnh hoặc tất cả cặp gần trùng. Chưa thể ước lượng chính xác số ảnh cuối cùng được chọn.

## Phát hiện

| Nguồn | Quy mô và nội dung | Phát hiện chất lượng | Quyết định sơ bộ |
|---|---|---|---|
| AIForMankind_V2_processed | 2.191 ảnh, chỉ khói, 2.317 box; 640×480 | 25.507 cặp dHash 0–3 trong metadata nguồn, nhiều cảnh camera lặp. Có ít nhất một cảnh nhìn trùng với D-Fire val trên sheet. Không bổ sung lửa. YAML nội bộ trỏ cả train/val/test về cùng thư mục ảnh. | Ưu tiên thấp; chỉ chọn khói nhỏ/xa sau lọc theo camera/cảnh và rà box. Không dùng split YAML. |
| Indoor_Fire_Smoke | 5.000 ảnh; class 0 là lửa, class 1 là khói theo xem mẫu; 3.592/3.343 box | **2.165 ảnh chỉ-lửa, 2.376 chỉ-khói, 459 cả hai** (đính chính: 2.376 không phải số chỉ-lửa); 3.944 stem gốc, 613 stem lặp, 358 stem xuất hiện ở nhiều split nguồn, liên quan 1.025 ảnh. Ảnh không đồng đều; có video frame, ảnh 224×224, watermark và box/định nghĩa cần xem lại. | Hữu ích để chọn lửa trong nhà, nhưng chỉ nhận ảnh thực sự khác nhau, rõ, nhãn đúng sau lọc và kiểm duyệt. Bỏ split nguồn, remap class theo D-Fire. |
| AIForMankind_Cloud_processed | 1.080 ảnh nền mây, không box; 3072×2048 | Chỉ 6 camera ×180 khung, nên số bối cảnh độc lập nhỏ. Chưa thấy cặp hash sát D-Fire đang dùng, nhưng D-Fire có ảnh HPWREN nên phải kiểm tra camera/cảnh. | Chọn rất ít ảnh mây/sương khó theo camera và thời điểm; không đưa cả 1.080 vào train. |

Không phát hiện ảnh trùng SHA-256 khi đối chiếu từng ZIP với các ảnh D-Fire đang dùng, nhưng việc mã hóa/kích thước khác nhau làm SHA không bắt được ảnh cùng cảnh. Các số ứng viên hash theo nguồn: AIForMankind smoke 87 train, 5 val, 2 test; indoor 2 train, 5 val, 3 test; cloud 0. Một số ứng viên hash là cặp khác cảnh, nên cần xem từng cặp trước khi loại.

## Đề xuất bước kế tiếp

1. Xem lại toàn bộ ứng viên gần D-Fire val/test, lập danh sách camera/cảnh cấm và loại cả chuỗi cùng nguồn.
2. Với indoor, lọc box và nội dung bằng mắt, ưu tiên lửa cháy trong nhà, bỏ ảnh mờ/nhỏ, ví dụ mơ hồ, cùng khung tái xuất và ảnh chồng chữ quá lớn. Có thể giữ khung video thực sự khác nhau trong train, nhưng theo dõi số khung mỗi video và loại toàn bộ cảnh liên quan val/test.
3. Với cloud, dùng một số ít ảnh gây model báo nhầm, cân bằng theo sáu camera; xem lại xem có khói thật trong nhãn âm tính không.
4. Chỉ sau khi có manifest ảnh được chọn và kiểm tra rò rỉ lần cuối mới tạo một bản train mới; giữ nguyên val/test và so sánh với model baseline trên video âm tính cố định cùng holdout mới.

Ảnh mẫu: `AIForMankind_V2_processed_sample.jpg`, `Indoor_Fire_Smoke_sample.jpg`, `AIForMankind_Cloud_processed_sample.jpg`. Các cặp cần kiểm tra: `dfire_val_test_hash_candidates.jpg`. Số liệu máy đọc được ở `runtime/new_sources_audit/`.
