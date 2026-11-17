# Kiểm tra ba nguồn bổ sung cho D-Fire

Ngày kiểm tra: 30/10/2026. Đây là kết quả khảo sát nguồn, chưa phải danh sách ảnh được duyệt để gộp. Tập validation và test của D-Fire sạch v2 vẫn cố định. Không dùng split do nguồn ngoài cung cấp để đánh giá mô hình.

## Phạm vi kiểm tra

- Đã tải và đọc ảnh thật của Home Fire Dataset v1.0.0, BoWFire và SAINetset v8.0. Không cần người dùng tải thêm ba nguồn này.
- Đã giải mã toàn bộ 6.500 ảnh Home Fire, 226 ảnh BoWFire và 5.596 ảnh SAINetset; kiểm tra toàn bộ nhãn YOLO của Home Fire và SAINetset, cùng kích thước ảnh/mask của BoWFire.
- Đã tính SHA-256 và hash cảm nhận, xem ảnh ghép các cặp nghi trùng, đối chiếu ứng viên với ba split D-Fire hiện dùng, và chạy checkpoint D-Fire sạch v2 trên mẫu ảnh âm tính SAINetset.
- Kiểm tra bằng mắt chỉ áp dụng cho các mẫu và cặp được nêu; chưa kiểm duyệt thủ công từng ảnh trong ba nguồn. Hash là bộ lọc ứng viên, không chứng minh hai cảnh độc lập hay chứng minh không còn rò rỉ.

## Kết quả

| Nguồn | Nội dung thực tế | Vấn đề chất lượng | Quyết định |
| --- | --- | --- | --- |
| BoWFire | 119 ảnh fire, 107 ảnh non-fire, có mask vùng lửa; 226 ảnh/mask đọc được | Mask chỉ gắn lửa, trong nhiều ảnh còn có khói. Một mask có giá trị xám không nhị phân. 240 patch 50×50 riêng không phù hợp để train YOLO ảnh đầy đủ. | Nguồn ứng viên tốt nhất cho đợt chọn nhỏ: ảnh lửa thật và âm tính màu giống lửa. Phải kiểm tra/ghi nhãn khói trên toàn ảnh trước khi nhận. |
| SAINetset v8.0 | 5.596 ảnh, 245 ảnh có box khói, 5.351 ảnh được gắn âm tính; train 4.462/val 1.134 | 41 nhóm ảnh trùng byte đi xuyên split nguồn (86 ảnh); trong đó 16 nhóm có nhãn mâu thuẫn giữa có khói và rỗng. 132 nhóm có cặp dHash+pHash giống hệt đi xuyên split (506 ảnh). Nhiều frame cùng camera; có ảnh tổng hợp. | Chỉ tuyển một số âm tính khó đã xem đủ độ phân giải và xác minh không có khói. Gộp theo camera/cảnh; bỏ nhóm nhãn mâu thuẫn và ảnh tổng hợp. Không nhập nguyên nguồn hoặc tin split/nhãn rỗng mặc định. |
| Home Fire Dataset v1.0.0 | 6.500 ảnh, nhãn fire/smoke, train 3.900/val 1.300/test 1.300; 135 ảnh rỗng | 176 nhóm có cặp dHash+pHash giống hệt đi xuyên split nguồn (736 ảnh). Ít nhất bốn cặp nghi trùng với bộ Indoor_Fire_Smoke người dùng đã cung cấp được xác nhận bằng mắt. Có nhiều cảnh lặp và lửa an toàn như nến, bếp, lò sưởi. | Ưu tiên thấp. Nếu cần lửa trong nhà, chỉ chọn cảnh cháy phù hợp mục tiêu, đối chiếu lại Indoor và D-Fire theo cảnh, rồi sửa/kiểm tra nhãn. Không dùng split nguồn. |

Home Fire dùng `0 = fire, 1 = smoke`; D-Fire dùng `0 = smoke, 1 = fire`. Nếu nhận ảnh Home phải đổi class ID. Toàn bộ nhãn Home Fire và SAINetset đọc được và hợp lệ về cú pháp/toạ độ; điều này không bảo đảm nhãn đúng về nội dung. BoWFire không có box khói để dùng trực tiếp cho detector hai lớp.

Trên 114 ảnh SAINetset được nguồn ghi là âm tính trong mẫu thử, checkpoint D-Fire sạch v2 phát hiện đối tượng ở 32 ảnh tại ngưỡng 0,25. Các ảnh này tập trung vào một số cảnh camera lặp; có cảnh sương/mây và đèn đêm phù hợp lỗi báo nhầm cần xử lý. Vì nguồn có nhãn mâu thuẫn, con số 32 chỉ là hàng đợi kiểm duyệt, không phải 32 false positive độc lập đã được xác nhận.

## Đối chiếu với D-Fire đang dùng

Không phát hiện ảnh trùng byte giữa ba nguồn và manifest D-Fire hiện dùng. Bộ lọc `dHash ≤ 6` và `pHash ≤ 12` tạo 10 cặp Home Fire, một cặp BoWFire và một cặp SAINetset với D-Fire; các cặp đã xem là trùng hash giả, không phải cùng cảnh. Kết quả này không loại trừ trùng cảnh với thay đổi góc nhìn, ánh sáng hoặc crop mạnh. Mọi ảnh được tuyển sau này vẫn phải so với train/val/test D-Fire và các nguồn đã gộp, rồi xem ứng viên bằng mắt.

## Quy trình tuyển chọn đề nghị

1. Chọn một đợt nhỏ BoWFire: ưu tiên đám cháy ngoài trời/công trình/xe và âm tính như mặt trời, hoàng hôn, vật đỏ/vàng, đèn sáng. Loại ảnh diễn tập/cảnh lặp nếu không thêm ngữ cảnh độc lập. Kiểm tra toàn ảnh cho cả lửa lẫn khói; chuyển mask lửa thành box sau khi xem chất lượng mask.
2. Chọn âm tính SAINetset theo cảnh camera độc lập, tập trung sương, mây thấp, đèn đêm và các ảnh khiến model báo nhầm. Xem ảnh gốc và các frame lân cận, sửa nhãn nếu thật sự có khói; không coi nhãn rỗng là ground truth tự động.
3. Chưa lấy Home Fire hàng loạt. Chỉ mở lại khi nhóm lửa trong nhà vẫn thiếu sau hai nguồn trên; loại nến/bếp/lò sưởi nếu chính sách cảnh báo nhắm đám cháy nguy hiểm.
4. Với mọi ảnh dự kiến nhận: kiểm tra nhãn hai lớp trên toàn khung, so trùng tuyệt đối và cùng cảnh với toàn bộ train/val/test cố định cùng các nguồn đã thêm, gom frame theo nguồn/camera, ghi manifest quyết định nhận/loại và lý do. Chỉ thêm ảnh đã duyệt vào train; giữ val/test D-Fire cố định. Chạy huấn luyện đối chứng khi số cảnh độc lập và phân bố lửa/âm tính thực sự đủ để kỳ vọng đo được tác động.

## Bằng chứng xem ảnh

- `bowfire_sample.jpg`: mẫu lửa và âm tính BoWFire.
- `sainet_sample.jpg`, `sainet_false_positive_sample.jpg`, `sainet_exact_label_conflicts.jpg`: mẫu SAINetset, ứng viên báo nhầm và nhãn mâu thuẫn trên ảnh trùng byte.
- `home_fire_sample.jpg`, `home_train_val_same_hash.jpg`, `home_indoor_candidates.jpg`: mẫu Home Fire, cảnh lặp xuyên split và nghi trùng Indoor.
- `dfire_overlap_candidates.jpg`: các cặp hash ứng viên với D-Fire đã kiểm tra bằng mắt.

Chỉ mục và số liệu kiểm tra chi tiết đã chuyển sang `reports/proposed_sources/source_indices/`; mã kiểm tra ở `scripts/proposed_*`. Cache ZIP/Parquet trong `runtime/proposed_sources/` đã được xóa theo yêu cầu dọn dẹp; nếu muốn tuyển thêm từ các nguồn này phải tải lại ảnh gốc.

## Kết quả tuyển chọn sau kiểm duyệt

Đã xem bảng ảnh của toàn bộ 226 ảnh BoWFire, rồi xem ảnh gốc/mask của các ứng viên lửa ở kích thước lớn. Nhiều ảnh fire có khói chưa được gắn nhãn, một cụm lớn là cùng buổi diễn tập, vài mask bỏ sót ngọn lửa. Chỉ nhận 6 ảnh lửa có box dùng được, gồm một cảnh diễn tập trong phòng, và 14 ảnh âm tính có mặt trời/hoàng hôn, đèn, ánh sáng màu cam. Box fire015 và fire037 được mở rộng thủ công sau khi xem ảnh để ôm đủ ngọn lửa. Các ảnh nguồn gọi là `not_fire063`, `not_fire064` và `not_fire094` có khói rõ nên không được coi là âm tính.

Từ SAINetset, chỉ nhận 4 ảnh âm tính đã xem toàn khung ở kích thước gốc: sương mù (`407608`), đèn đêm (`397418`, `305517`) và màn mờ trên núi (`400326`). Mỗi ảnh đại diện một camera/cảnh; nhãn gốc rỗng và không nằm trong nhóm ảnh trùng byte có nhãn mâu thuẫn. Các frame tương tự từ cùng camera không được nhân số lượng.

Vì lớp lửa vẫn thiếu, đã mở lại Home Fire theo đúng điều kiện trong kế hoạch: lấy mẫu cố định 180 ảnh có nhãn chỉ-lửa từ ba split, bỏ trước các ứng viên trùng Indoor/D-Fire đã biết và cặp có hash giống hệt, rồi xem cả sáu bảng ảnh tại `home_fire_pool/`. Mẫu này chủ yếu là nến, bếp, lò sưởi, đốt rác có kiểm soát, ảnh camera của cùng cảnh và ngọn lửa rất nhỏ; một số box có vẻ gắn ánh sáng hoặc vật thể mơ hồ. Không nhận ảnh Home Fire trong đợt này. Đây là kết luận từ mẫu 180 ảnh, không phải khẳng định mọi ảnh còn lại đều kém.

Bộ YOLO khảo sát từng được xuất tại `runtime/dfire_clean_v2_plus_pyro_catargiu_bow_sai_v1/` và ZIP cùng tên, nhưng đã xóa theo yêu cầu dọn dẹp. Train từng có 8.011, val 1.371, test 4.249 ảnh; đợt này thêm 6 ảnh lửa và 18 ảnh âm tính. Quality gate đã kiểm tra cặp ảnh/nhãn, cú pháp và biên box, SHA-256 từng ảnh mới, toàn bộ val/test giữ nguyên byte, và không có cặp hash nghi vấn `dHash ≤ 6` cùng `pHash ≤ 12` giữa ảnh mới với bộ hiện dùng hoặc với nhau. Manifest quyết định nhận đã lưu tại `reports/dataset_history/bow_sai_selected_manifest.csv`; ảnh ghép 24 ảnh nhận ở `bow_sai_selected.jpg`.

ZIP cuối có SHA-256 `bd716fb4d51d6d376deac5ccca03b143304da422b3787319f6b18273ed71593b`. Cấu hình `data.yaml` và `metadata/quality_gate.json` có trong ZIP.

Train hiện có 3.009 ảnh chứa lửa và 4.705 ảnh chứa khói; trong đó chỉ 486 ảnh chỉ-lửa so với 2.182 ảnh chỉ-khói. Đợt 24 ảnh này chưa giải quyết khoảng thiếu cảnh lửa độc lập và quá nhỏ để kỳ vọng một lần train mới cho kết luận rõ. Cần tìm nguồn lửa thật có nhãn cả khói/lửa đầy đủ, ít khung hình lặp, và tiếp tục ưu tiên âm tính camera đúng lỗi video thực tế. Không train lại chỉ vì đã xuất ZIP này.

## Điều chỉnh sau phản hồi của chủ dự án

Bản ZIP trên là hiện vật ghi lại 24 ảnh đã kiểm duyệt, **không phải bản dữ liệu được khuyến nghị train**. Việc xuất các bản kế tiếp mỗi lần chỉ thêm vài ảnh không đạt mục tiêu. Sáu ảnh lửa BoWFire chiếm khoảng 0,075% train; khoảng cách giữa chỉ-khói và chỉ-lửa còn 1.696 ảnh. Con số này là chỉ báo chọn nguồn, không phải hạn ngạch phải lấp bằng frame lặp.

Đã lấy mẫu ngẫu nhiên cố định 100 trong 730 ứng viên lửa Catargiu chưa xem, với bảng ảnh và manifest ở `runtime/catargiu_audit/unreviewed_fire_sample/`. Mẫu có cảnh cháy thật, nhưng nhiều nến, bếp, lò sưởi, lửa trại/đốt có kiểm soát, watermark/phụ đề, box khói thiếu và ít nhất một cặp cùng cảnh dưới tên khác. Đây là đánh giá mẫu thu nhỏ, chưa phải quyết định nhận/loại từng ảnh. Do đó không tự động nhận 730 ứng viên; nếu khai thác tiếp phải kiểm duyệt ảnh đủ độ phân giải và kiểm tra cảnh/nhãn.

Ưu tiên khảo sát tiếp [DFS của nhóm tác giả](https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset): nguồn công bố 9.462 ảnh, nhãn VOC cho fire, smoke và other. Chưa đọc được ảnh gốc trong môi trường này nên chưa kết luận nguồn phù hợp hoặc có bao nhiêu cảnh độc lập. Chỉ sau khi nhận được ảnh gốc mới kiểm trùng với D-Fire train/val/test và các nguồn đã dùng, kiểm nhãn đầy đủ, rồi ước lượng số cảnh lửa và âm tính khó có thể thêm. Không xuất ZIP train mới trước bước đó.

Cập nhật sau khi chủ dự án thử tải: link OneDrive gốc đã bị [báo hết hạn](https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset/issues/8); Baidu không truy cập được. Không yêu cầu chủ dự án tiếp tục tải DFS. [Bản Roboflow bên thứ ba](https://universe.roboflow.com/publicdataset-zxlfc/dfs-fire-0wdaz/dataset/1) chỉ ghi 8.939 ảnh, chưa xác minh là cùng pool đầy đủ với 9.462 ảnh công bố; không dùng mặc định. [MS-FSDB của nhóm tác giả](https://github.com/XiaoyiHan6/MS-FSDB) là ứng viên mới nhờ công bố 2.731 cảnh và âm tính khó, nhưng chưa kiểm tra được file tải, phân bố lửa hoặc trùng D-Fire. Chưa đề nghị tải hay gộp.
