# Kiểm duyệt sơ bộ FireAndSmoke của Catargiu

Ngày kiểm tra: 30/10/2026. Đây là báo cáo khảo sát và tuyển chọn nguồn. Bản D-Fire + Pyro gốc không bị thay đổi; bộ mới chỉ bổ sung vào train và giữ val/test D-Fire cố định.

## Nguồn và khả năng đọc

| ZIP | Ảnh | SHA-256 của ZIP |
|---|---:|---|
| `FireSmokeDataset.v2i.yolov8.zip` | 21.118 | `683f29abed307d92b97adf9517b92d61d3f08fe5d776259f643a86e7dd8940b2` |
| `FireSmokeNEWdataset.v1i.yolov8.zip` | 6.176 | `af089a14d6afbda855cada968575080205acead43bf3b41c529c6d3c1c5aaa6d` |

Hai ZIP đọc đầy đủ, `testzip()` không báo lỗi. Toàn bộ 27.294 JPEG giải mã được ở 640×640; mỗi ảnh có file nhãn tương ứng, không có box sai cấu trúc, tọa độ vượt biên hoặc kích thước không dương. `data.yaml` khai báo `0=fire, 1=other, 2=smoke`. Dữ liệu đã được Roboflow **stretch** về 640×640 nên không còn tỉ lệ khung hình gốc. Việc nhãn hợp lệ về cú pháp không xác nhận nhãn đúng về nội dung.

## Phân bố và rủi ro

| Chỉ số | Phần 1 | Phần 2 | Tổng |
|---|---:|---:|---:|
| Ảnh chứa lửa | 14.725 | 2.692 | 17.417 |
| Ảnh chứa khói | 16.038 | 4.418 | 20.456 |
| Ảnh chỉ lửa, không có `other` | 3.274 | 184 | 3.458 |
| Ảnh chỉ `other` | 582 | 1.365 | 1.947 |
| Ảnh không box | 670 | 72 | 742 |
| Nhóm tên cảnh | 7.460 | 268 | 7.728 |
| Nhóm xuất hiện ở nhiều split của nguồn | 882 | 254 | 1.136 |
| Ảnh thuộc những nhóm xuyên split | 14.131 | 6.083 | 20.214 |

Tên video + số frame là tín hiệu nhóm cảnh, không phải danh tính cảnh được chứng minh. Dù vậy, số lượng nhóm xuyên split rất lớn: **không sử dụng train/valid/test do nguồn cấp làm phép đánh giá độc lập**. Có 88 cặp ảnh trùng SHA-256 bên trong hai ZIP, trong đó 64 cặp nằm ở các split khác nhau.

Đối chiếu theo tên ảnh gốc tìm thấy **1.143 ảnh phần 1 có cùng tên với D-Fire gốc**: 633 thuộc train D-Fire sạch, **111 val**, **327 test**, 72 ảnh đã loại ở quy trình D-Fire. SHA-256 của ảnh không trùng vì nguồn này đã đổi kích thước/tái mã hóa. [Ảnh ghép cạnh D-Fire test](dfire_test_near_review.jpg) xác nhận nhiều cặp thực sự là cùng ảnh hoặc cùng cảnh. Trong các ảnh cùng tên, có trường hợp nhãn không nhất quán: 88 ảnh D-Fire thuộc loại có cả lửa và khói nhưng nguồn chỉ còn box khói.

Tìm kiếm dHash ≤6 trên ảnh nguồn so với toàn bộ D-Fire sạch phát hiện 3.354 ảnh có ít nhất một ứng viên gần; khi thêm pHash ≤12 còn 1.388 ảnh. Đây là **ứng viên cần kiểm tra**, không phải số ảnh chắc chắn rò rỉ: hình ít chi tiết và khung hình đã bị stretch có thể gây cả dương tính giả lẫn âm tính giả. Với 73 ảnh Pyro đã nhận vào train, không thấy cặp nào đạt đồng thời dHash ≤6 và pHash ≤12; kết quả này cũng không chứng minh không có ảnh cùng cảnh.

## Kiểm tra bằng mắt

- [Mẫu chỉ lửa](fire_only_review.jpg): có cảnh trong nhà, ngoài trời, ngày, đêm và lửa nhỏ, nhưng nhiều frame video liên tiếp hoặc có watermark/phụ đề. Không nên lấy hàng loạt frame trong cùng một video để tăng số lượng.
- [Mẫu `other`](other_only_review.jpg): có mặt trời, mây, đèn, ánh đỏ và sương phù hợp với lỗi báo nhầm. Tuy nhiên ảnh `frame3562` được gắn **chỉ `other` dù nhìn thấy lửa**, còn `frame3518` cùng cảnh có box lửa. Hai ảnh này có dHash cách 10, pHash cách 20, minh họa việc chỉ dùng hash không bắt được mọi cảnh trùng.
- [Mẫu lửa + khói](fire_smoke_review.jpg): nhiều box nhìn hợp lý, nhưng có cảnh tương tự D-Fire và đối tượng nhỏ/che khuất cần kiểm từng ảnh nếu nhận vào train.
- [Mẫu từ hàng đợi lửa](queue_fire_only_review.jpg) và [hàng đợi âm tính](queue_hard_negative_review.jpg): đây là ví dụ để duyệt, **chưa phải ảnh đã chấp nhận**. Một số âm tính nằm gần phương tiện chữa cháy/ánh đèn mạnh, cần xem toàn khung ở độ phân giải gốc trước khi xác nhận không có lửa hoặc khói.

## Hàng đợi đề xuất kiểm duyệt

`runtime/catargiu_audit/review_queue.csv` chứa **1.071 ứng viên một ảnh cho mỗi nhóm tên cảnh**: 970 ảnh chỉ lửa và 101 ảnh âm tính khó. Trong đó phần 1 đóng góp 932 lửa + 60 âm tính; phần 2 đóng góp 38 lửa + 41 âm tính. Nhóm cảnh đã được bổ sung quy tắc gộp cho tên `fire1-*`, `Img_*` và Pexels video sau khi kiểm tra bằng mắt thấy các chuỗi khung hình bị tách sai. Hàng đợi vẫn chỉ là danh sách ứng viên, không dùng để train trực tiếp.

Hàng đợi bảo thủ loại nhóm có tên gốc D-Fire, tiền tố `WEB`/`AoF`/`PublicDataset`, hash gần D-Fire theo ngưỡng nêu trong mã, hoặc tên `frameNNN` không cho biết video. Với ảnh âm tính, còn loại nhóm mà bất kỳ frame nào có box lửa/khói. Mỗi nhóm lấy một frame đại diện và gộp ảnh trùng SHA. Quy tắc này có thể bỏ sót ảnh tốt, cũng chưa phát hiện hết cùng cảnh đổi góc/đổi thời điểm; **không dùng hàng đợi để train trực tiếp**.

## Tuyển chọn đợt 1

Từ hàng đợi, xem bản thu nhỏ của 240 ứng viên lửa lấy mẫu cố định và toàn bộ 101 ứng viên âm tính. Sau đó xem ảnh 640 px của 100 ứng viên lửa và 45 ứng viên âm tính lọt vòng sơ bộ. Kết quả cuối: **75 ảnh chỉ lửa, 29 ảnh âm tính khó** được nhận; 237 ảnh đã xem bị loại. Các ảnh còn lại trong hàng đợi chưa được xem đầy đủ và **không được nhận mặc định**. Lý do loại được ghi theo từng ảnh ở `runtime/catargiu_audit/review_pages/visual_review.csv`: khói/lửa chưa gắn nhãn, watermark, text phủ lên ảnh, cảnh mơ hồ hoặc trùng ứng viên với D-Fire.

Bản mới tại `runtime/dfire_clean_v2_plus_pyro_catargiu_v1/` có train **7.987** ảnh, val **1.371**, test **4.249**. Đợt này bổ sung 83 box lửa; train có 3.003 ảnh chứa lửa và 4.705 ảnh chứa khói. Nhãn nguồn `fire=0` được đổi thành `fire=1` theo thứ tự của D-Fire, còn ảnh `other` thành background với file nhãn rỗng. Manifest nhận ảnh và toàn bộ quyết định kiểm duyệt nằm trong `metadata/`. Từng file ảnh và nhãn ở val/test đã được so SHA-256 với bản D-Fire + Pyro trước khi chốt.

Kiểm tra trùng trong nhóm nhận bằng SHA, dHash và pHash; 4 cặp còn ở ngưỡng dHash≤8 nhưng pHash từ 18 đến 34 đã được xem cạnh nhau trong `runtime/catargiu_audit/review_pages/remaining_hash_pairs.jpg` và là các cảnh khác nhau. Các ứng viên gần D-Fire ở ngưỡng bảo thủ được loại, kể cả một số cặp hash có thể là dương tính giả. **Không có bằng chứng để khẳng định sạch tuyệt đối mọi cảnh trùng**, vì ảnh đã bị stretch/tái mã hóa và nguồn thiếu định danh video cho nhiều ảnh. Bộ mới phù hợp cho một thí nghiệm đối chứng, chưa giải quyết hoàn toàn thiếu lửa, khói đêm hoặc false positive trên video người dùng. Cần chạy model trên val/test cố định và video âm tính; đánh giá cuối vẫn cần holdout camera/video mới chưa xem.

Mã tái hiện: `scripts/catargiu_audit.py`, `scripts/catargiu_analyze.py`, `scripts/catargiu_review_queue.py`, `scripts/catargiu_contact_sheets.py`, `scripts/catargiu_review_pages.py`, `scripts/catargiu_full_review.py`, `scripts/catargiu_finalize_review.py`, `scripts/catargiu_merge_selected.py`. Chỉ mục đầy đủ, cặp hash và nhóm tên cảnh nằm tại `runtime/catargiu_audit/`.
