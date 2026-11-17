# Kiểm duyệt nguồn Pyro-SDIS v2.0 cho D-Fire sạch v2

Nguồn: [pyronear/pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis), revision cố định `ebfc02d0cba7f7dfe0623d13bcf0f0aa8a38e3ee`. Nguồn công bố Apache-2.0. Mã đọc toàn bộ ảnh và nhãn: `scripts/pyro_sdis_audit.py`; mã chốt ứng viên: `scripts/pyro_sdis_select.py`; mã xuất dữ liệu: `scripts/pyro_sdis_merge.py`.

## Toàn bộ nguồn

| Mục | Kết quả |
|---|---:|
| Ảnh | 12.378 |
| Ảnh có box khói | 10.744 |
| Ảnh nhãn rỗng | 1.634 |
| Box khói | 12.137 |
| Camera theo `partner/camera` | 213 |
| Camera xuất hiện ở cả split train và val gốc | 87 |
| Ảnh có vấn đề nhãn/byte | 188 |
| Ảnh trùng byte trong chính nguồn | 159 |
| Box vượt biên | 30 box |
| Trùng byte với toàn bộ chỉ mục D-Fire gốc | 0 |

Nguồn có nhiều chuỗi camera và box rất nhỏ: 11.060/12.137 box nhỏ hơn 0,5% diện tích ảnh. Split train/val do nguồn cung cấp không bảo đảm tách camera, nên không giữ nguyên split đó khi gộp với D-Fire. D-Fire val/test vẫn cố định.

## Quyết định đợt 1

Sau khi xem các contact sheet kèm ô phóng to box và đối chiếu với lỗi của checkpoint D-Fire hiện tại, chỉ nhận **22 ảnh khói** có box nhìn được và **51 ảnh âm tính khó**, thuộc **65 camera**. Các ảnh âm tính gồm sương, mây thấp, mây sáng, hoàng hôn và cảnh mờ/ánh sáng từng tạo dự đoán khói giả. Một số âm tính có điểm khói cao với model hiện tại; điểm model chỉ dùng để ưu tiên xem, không dùng làm ground truth.

Danh sách ứng viên đã nhận và khoảng cách hash tới D-Fire ở `accepted_provisional.csv`. Không có ảnh chọn nào trùng byte hoặc là cặp dHash≤6 đồng thời pHash≤12 với chỉ mục D-Fire gốc; giữa 73 ảnh chọn cũng không có cặp theo cùng điều kiện. Hash không chứng minh hết trùng cùng cảnh. Nguồn camera và việc xem ảnh là kiểm tra bổ sung. Ảnh khói rất mờ hoặc âm tính có khả năng thực chất là khói bị loại khỏi bản này. Toàn bộ ảnh Pyro được chọn đưa vào train; không đưa video kiểm thử của người dùng vào train.

Bản gộp `runtime/dfire_clean_v2_plus_pyro_v1/` có train **7.883**, val **1.371**, test **4.249**. Val/test là bản sao byte của D-Fire sạch v2. ZIP để tải lên Kaggle là `runtime/dfire_clean_v2_plus_pyro_v1.zip`.

Theo manifest của D-Fire sạch v2, train trước khi bổ sung có 2.928 ảnh chứa lửa và 4.683 ảnh chứa khói. Sau đợt này: **2.928 ảnh chứa lửa, 4.705 ảnh chứa khói**. Chênh lệch còn 1.777 ảnh; đây là tín hiệu để ưu tiên tìm nguồn lửa, không phải chỉ tiêu phải lấp bằng mọi giá.

## Giới hạn và bước tiếp

Đây là đợt bổ sung nhỏ có kiểm soát, không phải bằng chứng model đã tốt hơn. Pyro-SDIS không có nhãn lửa và hầu như chỉ có camera ngoài trời. Chưa giải quyết thiếu ảnh lửa, khói ban đêm hoặc môi trường trong nhà. Cần huấn luyện cùng recipe với baseline để đo tác động trên validation/test cố định và video âm tính; không dùng test để chọn epoch hoặc ngưỡng. Do đã dùng lỗi của test D-Fire để định hướng, cần thêm holdout camera/video mới chưa xem để kết luận độc lập cuối cùng.
