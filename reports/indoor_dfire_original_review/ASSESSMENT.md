# Đánh giá Indoor Fire Smoke để bổ sung D-Fire gốc đã làm sạch

Ngày 11/11/2026. Nguồn được kiểm tra trực tiếp: `F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip`. Đích đối chiếu: `runtime/dfire_original_clean_split_v1.zip` (train 12.699, val 4.244, test 4.244; `0=smoke`, `1=fire`). Chưa nhập ảnh hoặc train.

## Số lượng và tình trạng kỹ thuật

- ZIP qua CRC. Cả 5.000 ảnh giải mã được và có nhãn tương ứng. Split nguồn: train 3.500, valid 750, test 750. Nhãn nguồn `0=fire`, `1=smoke` dựa trên kiểm tra box/ảnh; `data.yaml` chỉ gọi tên `0`, `1`, nên phải đảo ID nếu chọn ảnh để thêm vào D-Fire.
- Toàn nguồn có 2.165 ảnh chỉ-lửa, 2.376 chỉ-khói, 459 cả hai; không có ảnh nền. Trong nhóm chỉ-lửa, 1.808 ảnh có cạnh dài ít nhất 640 px; đây chỉ là tiêu chí ưu tiên, vì YOLO `imgsz=640` vẫn có thể resize ảnh kích thước khác.
- Kiểm tra hình học với dung sai `1e-5` phát hiện 32 dòng box lỗi ở 31 ảnh (chủ yếu box vượt biên rất ít do làm tròn): 1 chỉ-lửa, 25 chỉ-khói, 5 cả hai. Nếu loại nguyên cặp ảnh/nhãn như quy trình D-Fire, pool chỉ-lửa còn 2.164 trước các bước duyệt khác. Không tự sửa nhãn ở đây.
- Không có ảnh chỉ-lửa trùng byte trong chính pool này. Bỏ hậu tố Roboflow `.rf.<hash>` còn 1.881 stem trong 2.165 ảnh; 456 ảnh thuộc 172 nhóm stem lặp. Có 98 nhóm stem xuất hiện qua nhiều split nguồn (276 ảnh). Stem không phải mã cảnh đáng tin cậy, nên các con số này chỉ là cờ trùng/lặp.

## Trùng với D-Fire và chất lượng cảnh

- So SHA-256 toàn bộ 2.165 ảnh chỉ-lửa với 21.187 ảnh D-Fire hiện tại: **0 cặp trùng byte**.
- So dHash/pHash của toàn bộ hai pool: **5 cặp** đạt đồng thời dHash ≤6 và pHash ≤12; thực tế cả 5 có khoảng cách 0/0 và ảnh ghép xác nhận cùng khung hình qua mã hóa/cỡ ảnh khác. Ba bản Indoor khớp một ảnh **D-Fire test** `PublicDataset01326.jpg`; hai bản khớp D-Fire train. Không có cặp đạt ngưỡng này với D-Fire val. Xem `perceptual_candidates.csv` và `cross_dfire_pairs.jpg`. Ngưỡng hash không chứng minh đã bắt hết cảnh cùng video hoặc khác góc.
- Mẫu 120 ảnh chỉ-lửa, lấy sau lọc cạnh dài ≥640, stem không lặp trong mẫu và bỏ 5 bản khớp D-Fire, nằm trong `fire_only_sample_01.jpg` đến `fire_only_sample_06.jpg` với danh sách ở `fire_only_sample_manifest.csv`. Quan sát định tính: có camera trong phòng/bếp và cháy phòng đáng xem; đồng thời có nhiều thí nghiệm/cảnh dàn dựng, lửa nhỏ trong bát hoặc trên que, bếp gas, clip có chữ và nhiều khung từ cùng video/camera. Mẫu không đủ để suy ra chính xác bao nhiêu ảnh đạt chuẩn sau duyệt. Ảnh tổng hợp và nhãn bỏ sót khói vẫn cần xem từng ảnh gốc.
- Chỉ mục Indoor cũ tại `reports/indoor_fire_audit/` ghi 972/2.165 ảnh chỉ-lửa tham gia ít nhất một cặp dHash ≤1, pHash ≤4 trong nội bộ nguồn; cặp hash là ứng viên gần trùng, không tự động xóa toàn bộ. Báo cáo cũ còn ghi trùng nguồn với CQUniversity; không cộng hai pool nếu chưa đối chiếu lại.

## Kết luận cho đợt bổ sung này

Indoor **không phù hợp để nhập hàng loạt hoặc dùng làm nguồn chính nhằm lấp thiếu hụt khoảng 1.000 ảnh chỉ-lửa đa dạng**. Số ảnh chỉ-lửa thô 2.165 giảm sau lọc kỹ thuật, cảnh lặp và duyệt ngữ nghĩa; mẫu cho thấy nhiều cảnh lửa có kiểm soát. Nguồn **có thể dùng chọn lọc** cho một số cảnh camera trong nhà thực tế sau khi duyệt ảnh gốc/box, loại khói chưa gắn nhãn và khung cùng cảnh, loại mọi cảnh trùng hoặc gần trùng D-Fire val/test, rồi đảo ID nhãn. Giữ nguyên val/test D-Fire. Chưa có danh sách ảnh Indoor được duyệt để nhập train; không suy số lượng ảnh có thể dùng từ các thống kê thô.

## Sàng lọc đầy đủ ngày 11/11/2026

Sau đánh giá sơ bộ, đã tạo 28 bảng ảnh cho **toàn bộ 2.165 ảnh chỉ-lửa** ở `full_review/page_01.jpg` đến `page_28.jpg` bằng `scripts/indoor_dfire_full_review_sheets.py`. Đã xem toàn bộ bảng ảnh để nhận diện chuỗi video, lửa có kiểm soát, ảnh tổng hợp, ảnh nhỏ và cảnh đáng xem lớn. Đã chọn 53 ảnh có triển vọng để xem lớn (`full_review/candidate_page_01.jpg` đến `candidate_page_05.jpg`), rồi mở ảnh gốc của danh sách cuối với box lửa. Ảnh ứng viên được so trực quan với D-Fire gần nhất theo hash (`full_review/candidate_cross_01.jpg` đến `candidate_cross_03.jpg`); các cặp gần nhất trong danh sách cuối là cảnh khác nhau. Quyết định thủ công và điều kiện kỹ thuật ở `scripts/finalize_indoor_dfire_review.py`; kết quả từng ảnh ở `curation_manifest.csv`, danh sách đề xuất ở `proposed_images.csv`, số liệu ở `curation_summary.json`. Notebook `notebooks/data_preparation/22_review_indoor_for_dfire.ipynb` kiểm chứng danh sách và hiển thị ảnh có box. Không sửa ZIP nguồn hoặc D-Fire.

| Trạng thái | Ảnh | Ý nghĩa |
|---|---:|---|
| Đề xuất cảnh camera/hiện trường | 3 | `I0626`, `I0810`, `I1748`; ba cảnh khác nhau, box nguồn hợp lệ, không thấy khói thiếu nhãn rõ ở ảnh gốc |
| Tùy chọn nếu chấp nhận cảnh đốt thử trong phòng thí nghiệm | 2 | `I1091`, `I1173`; ảnh camera thật, có lửa rõ nhưng không đại diện sự cố tự nhiên |
| Cần xem/sửa nhãn hoặc chất lượng thêm | 10 | Có khói nghi thiếu nhãn, box chưa bao hết lửa hoặc hình ảnh khó xác nhận; chưa tính là phù hợp |
| Trùng khung hình với D-Fire | 5 | Bao gồm 3 bản cùng ảnh test; loại khỏi train đề xuất |
| Box lỗi hình học quá ngưỡng dung sai `1e-5` | 1 | Loại theo quy trình giữ nguyên nhãn nguồn |
| Không được ưu tiên sau khi xem bảng toàn nguồn | 2.144 | Phần lớn là khung lặp/cảnh đốt thử/ảnh kém phù hợp; trạng thái này không chứng minh từng ảnh vô dụng cho mọi mục tiêu |
| **Tổng** | **2.165** | |

Danh sách đề xuất hiện có **5 ảnh với 8 box lửa**; tất cả từ split `train` của nguồn, cạnh dài ≥640 px, không có cặp gần trùng nội bộ ở ngưỡng dHash≤6 và pHash≤12, không khớp khung D-Fire theo ngưỡng đó. Nếu chỉ lấy nhóm sát mục tiêu thực tế, số là **3 ảnh**. Nếu nhận cả hai cảnh phòng thí nghiệm, D-Fire train chỉ-lửa tăng từ 686 lên 691 ảnh; tác động cân bằng rất nhỏ. Đây là số ảnh **đã xác nhận phù hợp theo bộ tiêu chí bảo thủ hiện tại**, không phải ước lượng số tối đa có thể thu được nếu sửa nhãn hoặc nới tiêu chí. Chưa gộp; chủ dự án quyết định bước tiếp theo.

## Quyết định của chủ dự án

Ngày 11/11/2026, chủ dự án quyết định **không bổ sung Indoor vào D-Fire** vì 3 ảnh đề xuất chính và 2 ảnh thí nghiệm tùy chọn quá ít để tạo khác biệt đáng kể. `proposed_images.csv` chỉ lưu bằng chứng sàng lọc, không phải danh sách đã phê duyệt để gộp. Không tạo bản train mới hoặc train từ nguồn này; D-Fire train/val/test giữ nguyên.
