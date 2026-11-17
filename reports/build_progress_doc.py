from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path

ROOT = Path(r'F:\6\KLTN\Real_Time_Smoke_Fire_Detection_YOLO')
OUT = ROOT / 'reports' / 'noi-dung-bao-cao-tien-do-FLARE-15-slide.docx'
IMG_IOU = Path(r'C:\Users\User\AppData\Local\Temp\codex-clipboard-2592d729-1884-4c8b-9548-4825cd727834.png')
IMG_SYSTEM = Path(r'C:\Users\User\AppData\Local\Temp\codex-clipboard-4a7881dd-6841-425a-aeee-ab94cefe2c51.png')
IMG_DATA = Path(r'C:\Users\User\AppData\Local\Temp\codex-clipboard-38c98a5a-524e-4e00-a277-541b8bc89f19.png')

doc = Document()
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width = Inches(13.333)
sec.page_height = Inches(7.5)
sec.top_margin = Inches(0.48)
sec.bottom_margin = Inches(0.42)
sec.left_margin = Inches(0.58)
sec.right_margin = Inches(0.58)

styles = doc.styles
normal = styles['Normal']
normal.font.name = 'Aptos'
normal.font.size = Pt(11)
normal.font.color.rgb = RGBColor(35, 48, 56)
normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Aptos')
normal.paragraph_format.space_after = Pt(5)
normal.paragraph_format.line_spacing = 1.05
for name, size, color in [('Title', 30, '17364A'), ('Heading 1', 22, '17364A'), ('Heading 2', 14, 'B44937')]:
    st = styles[name]
    st.font.name = 'Aptos Display' if name != 'Heading 2' else 'Aptos'
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor.from_string(color)
    st._element.rPr.rFonts.set(qn('w:eastAsia'), st.font.name)
    st.paragraph_format.space_before = Pt(0)
    st.paragraph_format.space_after = Pt(8 if name != 'Heading 2' else 4)

# Footer is intentionally omitted so image-only pipeline pages contain no text.

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)

def set_cell_text(cell, text, bold=False, color=None, size=9.5):
    cell.text = ''
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(1)
    run = p.add_run(str(text))
    run.bold = bold
    run.font.name = 'Aptos'
    run.font.size = Pt(size)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Aptos')
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(headers, rows, widths=None, font_size=9.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, True, 'FFFFFF', font_size)
        shade(table.rows[0].cells[i], '17364A')
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value, False, None, font_size)
            if row_index % 2 == 1:
                shade(cells[i], 'EEF3F5')
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table

def p(text='', bold_prefix=None, size=None, color=None, italic=False):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(5)
    if bold_prefix and text.startswith(bold_prefix):
        a = para.add_run(bold_prefix)
        a.bold = True
        b = para.add_run(text[len(bold_prefix):])
        runs = (a,b)
    else:
        runs = (para.add_run(text),)
    for run in runs:
        run.font.name = 'Aptos'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Aptos')
        if size: run.font.size = Pt(size)
        if color: run.font.color.rgb = RGBColor.from_string(color)
        run.italic = italic
    return para

def bullet(text):
    para = doc.add_paragraph(style='List Bullet')
    para.paragraph_format.left_indent = Inches(0.22)
    para.paragraph_format.first_line_indent = Inches(-0.12)
    para.paragraph_format.space_after = Pt(3)
    run = para.add_run(text)
    run.font.name = 'Aptos'; run.font.size = Pt(10.5)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Aptos')
    return para

def slide(number, title, subtitle=None):
    if number > 1:
        doc.add_page_break()
    if title:
        h = doc.add_paragraph(style='Heading 1')
        h.paragraph_format.keep_with_next = True
        h.add_run(f'{number:02d}  {title}')
    if subtitle:
        q = doc.add_paragraph()
        q.paragraph_format.space_after = Pt(8)
        r = q.add_run(subtitle); r.italic = True; r.font.color.rgb = RGBColor(92,112,124); r.font.size = Pt(10.5)

def image_only(path, alt):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)
    run = para.add_run()
    inline = run.add_picture(str(path), width=Inches(11.6))
    inline._inline.docPr.set('descr', alt)

# Slide 1
h = doc.add_paragraph(style='Title')
h.paragraph_format.space_before = Pt(85)
h.alignment = WD_ALIGN_PARAGRAPH.CENTER
h.add_run('BÁO CÁO TIẾN ĐỘ FLARE')
p('Dữ liệu V6, YOLO11n và ứng dụng web', size=18, color='B44937').alignment = WD_ALIGN_PARAGRAPH.CENTER
p('Cập nhật kết quả đã thực hiện và các điều chỉnh so với báo cáo trước', size=12).alignment = WD_ALIGN_PARAGRAPH.CENTER
p('Khóa luận tốt nghiệp  |  Nguyễn Thanh Quyền  |  MSSV 21045131', size=11).alignment = WD_ALIGN_PARAGRAPH.CENTER
p('Giảng viên hướng dẫn: ThS. Nguyễn Hữu Tình, TS. Nguyễn Hữu Vũ', size=11).alignment = WD_ALIGN_PARAGRAPH.CENTER

# Slide 2
slide(2, 'Các cập nhật chính so với báo cáo trước')
bullet('Bộ dữ liệu được cập nhật thành V6: 15.195 ảnh từ 6 nguồn; DBA-Fire được loại khỏi bộ dữ liệu do trùng lặp nhiều với D-Fire.')
bullet('Baseline YOLO11n được đánh giá lại trên tập Test V6 đã khử trùng lặp: mAP50 = 0,749 và mAP50–95 = 0,423.')
bullet('Đã khảo sát ngưỡng IoU của NMS trong khoảng 0,3–0,8; cấu hình hiện chọn 0,6 theo tiêu chí cân bằng nhiều chỉ số.')
bullet('Web FLARE đã có các luồng cốt lõi: tải video, xử lý nền, xem video kết quả và sự kiện, cảnh báo Zalo tùy chọn, phản hồi và giao diện quản trị theo vai trò.')
bullet('ViT + YOLO11n, chatbot RAG và camera trực tiếp chưa có kết quả hoàn tất trong bản demo hiện tại.')
p('Phạm vi demo: phát hiện trên video người dùng tải lên; chưa giám sát camera trực tiếp.', bold_prefix='Phạm vi demo:')

# Slide 3
slide(3, 'Nguồn dữ liệu và quy ước nhãn', 'Bộ V6 thay số liệu và vai trò nguồn so với lần báo cáo trước.')
add_table(['Nguồn', 'Vai trò', 'Thu thập', 'Giữ lại'], [
    ['D-Fire', 'Dương tính chính: khói và lửa', '21.527', '9.852'],
    ['Indoor Fire Smoke', 'Dương tính trong nhà', '5.000', '2.898'],
    ['AIForMankind_V2', 'Dương tính, chủ yếu khói', '2.191', '344'],
    ['AIForMankind_Cloud', 'Hard-negative: mây dễ nhầm khói', '1.080', '739'],
    ['FLAME No-Fire', 'Ảnh nền', '17.837', '1.302'],
    ['BoWFire Not-Fire', 'Ảnh nền', '107', '60'],
    ['Tổng', '6 nguồn; DBA-Fire không thuộc V6', '47.742', '15.195'],
], widths=[3.4,5.0,1.7,1.7], font_size=10)
p('Nhãn thống nhất: 0 = khói, 1 = lửa. Nhãn Indoor Fire Smoke đã được đảo thứ tự và kiểm tra trực quan. Ảnh nền dùng nhãn rỗng; AIForMankind_Cloud được giữ làm hard-negative.', size=10.5)
p('[ẢNH CHO NOTEBOOKLM: tải từ Kaggle output ảnh report_dataset_overview... nếu muốn minh họa phân bố dữ liệu.]', size=9, color='637986', italic=True)

# Slide 4: image only
slide(4, None)
image_only(IMG_DATA, 'Pipeline 11 bước xử lý và chia bộ dữ liệu V6')

# Slide 5
slide(5, 'Khử trùng lặp và kiểm soát rò rỉ dữ liệu')
p('Các frame liền kề trong cùng video có thể gần như giống nhau. Nếu chia ngẫu nhiên từng ảnh, nhóm ảnh này có thể xuất hiện đồng thời trong Train và Test, khiến kết quả đánh giá lạc quan hơn.')
bullet('Tạo metadata toàn cục cho 47.742 ảnh; kiểm tra ảnh lỗi và đối chiếu ảnh với nhãn. Không có ảnh không đọc được.')
bullet('SHA256 tìm trùng tuyệt đối; dHash với Hamming ≤ 3 tạo ứng viên gần trùng. Histogram màu và tỉ lệ khung hình được dùng để xác minh cặp.')
bullet('Leader Clustering gom ảnh tương tự, tránh hiệu ứng bắc cầu; chia Train/Validation/Test theo nhóm trùng lặp với tỉ lệ 70/15/15.')
bullet('Loại 26.295 ảnh trùng/gần trùng và thêm 6.252 ảnh nền vượt hạn mức; phát hiện thêm 14 ảnh trùng chéo nguồn.')
bullet('Còn 406 cặp gần giống ở mức thấp nằm chéo giữa các tập; giới hạn này đã được ghi nhận trong báo cáo cảnh báo dữ liệu.')
p('Kết quả quy mô: 47.742 → 15.195 ảnh V6 sau khử trùng lặp và cân bằng ảnh nền.', bold_prefix='Kết quả quy mô:')

# Slide 6
slide(6, 'Phân bố Train, Validation và Test')
add_table(['Tập', 'Chỉ lửa', 'Chỉ khói', 'Cả hai', 'Nền', 'Tổng'], [
    ['Train', '1.437', '3.979', '3.027', '2.459', '10.902'],
    ['Validation', '284', '779', '591', '516', '2.170'],
    ['Test', '280', '799', '600', '444', '2.123'],
    ['Tổng', '2.001', '5.557', '4.218', '3.419', '15.195'],
], widths=[2.2,1.8,1.8,1.8,1.8,1.8], font_size=10)
bullet('Trong Train, số ảnh có khói so với số ảnh có lửa xấp xỉ 1,57:1; khoảng 37% box thuộc nhóm box nhỏ.')
bullet('Ảnh nền chiếm khoảng 22,5% theo bảng phân bố các tập. Các ảnh “cả hai” chứa đồng thời khói và lửa.')
p('[ẢNH CHO NOTEBOOKLM: tải biểu đồ phân bố nhãn hoặc box từ Kaggle output, ví dụ report_metrics_by_class... nếu có.]', size=9, color='637986', italic=True)

# Slide 7
slide(7, 'Thiết lập huấn luyện baseline YOLO11n V6')
bullet('Kích thước ảnh 832 × 832, nhằm giữ thêm chi tiết cho các box nhỏ.')
bullet('AdamW, batch 32, lịch học Cosine; tối đa 200 epoch và có dừng sớm.')
bullet('Huấn luyện dừng ở epoch 133; checkpoint tốt nhất tại epoch 113. Thời gian khoảng 6,96 giờ trên GPU T4.')
bullet('Dùng trọng số loss lớp cls = 0,57 để xử lý lệch lớp; không nhân bản ảnh.')
bullet('Augmentation gồm Mosaic, MixUp, CutMix, HSV, lật ngang và scale; tắt Mosaic trong 15 epoch cuối.')
p('[ẢNH CHO NOTEBOOKLM: tải đường cong loss/metric từ Kaggle output, tên bắt đầu report_training_curves_... hoặc results.png.]', size=9, color='637986', italic=True)

# Slide 8
slide(8, 'Kết quả YOLO11n trên tập Test V6', 'IoU đánh giá = 0,6 theo thiết lập báo cáo; đây là kết quả trên ảnh có nhãn.')
add_table(['Lớp', 'Precision', 'Recall', 'mAP50', 'mAP50–95'], [
    ['Khói', '0,816', '0,738', '0,802', '0,487'],
    ['Lửa', '0,721', '0,621', '0,696', '0,358'],
    ['Trung bình', '0,768', '0,680', '0,749', '0,423'],
], widths=[3.4,2.1,2.1,2.1,2.1], font_size=10.5)
bullet('Khói có kết quả cao hơn lửa ở các chỉ số chính; phát hiện lửa vẫn là phần cần cải thiện.')
bullet('mAP50 của V6 thấp hơn baseline cũ 0,797. Tập Test V6 đã khử trùng lặp nên có thể khó hơn; không so sánh trực tiếp nếu tập hoặc giao thức đánh giá khác nhau.')
p('Precision, Recall và mAP là chỉ số trên ảnh có nhãn; chúng không đo trực tiếp độ chính xác cảnh báo theo sự kiện của video người dùng.', size=10)
p('[ẢNH CHO NOTEBOOKLM: tải biểu đồ theo lớp từ Kaggle output report_metrics_by_class... hoặc report_test_metrics_yolo... .]', size=9, color='637986', italic=True)

# Slide 9
slide(9, 'Khảo sát ngưỡng IoU và heuristic chọn cấu hình', 'Bảng dưới là khảo sát ngưỡng IoU của NMS trong suy luận, không phải confidence threshold.')
add_table(['IoU NMS', 'Precision', 'Recall', 'mAP50', 'mAP50–95'], [
    ['0,3', '0,7687', '0,6849', '0,7279', '0,4023'],
    ['0,4', '0,7644', '0,6867', '0,7424', '0,4098'],
    ['0,5', '0,7627', '0,6886', '0,7499', '0,4169'],
    ['0,6', '0,7683', '0,6796', '0,7492', '0,4226'],
    ['0,7', '0,7623', '0,6757', '0,7416', '0,4247'],
    ['0,8', '0,7489', '0,6644', '0,7245', '0,4219'],
], widths=[1.7,2.1,2.1,2.1,2.3], font_size=9.5)
p('IoU của NMS quyết định mức chồng lấp giữa hai box dự đoán để xử lý box trùng: ngưỡng cao hơn cho phép giữ lại các box chồng lấp nhiều hơn; ngưỡng thấp hơn loại box mạnh tay hơn.', size=9.8)
p('Heuristic chọn 0,6: chọn điểm vận hành cân bằng nhiều chỉ số thay vì tối đa hóa một chỉ số riêng. Ở 0,6, Precision = 0,7683 gần mức cao nhất 0,7687; mAP50 = 0,7492 gần cao nhất 0,7499; mAP50–95 = 0,4226 gần cao nhất 0,4247. Recall giảm nhẹ so với mức cao nhất 0,6886 tại 0,5.', size=9.8)
p('Đây là lựa chọn thực nghiệm có tính cân bằng, không phải nghiệm tối ưu được chứng minh bằng hàm mục tiêu có trọng số. Cần tiếp tục đối chiếu trên video dương tính và âm tính.', size=9.8)
p('[ẢNH BẢNG: ảnh khảo sát ngưỡng IoU đính kèm; có thể tải bản tương ứng từ Kaggle output report_iou_sensitivity_... .]', size=9, color='637986', italic=True)

# Slide 10
slide(10, 'Tối ưu suy luận và thử nghiệm ViT + YOLO11n')
p('TTA: tăng mAP50–95 khoảng 0,006 nhưng suy luận chậm hơn khoảng 1,7 lần, nên không bật trong cấu hình triển khai ban đầu.')
p('ONNX: export ở kích thước 832 × 832, đầu ra (1, 6, N), gồm 4 tọa độ và 2 lớp. Trước khi thay engine web cần so sánh kết quả ONNX với PyTorch trên cùng đầu vào.')
p('ViT + YOLO11n: YOLO11n đảm nhận định vị đặc trưng cục bộ; ViT bổ sung ngữ cảnh toàn khung. Đặc trưng ViT được đưa vào P3/P4/P5, chiếu 1×1 và kết hợp qua cổng học được.')
bullet('Code đã chạy thử trên GPU với ViT-Small 24,4 triệu tham số và ViT-Base 88,7 triệu tham số.')
bullet('Chưa có số liệu đánh giá hoàn tất để kết luận mô hình kết hợp tốt hơn YOLO11n V6; cần so sánh trên cùng tập Test và các video âm tính test_3, test_4, test_5.')
p('[ẢNH CHO NOTEBOOKLM: nếu có, tải biểu đồ so sánh TTA từ report_tta_comparison_... và sơ đồ/đường cong của ViT từ output Kaggle.]', size=9, color='637986', italic=True)

# Slide 11
slide(11, 'Thử nghiệm trên sáu video')
add_table(['Video', 'Frame có detection', 'Chuyển trạng thái', 'Độ trễ/frame', 'Ước tính FPS'], [
    ['test_1', '88,5%', '20 (7,0%)', '8,5 ms', '117,2'],
    ['test_2', '100%', '0 (0%)', '7,9 ms', '126,2'],
    ['test_3', '41,9%', '33 (11,2%)', '8,1 ms', '122,7'],
    ['test_4', '82,8%', '27 (8,3%)', '7,8 ms', '128,3'],
    ['test_5', '57,6%', '8 (3,2%)', '8,8 ms', '114,0'],
    ['test_6', '100%', '0 (0%)', '8,2 ms', '122,4'],
], widths=[1.4,2.8,2.5,2.3,2.3], font_size=9.5)
p('test_2 và test_6 là video cháy được phát hiện ổn định; test_1 còn dao động. test_3 (hoàng hôn), test_4 và test_5 (mây/sương núi) có báo khói nhầm.', size=10)
p('Độ trễ và FPS trong bảng phản ánh suy luận theo frame; FPS được ước tính bằng nghịch đảo độ trễ, không phải tốc độ toàn luồng từ upload đến xuất video.', size=9.5)
p('[ẢNH CHO NOTEBOOKLM: tải ảnh/video minh họa kết quả có box từ Kaggle hoặc dùng ảnh test_1–test_6 đã lưu; ghi chú rõ test_3–test_5 là ca báo nhầm.]', size=9, color='637986', italic=True)

# Slide 12
slide(12, 'Heuristic tổng hợp sự kiện theo thời gian')
p('Bộ AlertSmoother gom detection theo từng lớp trong cửa sổ trượt. Hệ thống chỉ mở sự kiện khi đủ số mẫu và tỉ lệ frame dương tính đạt ngưỡng; sau đó đóng sự kiện khi tỉ lệ giảm, bỏ sự kiện quá ngắn và gộp các đoạn gần nhau.')
add_table(['Tham số hiện tại', 'Giá trị', 'Ý nghĩa'], [
    ['window_ms', '1.200 ms', 'Cửa sổ quan sát các frame gần nhất'],
    ['trigger_ratio', '0,6', 'Tỉ lệ frame dương tính để mở sự kiện'],
    ['release_ratio', '0,2', 'Tỉ lệ để đóng sự kiện đang hoạt động'],
    ['min_event_ms', '600 ms', 'Thời lượng tối thiểu để giữ sự kiện'],
    ['merge_gap_ms', '500 ms', 'Gộp hai đoạn cùng lớp nếu cách nhau không quá mức này'],
], widths=[2.6,1.7,6.8], font_size=9.5)
p('Đây là heuristic làm ổn định sự kiện, không phải cơ chế hiểu ngữ cảnh hình ảnh. Kiểm thử hiện có cho thấy detection rời rạc không tạo sự kiện, detection kéo dài tạo sự kiện; false positive kéo dài vẫn có thể tạo cảnh báo.', size=10)
p('Các ngưỡng cần được tinh chỉnh bằng video dương tính lẫn âm tính; tăng cửa sổ không đảm bảo loại được mây hoặc sương bị nhận nhầm liên tục.', size=10)

# Slide 13: image only
slide(13, None)
image_only(IMG_SYSTEM, 'Sơ đồ pipeline tổng thể hệ thống FLARE, gồm web, backend video, hai vai trò admin, vòng đời model, RAG và hướng camera tương lai')

# Slide 14
slide(14, 'Các chức năng web đã hoàn thành')
p('Luồng user: đăng ký/đăng nhập và quản lý hồ sơ; cấu hình bot Zalo tùy chọn; tải một video cho mỗi tác vụ; theo dõi xử lý nền; xem MP4 có bounding box, timeline sự kiện, ảnh bằng chứng và lịch sử của chính mình.')
p('Video đầu vào hiện giới hạn 30 giây, 50 MB, MP4, tối đa 3840 × 2160 và 60 FPS; worker xử lý tuần tự, giảm frame tối đa về 1280 × 720 trước suy luận/xuất video. Zalo chỉ gửi sau khi job hoàn tất.')
p('Phản hồi: user có thể báo false positive, false negative, sai lớp hoặc sai box tại frame; ảnh gửi tới admin mô hình để kiểm tra nhãn trên ảnh gốc. Có thể vẽ/sửa/xóa nhiều box khói-lửa hoặc xác nhận ảnh âm tính, rồi xuất gói YOLO đã duyệt kèm manifest.')
p('Admin vận hành: xem luồng sự kiện, lọc theo ngày, mở chi tiết video, quản lý user và duyệt kích hoạt/rollback model. Admin mô hình: xử lý phản hồi, duyệt nhãn và đăng ký model ứng viên. Hai vai trò có giao diện và quyền riêng trong cùng ứng dụng.')
p('Backend dùng FastAPI, worker riêng và SQLite; video, ảnh bằng chứng, gói dữ liệu và model lưu trên đĩa. Bot Token và chat ID Zalo được mã hóa ở backend. Web chưa kết nối camera trực tiếp; chatbot RAG chưa triển khai.', size=10)
p('[ẢNH CHO NOTEBOOKLM: dùng ảnh chụp hiện tại của trang user, luồng admin vận hành và màn hình gắn nhãn admin mô hình nếu cần minh họa.]', size=9, color='637986', italic=True)

# Slide 15
slide(15, 'Vòng phản hồi và công việc tiếp theo')
p('Vòng cải thiện model: user gửi phản hồi → admin mô hình kiểm tra frame gốc và sửa nhãn → xuất ảnh, nhãn YOLO và manifest → người phụ trách dữ liệu khử trùng, chia tập theo video và huấn luyện ngoài web → đánh giá trên Test cố định cùng video âm tính → admin vận hành duyệt kích hoạt hoặc rollback.')
bullet('Hoàn tất đánh giá ViT + YOLO11n trên cùng tập Test với baseline; kiểm tra riêng các video test_3, test_4 và test_5.')
bullet('Bổ sung hard-negative cho mây, sương và hoàng hôn; đo false positive trước/sau trên bộ video cố định.')
bullet('Tiếp tục kiểm tra giới hạn 406 cặp gần giống còn chéo tập và cập nhật báo cáo dữ liệu khi có kết quả mới.')
bullet('Sau khi web cốt lõi ổn định, tích hợp chatbot RAG trả lời từ tài liệu có nguồn dẫn. Chatbot không thay detector và không phát hoặc xác nhận cảnh báo.')
bullet('Camera trực tiếp/RTSP là hướng phát triển sau; không thuộc demo video tải lên hiện tại.')

# Metadata
doc.core_properties.title = 'Nội dung báo cáo tiến độ FLARE trong 15 slide'
doc.core_properties.subject = 'Dữ liệu V6, YOLO11n, heuristic IoU và hệ thống web FLARE'
doc.core_properties.author = 'FLARE'
doc.save(OUT)
print(OUT)

