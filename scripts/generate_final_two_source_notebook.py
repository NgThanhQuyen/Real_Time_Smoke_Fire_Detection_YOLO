"""Tạo notebook phân tích và tái lập bộ dữ liệu hai nguồn."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/data_preparation/16_build_final_two_source_dataset.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(r"""
# Xây dựng bộ dữ liệu Fire–Smoke từ hai nguồn

**Nguồn:** `Fire-Smoke-Detection-Yolov11.zip` và `D-Fire-YOLO.zip`  
**Đầu ra:** một ZIP YOLO với `0 = Fire`, `1 = Smoke`, ba tập train/valid/test.  
**Mục tiêu chia tập:** xấp xỉ 8:1:1, đồng thời giữ lại các ảnh của bộ gốc và các ảnh D-Fire đã dùng trong thí nghiệm 2.000 ảnh khói.

Notebook này lưu toàn bộ thống kê và biểu đồ phục vụ báo cáo. File `selection_manifest.jsonl` ghi quyết định chọn cho từng ảnh: nguồn, SHA-256, tập, nhãn cuối và tên trong ZIP. Phần tạo manifest dùng thuật toán ở `scripts/build_final_two_source_dataset.py`; manifest đã đóng băng để việc dựng lại từ hai ZIP gốc cho đúng cùng kết quả. Các ảnh gần giống nhưng khác byte không bị tự động loại. Các ảnh có SHA-256 trùng tuyệt đối không được xuất hai lần, đặc biệt không xuất xuyên tập.
"""),
    markdown(r"""
## 1. Cấu hình và đọc manifest

Khi chạy trên máy khác hoặc Kaggle, chỉnh ba đường dẫn nguồn dưới đây và đưa `selection_manifest.jsonl` đi cùng notebook. Để xây lại ZIP, đặt `BUILD_ZIP = True`; mặc định notebook chỉ kiểm tra và phân tích ZIP đã tạo, không ghi đè dữ liệu.
"""),
    code(r"""
from pathlib import Path
from collections import Counter, defaultdict
from io import BytesIO
import hashlib, json, zipfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path.cwd()
while not (ROOT / 'codex.md').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / 'codex.md').exists():
    raise FileNotFoundError('Hãy đặt ROOT tới thư mục dự án chứa codex.md.')

MAIN_ZIP = Path(r'F:\6\KLTN\Smoke-Fire-Dataset\Fire-Smoke-Detection-Yolov11.zip')
DFIRE_ZIP = Path(r'F:\6\KLTN\Smoke-Fire-Dataset\D-Fire-YOLO.zip')
REPORT = ROOT / 'reports/final_two_source_dataset'
MANIFEST = REPORT / 'selection_manifest.jsonl'
FINAL_ZIP = ROOT / 'runtime/fire_smoke_dfire_final.zip'
FIGURES = REPORT / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
BUILD_ZIP = False

rows = [json.loads(line) for line in MANIFEST.read_text(encoding='utf-8').splitlines() if line]
data = pd.DataFrame([{key: value for key, value in row.items() if key != 'label_text'} for row in rows])
print(f'Số ảnh trong manifest: {len(rows):,}')
print('Nguồn:', ', '.join(sorted(data.source_dataset.unique())))
"""),
    markdown(r"""
## 2. Quy tắc xử lý

1. Giữ 10.459 ảnh chính sau khi loại bốn bản trùng byte đã được phát hiện trước đây; không loại ảnh gần giống và giữ nguyên nhãn gốc, kể cả dòng polygon hợp lệ.
2. Giữ 2.000 ảnh chỉ khói D-Fire và 1.404 ảnh nền D-Fire đã có trong bản thí nghiệm. Bỏ 296 ảnh Fog/Cloud để bản cuối chỉ còn hai nguồn.
3. Bổ sung 176 ảnh chỉ khói và 796 ảnh nền D-Fire vào train. Trong 796 ảnh nền, 400 ảnh được ưu tiên theo điểm báo nhầm từ model cũ, 396 ảnh chọn bằng thứ tự băm có seed cố định. 296 ảnh trong nhóm này thay Fog/Cloud, 500 ảnh làm tăng quy mô train.
4. Bổ sung 695 ảnh D-Fire vào valid và 1.131 ảnh vào test từ các split D-Fire sạch tương ứng. Việc chọn dùng thứ tự băm có seed `20261108`. Nhãn D-Fire được đổi `0 = Smoke, 1 = Fire` sang `0 = Fire, 1 = Smoke`.
5. Mỗi ảnh có SHA-256 duy nhất trong manifest cuối. Trước khi đóng ZIP, mã đọc và giải mã từng ảnh, so hash nguồn, ghi ảnh và file nhãn, rồi kiểm tra CRC của ZIP.

Đây là cách kiểm soát **ảnh trùng đúng byte** giữa các tập theo tiêu chí đã chốt. Hash cảm nhận hoặc việc ảnh cùng bối cảnh không được dùng để tự xóa ảnh khác byte.
"""),
    code(r"""
assert len(rows) == 16661
assert data.source_dataset.value_counts().to_dict() == {
    'Fire-Smoke-Detection-Yolov11': 10459, 'D-Fire': 6202
}
assert data.source_sha256.nunique() == len(data), 'Có ảnh trùng byte trong manifest.'
assert data.output_image.nunique() == len(data)
assert data.output_label.nunique() == len(data)
print('Không có SHA-256 hoặc tên đầu ra lặp trong manifest.')
print(pd.crosstab(data['split'], data['source_dataset']).to_string())
"""),
    markdown(r"""
## 3. Dựng lại ZIP từ hai ZIP nguồn

Đặt `BUILD_ZIP = True` ở phần cấu hình nếu muốn dựng lại. Quy trình lấy byte ảnh trực tiếp từ hai ZIP nguồn và nhãn cuối đã ghi trong manifest; không phụ thuộc các thư mục data trung gian đã xóa. File tạm chỉ được đổi tên thành ZIP cuối sau khi ghi xong và kiểm tra CRC. Nếu đã có ZIP cuối, phải chọn một đường dẫn đầu ra khác để giữ an toàn.
"""),
    code(r"""
def verify_pixels(blob, name):
    with Image.open(BytesIO(blob)) as image:
        image.verify()
    with Image.open(BytesIO(blob)) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f'Kích thước ảnh không hợp lệ: {name}')

def build_zip_from_manifest(rows, main_zip, dfire_zip, target):
    if target.exists():
        raise FileExistsError(f'Đã có đầu ra: {target}')
    temporary = target.with_suffix('.zip.building')
    if temporary.exists():
        raise FileExistsError(f'File tạm tồn tại: {temporary}')
    yaml = 'train: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames:\n  0: Fire\n  1: Smoke\n'
    info = ('# Bộ dữ liệu Fire và Smoke cho khóa luận\n\n'
            'Bộ YOLO này được tổng hợp từ hai nguồn ảnh: Fire-Smoke-Detection-Yolov11 '
            'và D-Fire. Nhãn đầu ra: 0 = Fire, 1 = Smoke. Ảnh nền có file nhãn rỗng.\n\n'
            'Ảnh trùng byte được kiểm tra bằng SHA-256. Danh sách nguồn, split, hash '
            'và nhãn cuối nằm trong metadata/selection_manifest.jsonl. '
            'Quy trình tái lập và phân tích nằm trong notebook của dự án.\n')
    try:
        with zipfile.ZipFile(main_zip) as main, zipfile.ZipFile(dfire_zip) as dfire, \
             zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=4, allowZip64=True) as output:
            output.writestr('data.yaml', yaml)
            output.writestr('DATASET_INFO.md', info)
            for row in rows:
                source = main if row['source_dataset'] == 'Fire-Smoke-Detection-Yolov11' else dfire
                blob = source.read(row['source_image'])
                digest = hashlib.sha256(blob).hexdigest()
                if digest != row['source_sha256']:
                    raise ValueError(f'Ảnh nguồn đổi nội dung: {row["source_image"]}')
                verify_pixels(blob, row['source_image'])
                output.writestr(row['output_image'], blob)
                output.writestr(row['output_label'], row['label_text'].encode('utf-8'))
            manifest_text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
            output.writestr('metadata/selection_manifest.jsonl', manifest_text.encode('utf-8'))
        with zipfile.ZipFile(temporary) as output:
            corrupt = output.testzip()
            if corrupt:
                raise ValueError(f'Lỗi CRC trong ZIP: {corrupt}')
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

if BUILD_ZIP:
    build_zip_from_manifest(rows, MAIN_ZIP, DFIRE_ZIP, FINAL_ZIP)
    print('Đã tạo:', FINAL_ZIP)
else:
    print('Dùng ZIP cuối hiện có:', FINAL_ZIP)
"""),
    markdown("## 4. Kiểm tra ZIP thành phẩm"),
    code(r"""
with zipfile.ZipFile(FINAL_ZIP) as archive:
    names = set(archive.namelist())
    expected = {'data.yaml', 'DATASET_INFO.md', 'metadata/selection_manifest.jsonl'}
    expected.update(data.output_image)
    expected.update(data.output_label)
    assert names == expected, f'Sai số file: thiếu {len(expected-names)}, thừa {len(names-expected)}'
    assert archive.read('metadata/selection_manifest.jsonl').decode('utf-8').splitlines() == MANIFEST.read_text(encoding='utf-8').splitlines()
    assert archive.testzip() is None
    sample = data.sample(n=min(120, len(data)), random_state=20261108)
    for row in sample.itertuples():
        assert hashlib.sha256(archive.read(row.output_image)).hexdigest() == row.source_sha256
        assert archive.read(row.output_label).decode('utf-8') == rows[row.Index]['label_text']
print(f'ZIP đạt CRC, đủ {len(data):,} cặp ảnh–nhãn; 120 ảnh mẫu khớp SHA và nhãn.')
print(f'Kích thước ZIP: {FINAL_ZIP.stat().st_size / 1024**3:.2f} GiB')
"""),
    markdown("## 5. Thống kê ảnh, lớp và nguồn"),
    code(r"""
order = ['train', 'valid', 'test']
classes = ['fire_only', 'smoke_only', 'both', 'background']
table = pd.crosstab(data['split'], data['category']).reindex(index=order, columns=classes, fill_value=0)
table['total'] = table.sum(axis=1)
table['with_fire'] = table['fire_only'] + table['both']
table['with_smoke'] = table['smoke_only'] + table['both']
display(table)
display((table['total'] / table['total'].sum() * 100).round(2).rename('Tỷ lệ (%)'))
"""),
    code(r"""
label_rows = Counter()
polygon_rows = Counter()
box_areas = defaultdict(list)
for row in rows:
    for line in row['label_text'].splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        label_rows[(row['split'], cls)] += 1
        if len(parts) > 5:
            polygon_rows[row['split']] += 1
        elif len(parts) == 5:
            box_areas[cls].append(float(parts[3]) * float(parts[4]))
        else:
            raise ValueError(f'Dòng nhãn sai cấu trúc: {line}')
boxes = pd.DataFrame({
    'Fire': [label_rows[(split, 0)] for split in order],
    'Smoke': [label_rows[(split, 1)] for split in order],
    'Polygon rows': [polygon_rows[split] for split in order]
}, index=order)
display(boxes)
"""),
    markdown(r"""
## 6. Biểu đồ cho báo cáo

Biểu đồ được lưu dưới `reports/final_two_source_dataset/figures/` và cũng hiển thị ngay trong notebook. Mỗi biểu đồ chỉ dùng số liệu của manifest cuối. Ảnh nền là ảnh có nhãn rỗng, không phải lớp YOLO thứ ba.
"""),
    code(r"""
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180, 'font.size': 10})
palette = ['#df5d43', '#688fa5', '#7d688f', '#b8b9ad']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
table[classes].plot.bar(stacked=True, ax=axes[0], color=palette, width=0.68)
axes[0].set_title('Thành phần ảnh theo tập')
axes[0].set_xlabel('Tập dữ liệu'); axes[0].set_ylabel('Số ảnh')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(['Chỉ lửa', 'Chỉ khói', 'Cả hai', 'Nền'], frameon=False)
axes[1].pie(table['total'], labels=order, autopct='%1.1f%%',
            colors=['#344e5b', '#a3b5a9', '#e2af72'], startangle=90)
axes[1].set_title('Tỷ lệ train / valid / test')
fig.savefig(FIGURES / '01_split_and_categories.png', bbox_inches='tight')
plt.show()
"""),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
table[['with_fire', 'with_smoke']].plot.bar(ax=axes[0], color=['#df5d43', '#688fa5'], width=0.68)
axes[0].set_title('Ảnh có lửa và ảnh có khói')
axes[0].set_ylabel('Số ảnh'); axes[0].set_xlabel('Tập dữ liệu')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(['Có lửa', 'Có khói'], frameon=False)
boxes[['Fire', 'Smoke']].plot.bar(ax=axes[1], color=['#df5d43', '#688fa5'], width=0.68)
axes[1].set_title('Số dòng nhãn theo lớp')
axes[1].set_ylabel('Số dòng'); axes[1].set_xlabel('Tập dữ liệu')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(frameon=False)
fig.savefig(FIGURES / '02_class_presence_and_labels.png', bbox_inches='tight')
plt.show()
"""),
    code(r"""
sources = pd.crosstab(data['split'], data['source_dataset']).reindex(order).fillna(0)
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
sources.plot.bar(stacked=True, ax=ax, color=['#344e5b', '#d3a66a'], width=0.68)
ax.set_title('Nguồn ảnh trong từng tập')
ax.set_ylabel('Số ảnh'); ax.set_xlabel('Tập dữ liệu')
ax.tick_params(axis='x', rotation=0)
ax.legend(frameon=False)
fig.savefig(FIGURES / '03_sources_by_split.png', bbox_inches='tight')
plt.show()
"""),
    markdown(r"""
## 7. Kết luận cho báo cáo

- Bộ final có **16.661 ảnh**: 13.329 train, 1.666 valid, 1.666 test, gần đúng **80% / 10% / 10%**.
- Chỉ dùng hai nguồn ảnh. Có **10.459** ảnh từ Fire-Smoke-Detection-Yolov11 và **6.202** ảnh từ D-Fire.
- Số ảnh chứa lửa là **7.479 / 935 / 935**; số ảnh chứa khói là **5.364 / 671 / 671** cho train/valid/test. Tỷ lệ tổng và tỷ lệ ảnh có từng lớp gần 8:1:1. Các nhóm con như ảnh chỉ lửa, cả hai hoặc nền không đồng thời đạt tỷ lệ đó.
- Ảnh nền có nhãn rỗng. Các nhãn polygon hợp lệ của nguồn chính được giữ và được thống kê riêng với box.
- Không có ảnh trùng byte SHA-256 trong manifest. ZIP được kiểm tra CRC, ảnh được giải mã trong lúc đóng gói. Kiểm tra này không chứng minh rằng mọi ảnh cùng cảnh nhưng khác byte đều độc lập; theo quy tắc đã chốt, chúng được giữ.
"""),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
