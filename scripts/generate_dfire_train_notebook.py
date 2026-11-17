"""Tạo notebook Kaggle độc lập để fine-tune YOLO11n trên D-Fire sạch v2."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from textwrap import dedent


OUTPUT = Path(__file__).resolve().parents[1] / "notebooks/data_preparation/10_train_yolo11n_dfire_clean_v2.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip().splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # Fine-tune YOLO11n pretrained trên D-Fire sạch v2

        Notebook này dùng đúng bộ `dfire_clean_v2_complete.zip` (7.810 train, 1.371 val, 4.249 test), không huấn luyện từ trọng số ngẫu nhiên. Bật **GPU** và **Internet** trong Kaggle trước khi chạy. Có thể gắn ZIP làm Kaggle Dataset; notebook tự tìm bản đã giải nén hoặc giải nén ZIP vào `/kaggle/working`.

        Quy trình: kiểm tra dữ liệu → tải `yolo11n.pt` pretrained COCO → thích nghi với backbone đóng băng ngắn hạn → fine-tune toàn bộ với learning rate thấp → chọn checkpoint theo validation → đánh giá test cố định, tách theo nguồn và đếm cảnh nền bị báo nhầm. Test không dùng để chọn checkpoint, hyperparameter hay confidence.

        Cấu hình là một **baseline có kiểm soát**, không phải lời khẳng định tối ưu trước khi có kết quả thực nghiệm. YOLO11n được chọn để phù hợp mục tiêu suy luận trên laptop CPU; nếu đổi model size cần đo tốc độ và điểm val trên cùng dữ liệu. Cơ sở: [YOLO11 pretrained](https://docs.ultralytics.com/models/yolo11/), [training và freeze/early stopping](https://docs.ultralytics.com/modes/train/), [augmentation và close mosaic](https://docs.ultralytics.com/guides/yolo-data-augmentation/), [nghiên cứu YOLO11n trên hai benchmark lửa/khói và nhiều seed](https://www.mdpi.com/2571-6255/9/9/386), [nghiên cứu về trùng/rò rỉ tập ảnh](https://arxiv.org/abs/2304.02296), [tổng quan mất cân bằng trong object detection](https://arxiv.org/abs/1909.00169). D-Fire sạch v2 còn thiếu ảnh lửa độc lập và âm tính khó; notebook không dùng oversampling để tạo thêm bản sao ảnh. Sau run đầu, nếu muốn kết luận ổn định, nên lặp thêm seed trên cùng train/val rồi so trên test khóa một lần cho mỗi model đã chốt.
        """
    ),
    code(
        r"""
        %pip install -q ultralytics==8.4.177
        """
    ),
    code(
        r"""
        from collections import Counter, defaultdict
        from pathlib import Path, PurePosixPath
        from zipfile import ZipFile, is_zipfile
        import csv
        import hashlib
        import json
        import platform
        import shutil

        import torch
        import ultralytics
        import yaml
        from ultralytics import YOLO

        INPUT_ROOT = Path('/kaggle/input')
        WORK_ROOT = Path('/kaggle/working')
        DATASET_ROOT_OVERRIDE = ''  # Điền đường dẫn thư mục nếu Kaggle có nhiều bộ D-Fire.
        DATASET_ZIP_OVERRIDE = ''   # Điền đường dẫn ZIP nếu notebook không tự tìm thấy.
        PRETRAINED_WEIGHTS = 'yolo11n.pt'
        RUN_ROOT = WORK_ROOT / 'dfire_clean_v2_yolo11n'
        IMG_SIZE = 640
        BATCH_SIZE = 16
        WORKERS = 2
        SEED = 42
        EXPECTED_COUNTS = {'train': 7810, 'val': 1371, 'test': 4249}
        EXPECTED_SOURCE_SHA256 = 'b0dd9fec2b37f319b79f7880f15fc87bc90bd4efb917c0311298209536f62f0b'

        if not torch.cuda.is_available():
            raise RuntimeError('Hãy bật GPU trong Kaggle Settings trước khi train.')
        DEVICE = 0
        print('Python:', platform.python_version())
        print('PyTorch:', torch.__version__)
        print('Ultralytics:', ultralytics.__version__)
        print('GPU:', torch.cuda.get_device_name(0))
        """
    ),
    code(
        r"""
        def valid_dataset_root(root: Path) -> bool:
            return (root / 'data.yaml').is_file() and (root / 'metadata/summary.json').is_file()


        def archive_is_dfire(path: Path) -> bool:
            if not is_zipfile(path):
                return False
            with ZipFile(path) as archive:
                names = set(archive.namelist())
                return 'data.yaml' in names and 'metadata/summary.json' in names


        def extract_checked(path: Path, destination: Path) -> Path:
            if destination.exists():
                if valid_dataset_root(destination):
                    return destination
                raise RuntimeError(f'Thư mục giải nén tồn tại nhưng thiếu dữ liệu: {destination}')
            destination.mkdir(parents=True)
            with ZipFile(path) as archive:
                for member in archive.infolist():
                    relative = PurePosixPath(member.filename)
                    if relative.is_absolute() or '..' in relative.parts or ':' in member.filename:
                        raise RuntimeError(f'Đường dẫn không an toàn trong ZIP: {member.filename}')
                archive.extractall(destination)
            return destination


        if DATASET_ROOT_OVERRIDE:
            DATASET_ROOT = Path(DATASET_ROOT_OVERRIDE).resolve()
        else:
            candidates = [p.parent.parent for p in INPUT_ROOT.rglob('metadata/summary.json')
                          if valid_dataset_root(p.parent.parent)]
            if len(candidates) == 1:
                DATASET_ROOT = candidates[0].resolve()
            elif len(candidates) > 1:
                raise RuntimeError(f'Có nhiều bộ dữ liệu: {candidates}. Hãy đặt DATASET_ROOT_OVERRIDE.')
            else:
                zip_candidates = [Path(DATASET_ZIP_OVERRIDE)] if DATASET_ZIP_OVERRIDE else [
                    p for p in INPUT_ROOT.rglob('*.zip') if archive_is_dfire(p)]
                if len(zip_candidates) != 1:
                    raise RuntimeError(f'Cần đúng một ZIP D-Fire, tìm được: {zip_candidates}. '
                                       'Hãy đặt DATASET_ZIP_OVERRIDE.')
                DATASET_ROOT = extract_checked(zip_candidates[0], WORK_ROOT / 'dfire_clean_v2_data')

        if not valid_dataset_root(DATASET_ROOT):
            raise FileNotFoundError(f'Thiếu data.yaml hoặc metadata/summary.json: {DATASET_ROOT}')
        source_yaml = yaml.safe_load((DATASET_ROOT / 'data.yaml').read_text(encoding='utf-8'))
        if source_yaml.get('names') not in ({0: 'smoke', 1: 'fire'}, ['smoke', 'fire']):
            raise RuntimeError('Thứ tự lớp trong data.yaml không phải 0=smoke, 1=fire.')
        summary = json.loads((DATASET_ROOT / 'metadata/summary.json').read_text(encoding='utf-8'))
        if summary.get('source_zip_sha256') != EXPECTED_SOURCE_SHA256:
            raise RuntimeError('Sai phiên bản ZIP D-Fire gốc.')
        split_stems = {}
        for split, expected in EXPECTED_COUNTS.items():
            if summary.get(split) != expected:
                raise RuntimeError(f'Sai số ảnh {split}: {summary.get(split)} != {expected}')
            images = list((DATASET_ROOT / split / 'images').glob('*'))
            labels = list((DATASET_ROOT / split / 'labels').glob('*.txt'))
            if len(images) != expected or len(labels) != expected:
                raise RuntimeError(f'Thiếu ảnh/nhãn ở {split}: {len(images)}/{len(labels)}')
            image_stems = {p.stem for p in images}
            label_stems = {p.stem for p in labels}
            if len(image_stems) != expected or image_stems != label_stems:
                raise RuntimeError(f'Tên ảnh/nhãn không khớp ở {split}')
            split_stems[split] = image_stems
            for label in labels:
                for line in label.read_text(encoding='utf-8-sig').splitlines():
                    parts = line.split()
                    if len(parts) != 5 or parts[0] not in {'0', '1'}:
                        raise RuntimeError(f'Nhãn sai: {label}')
                    x, y, w, h = map(float, parts[1:])
                    if not (0 < w <= 1 and 0 < h <= 1 and
                            -0.000001 <= x - w / 2 <= x + w / 2 <= 1.000001 and
                            -0.000001 <= y - h / 2 <= y + h / 2 <= 1.000001):
                        raise RuntimeError(f'Box không hợp lệ: {label}')
            print(split, expected)
        if any(split_stems[a] & split_stems[b] for a, b in
               (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise RuntimeError('Trùng tên ảnh giữa các tập.')

        DATA_YAML = WORK_ROOT / 'dfire_clean_v2_train.yaml'
        DATA_YAML.write_text(
            f'path: {json.dumps(DATASET_ROOT.as_posix())}\n'
            'train: train/images\nval: val/images\ntest: test/images\n'
            'names:\n  0: smoke\n  1: fire\n', encoding='utf-8')
        print('Nguồn:', DATASET_ROOT)
        print('YAML dùng train:', DATA_YAML)
        """
    ),
    markdown(
        """
        ## Cấu hình huấn luyện

        Giai đoạn 1 đóng băng 10 lớp đầu trong 12 epoch để phần detection học hai lớp mới. Giai đoạn 2 mở toàn bộ model, giảm learning rate và train tối đa 100 epoch với early stopping 25 epoch. Augmentation màu và hình học được giữ vừa phải vì màu/biên khói, lửa là tín hiệu quan trọng; không dùng mixup, cutmix hoặc lật dọc. Mosaic được tắt trong 10 epoch cuối. Không cân bằng bằng cách nhân bản ảnh.

        Các con số này có thể đổi **sau khi xem validation**, nhưng phải ghi run mới và không tham khảo test để chọn. Test chỉ được đánh giá với checkpoint đã chọn.
        """
    ),
    code(
        r"""
        base_model = YOLO(PRETRAINED_WEIGHTS)
        pretrained_names = [str(name).lower() for name in base_model.names.values()]
        print('Checkpoint pretrained:', PRETRAINED_WEIGHTS)
        print('Số lớp pretrained:', len(pretrained_names))
        print('Lớp fire/smoke trùng tên chính xác:', sorted({'fire', 'smoke'} & set(pretrained_names)))
        if len(pretrained_names) != 80:
            raise RuntimeError('Checkpoint không giống YOLO11n COCO pretrained 80 lớp.')

        COMMON_TRAIN = dict(
            data=str(DATA_YAML), imgsz=IMG_SIZE, batch=BATCH_SIZE, workers=WORKERS,
            device=DEVICE, project=str(RUN_ROOT), optimizer='AdamW',
            seed=SEED, deterministic=True, amp=True, cache=False,
            save=True, plots=True, val=True, verbose=True,
            hsv_h=0.01, hsv_s=0.35, hsv_v=0.25,
            degrees=0.0, translate=0.05, scale=0.30,
            fliplr=0.5, flipud=0.0, perspective=0.0,
            mixup=0.0, cutmix=0.0,
        )

        STAGE1_DIR = RUN_ROOT / 'stage1_frozen'
        STAGE1_BEST = STAGE1_DIR / 'weights/best.pt'
        if not STAGE1_BEST.is_file():
            base_model.train(
                **COMMON_TRAIN, name='stage1_frozen', exist_ok=False,
                epochs=12, patience=12, freeze=10,
                lr0=0.001, lrf=0.1, cos_lr=True, warmup_epochs=2,
                mosaic=0.2, close_mosaic=0, save_period=-1,
            )
        if not STAGE1_BEST.is_file():
            raise FileNotFoundError(f'Không thấy checkpoint giai đoạn 1: {STAGE1_BEST}')
        print('Stage 1 best:', STAGE1_BEST)
        """
    ),
    code(
        r"""
        STAGE2_DIR = RUN_ROOT / 'stage2_full'
        STAGE2_BEST = STAGE2_DIR / 'weights/best.pt'
        STAGE2_LAST = STAGE2_DIR / 'weights/last.pt'
        RESUME_STAGE2 = False  # Đổi True khi Kaggle giữ lại last.pt của run bị ngắt.

        if RESUME_STAGE2:
            if not STAGE2_LAST.is_file():
                raise FileNotFoundError(f'Thiếu checkpoint để resume: {STAGE2_LAST}')
            YOLO(str(STAGE2_LAST)).train(resume=True)
        elif STAGE2_DIR.exists():
            raise RuntimeError(f'Run giai đoạn 2 đã tồn tại: {STAGE2_DIR}. '
                               'Để tiếp tục, đặt RESUME_STAGE2=True; để chạy mới, đổi RUN_ROOT.')
        else:
            YOLO(str(STAGE1_BEST)).train(
                **COMMON_TRAIN, name='stage2_full', exist_ok=False,
                epochs=100, patience=25, freeze=0,
                lr0=0.0003, lrf=0.05, cos_lr=True, warmup_epochs=2,
                mosaic=0.5, close_mosaic=10, save_period=10,
            )
        if not STAGE2_BEST.is_file():
            raise FileNotFoundError(f'Không thấy checkpoint giai đoạn 2: {STAGE2_BEST}')
        print('Stage 2 best:', STAGE2_BEST)
        """
    ),
    code(
        r"""
        def validate_checkpoint(path: Path, run_name: str):
            metrics = YOLO(str(path)).val(
                data=str(DATA_YAML), split='val', imgsz=IMG_SIZE,
                batch=BATCH_SIZE, workers=WORKERS, device=DEVICE,
                project=str(RUN_ROOT), name=run_name, exist_ok=True,
                plots=True, verbose=False,
            )
            return metrics


        stage1_val = validate_checkpoint(STAGE1_BEST, 'validation_stage1')
        stage2_val = validate_checkpoint(STAGE2_BEST, 'validation_stage2')
        stage1_score = float(stage1_val.box.map)
        stage2_score = float(stage2_val.box.map)
        SELECTED_CHECKPOINT = STAGE2_BEST if stage2_score >= stage1_score else STAGE1_BEST
        print('Validation mAP50-95 stage1:', round(stage1_score, 4))
        print('Validation mAP50-95 stage2:', round(stage2_score, 4))
        print('Chọn theo validation:', SELECTED_CHECKPOINT)
        """
    ),
    markdown(
        """
        ## Đánh giá cuối

        Các cell tiếp theo chấm **một checkpoint đã chọn** trên test. Kết quả theo nguồn dùng cùng trọng số, cùng cấu hình suy luận; không chọn lại model hay ngưỡng sau khi xem test. Tỉ lệ ảnh nền báo nhầm được đo ở confidence 0,25 đã định trước, giúp xác định loại âm tính khó cần bổ sung. Đây là chỉ số theo ảnh, chưa phải tỉ lệ cảnh báo sai theo video/sự kiện.
        """
    ),
    code(
        r"""
        selected_model = YOLO(str(SELECTED_CHECKPOINT))

        def metric_summary(metrics):
            box = metrics.box
            names = metrics.names
            class_map = box.maps.tolist() if hasattr(box.maps, 'tolist') else list(box.maps)
            return {
                'precision_mean': float(box.mp), 'recall_mean': float(box.mr),
                'mAP50': float(box.map50), 'mAP50_95': float(box.map),
                'mAP50_95_by_class': {str(names[i]): float(class_map[i]) for i in range(len(class_map))},
            }


        overall_test = selected_model.val(
            data=str(DATA_YAML), split='test', imgsz=IMG_SIZE,
            batch=BATCH_SIZE, workers=WORKERS, device=DEVICE,
            project=str(RUN_ROOT), name='test_all', exist_ok=True,
            plots=True, verbose=False,
        )
        test_results = {'all': metric_summary(overall_test)}

        with (DATASET_ROOT / 'metadata/manifest.csv').open('r', encoding='utf-8-sig', newline='') as stream:
            test_rows = [row for row in csv.DictReader(stream) if row['output_split'] == 'test']
        if len(test_rows) != EXPECTED_COUNTS['test']:
            raise RuntimeError('Manifest test không khớp tập test cố định.')

        for source in ('WEB', 'PublicDataset', 'AoF'):
            source_rows = [row for row in test_rows if row['source_prefix'] == source]
            list_path = WORK_ROOT / f'dfire_test_{source}.txt'
            list_path.write_text(''.join(str(DATASET_ROOT / row['output_image']) + '\n'
                                         for row in source_rows), encoding='utf-8')
            source_yaml = WORK_ROOT / f'dfire_test_{source}.yaml'
            source_yaml.write_text(
                f'path: {json.dumps(DATASET_ROOT.as_posix())}\n'
                'train: train/images\nval: val/images\n'
                f'test: {json.dumps(list_path.as_posix())}\n'
                'names:\n  0: smoke\n  1: fire\n', encoding='utf-8')
            source_metrics = selected_model.val(
                data=str(source_yaml), split='test', imgsz=IMG_SIZE,
                batch=BATCH_SIZE, workers=WORKERS, device=DEVICE,
                project=str(RUN_ROOT), name=f'test_{source}', exist_ok=True,
                plots=False, verbose=False,
            )
            test_results[source] = {'images': len(source_rows), **metric_summary(source_metrics)}

        print(json.dumps(test_results, ensure_ascii=False, indent=2))
        """
    ),
    code(
        r"""
        BACKGROUND_CONF = 0.25
        backgrounds = [row for row in test_rows if row['output_category'] == 'background']
        false_alarm = defaultdict(lambda: Counter({'images': 0, 'any_detection': 0,
                                                     'smoke_detection': 0, 'fire_detection': 0}))
        for start in range(0, len(backgrounds), 32):
            batch_rows = backgrounds[start:start + 32]
            paths = [str(DATASET_ROOT / row['output_image']) for row in batch_rows]
            predictions = list(selected_model.predict(
                source=paths, imgsz=IMG_SIZE, conf=BACKGROUND_CONF, iou=0.7,
                device=DEVICE, batch=BATCH_SIZE, stream=True, verbose=False,
            ))
            for row, result in zip(batch_rows, predictions):
                classes = {int(value) for value in result.boxes.cls.tolist()}
                record = false_alarm[row['source_prefix']]
                record['images'] += 1
                record['any_detection'] += bool(classes)
                record['smoke_detection'] += 0 in classes
                record['fire_detection'] += 1 in classes

        background_results = {}
        for source, counter in false_alarm.items():
            background_results[source] = {
                **dict(counter),
                'false_alarm_image_rate': counter['any_detection'] / counter['images'],
            }
        print('Báo nhầm trên ảnh nền, conf=0.25:')
        print(json.dumps(background_results, ensure_ascii=False, indent=2))
        """
    ),
    code(
        r"""
        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()


        BEST_EXPORT = WORK_ROOT / 'dfire_clean_v2_yolo11n_best.pt'
        shutil.copy2(SELECTED_CHECKPOINT, BEST_EXPORT)
        report = {
            'source_zip_sha256': summary['source_zip_sha256'],
            'dataset_counts': EXPECTED_COUNTS,
            'class_names': {0: 'smoke', 1: 'fire'},
            'pretrained_weights': PRETRAINED_WEIGHTS,
            'selected_checkpoint': str(SELECTED_CHECKPOINT),
            'best_export': str(BEST_EXPORT),
            'best_export_sha256': sha256(BEST_EXPORT),
            'python': platform.python_version(),
            'pytorch': torch.__version__,
            'ultralytics': ultralytics.__version__,
            'gpu': torch.cuda.get_device_name(0),
            'imgsz': IMG_SIZE, 'batch': BATCH_SIZE, 'seed': SEED,
            'stage1_validation_map50_95': stage1_score,
            'stage2_validation_map50_95': stage2_score,
            'test_metrics': test_results,
            'background_confidence': BACKGROUND_CONF,
            'background_false_alarms': background_results,
            'note': 'Test chỉ dùng để báo cáo; chưa có đánh giá video âm tính theo sự kiện.',
        }
        REPORT_PATH = WORK_ROOT / 'dfire_clean_v2_train_report.json'
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Tải checkpoint:', BEST_EXPORT)
        print('Tải báo cáo:', REPORT_PATH)
        """
    ),
]

for index, cell in enumerate(cells):
    source = "".join(cell["source"])
    cell["id"] = hashlib.sha1(f"{index}:{source}".encode("utf-8")).hexdigest()[:12]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
