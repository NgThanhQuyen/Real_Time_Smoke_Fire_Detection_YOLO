"""Extend the generated FinalSFD notebook with controlled experiments."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/model_training/train_yolo11n_final_sfd.ipynb"
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source(index):
    return "".join(nb["cells"][index]["source"])


def replace(index, content):
    nb["cells"][index]["source"] = content.splitlines(keepends=True)


def md(content):
    return {"cell_type": "markdown", "metadata": {}, "source": content.splitlines(keepends=True)}


def code(content):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": content.splitlines(keepends=True)}


replace(0, """# Huấn luyện và đánh giá YOLO11n trên FinalSFD

Notebook đọc FinalSFD đã chốt từ Kaggle Input, class 0=smoke và 1=fire. Dữ liệu nguồn không bị sửa; nhãn hình học lỗi được sửa trên bản sao và ghi báo cáo.

Quy trình: thống kê ảnh/box → minh họa model COCO pretrained → fine-tune YOLO11n ở `imgsz=640` → tùy chọn train từ đầu (scratch) → đánh giá val/test → khảo sát NMS IoU 0.3/0.5/0.7/0.9 với **0.5 là cấu hình chính** → chọn confidence từ val → đánh giá mức ảnh, kích thước box và ảnh sai trên test → đóng gói kết quả.

`RUN_SCRATCH=False` mặc định vì scratch cần thêm nhiều giờ trên T4. Có thể chạy hai phiên: phiên đầu train transfer và tải ZIP kết quả; phiên sau gắn ZIP đó vào Kaggle Input, đặt `TRANSFER_RESULTS_ZIP` và bật `RUN_SCRATCH=True`. Notebook kiểm tra dataset/config trước khi dùng lại checkpoint. Nếu không chạy scratch, notebook ghi rõ chưa có so sánh định lượng. Dự đoán toàn tập test xử lý từng ảnh trên CPU để tránh CUDA out of memory.

NMS IoU là ngưỡng lọc box dự đoán chồng nhau; nó khác IoU khớp box dùng cho mAP50 và mAP50–95.
""")
replace(4, source(4).replace(
    "NMS_IOU = 0.7",
    "NMS_IOU = 0.5\nIOU_SWEEP = (0.3, 0.5, 0.7, 0.9)\nRUN_SCRATCH = False  # True: train từ đầu để so sánh; tốn thêm nhiều giờ.\nTRANSFER_RESULTS_ZIP = None  # Phiên 2: ZIP kết quả của phiên 1 với notebook này và NMS IoU 0.5."
))
replace(8, '''# Có thể dùng lại transfer checkpoint từ phiên đầu để chạy scratch ở phiên thứ hai.
os.chdir(RUN_DIR)
TRAIN_DIR = RUN_DIR / "train"
BEST_PT = TRAIN_DIR / "weights" / "best.pt"
transfer_reused = TRANSFER_RESULTS_ZIP is not None
if transfer_reused:
    source_results = Path(TRANSFER_RESULTS_ZIP).expanduser().resolve()
    if not source_results.is_file() or source_results.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Không thấy ZIP kết quả transfer: {source_results}")
    with zipfile.ZipFile(source_results) as archive:
        required = {"best.pt", "train/results.csv", "reports/run_summary.json"}
        if not required.issubset(archive.namelist()):
            raise RuntimeError(f"ZIP transfer thiếu: {sorted(required - set(archive.namelist()))}")
        previous = json.loads(archive.read("reports/run_summary.json"))
        if (previous.get("nms_iou_main") != NMS_IOU
                or previous.get("dataset_yaml_sha256") != hashlib.sha256(SOURCE_YAML.read_bytes()).hexdigest()
                or previous.get("dataset_images") != dataset_rows
                or previous.get("local_label_fixes") != label_fix_summary
                or previous.get("training", {}).get("imgsz") != IMAGE_SIZE
                or previous.get("training", {}).get("batch") != BATCH_SIZE
                or previous.get("training", {}).get("seed") != SEED
                or previous.get("training", {}).get("epochs_max") != EPOCHS
                or previous.get("training", {}).get("patience") != PATIENCE):
            raise RuntimeError("ZIP transfer không cùng dataset/cấu hình NMS IoU 0.5, imgsz, batch, seed. "
                               "Hãy dùng ZIP tạo từ phiên đầu của chính notebook mới này.")
        BEST_PT.parent.mkdir(parents=True, exist_ok=True)
        BEST_PT.write_bytes(archive.read("best.pt"))
        (TRAIN_DIR / "results.csv").write_bytes(archive.read("train/results.csv"))
        if "train/results.png" in archive.namelist():
            (TRAIN_DIR / "results.png").write_bytes(archive.read("train/results.png"))
    training_seconds = 0.0
    print("Dùng lại transfer best.pt:", BEST_PT)
else:
    train_started = time.perf_counter()
    train_model = YOLO("yolo11n.pt")
    train_model.train(
        data=str(TRAIN_YAML), epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
        patience=PATIENCE, iou=NMS_IOU, seed=SEED, deterministic=True,
        workers=WORKERS, device=0, project=str(RUN_DIR), name="train",
        exist_ok=True, plots=True, save=True, cache=False,
    )
    training_seconds = time.perf_counter() - train_started
    TRAIN_DIR = Path(train_model.trainer.save_dir).resolve()
    BEST_PT = TRAIN_DIR / "weights" / "best.pt"
    del train_model
    gc.collect()
    torch.cuda.empty_cache()
if not BEST_PT.is_file():
    raise FileNotFoundError(f"Không có checkpoint transfer: {BEST_PT}")
print("Checkpoint transfer:", BEST_PT)
print("Thời gian train transfer trong phiên này (giây):", round(training_seconds, 1))
''')

# Additional cells and edits are assembled below.

size_md = md("""## Phân tích tỷ lệ lớp và kích thước box

Diện tích box là tỷ lệ của ảnh gốc. Nhóm `<1%`, `1–10%`, `≥10%` dùng để chẩn đoán bộ này; không phải nhóm small/medium/large của COCO.
""")
size_code = code('''box_rows = []
for split in ("train", "val", "test"):
    for label_path in (LOCAL_DATASET_ROOT / "labels" / split).glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, _, _, width, height = line.split()
            area = float(width) * float(height)
            size_group = "<1%" if area < 0.01 else "1–10%" if area < 0.10 else "≥10%"
            box_rows.append({"split": split, "class": "smoke" if class_id == "0" else "fire",
                             "normalized_area": area, "size_group": size_group})
box_data = pd.DataFrame(box_rows)
box_size_table = box_data.groupby(["split", "class", "size_group"]).size().rename("boxes").reset_index()
box_size_table.to_csv(RUN_DIR / "reports" / "box_size_distribution.csv", index=False)
area_table = box_data.groupby(["split", "class"])["normalized_area"].agg(
    boxes="size", median_area="median", mean_area="mean"
).reset_index()
area_table.to_csv(RUN_DIR / "reports" / "box_area_summary.csv", index=False)
display(area_table)
display(box_size_table.pivot_table(index=["split", "class"], columns="size_group",
                             values="boxes", fill_value=0))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
dataset_table[["fire_only", "smoke_only", "both", "background"]].div(
    dataset_table["images"], axis=0
).mul(100).plot.bar(stacked=True, ax=axes[0], rot=0)
axes[0].set(title="Tỷ lệ loại ảnh", ylabel="% ảnh")
train_sizes = box_size_table[box_size_table["split"] == "train"].pivot(
    index="class", columns="size_group", values="boxes"
).fillna(0)
train_sizes.div(train_sizes.sum(axis=1), axis=0).mul(100).plot.bar(
    stacked=True, ax=axes[1], rot=0
)
axes[1].set(title="Kích thước box trong train", ylabel="% box")
fig.tight_layout()
fig.savefig(RUN_DIR / "reports" / "data_balance_and_box_size.png", dpi=150)
display(fig)
plt.close(fig)
train_fire_images = int(dataset_table.loc["train", "fire_present"])
train_smoke_images = int(dataset_table.loc["train", "smoke_present"])
train_fire_boxes = int(dataset_table.loc["train", "fire_boxes"])
train_smoke_boxes = int(dataset_table.loc["train", "smoke_boxes"])
train_area = area_table[area_table["split"] == "train"].set_index("class")
imbalance_diagnosis = {
    "fire_present_images": train_fire_images, "smoke_present_images": train_smoke_images,
    "smoke_to_fire_image_ratio": train_smoke_images / train_fire_images,
    "fire_boxes": train_fire_boxes, "smoke_boxes": train_smoke_boxes,
    "fire_to_smoke_box_ratio": train_fire_boxes / train_smoke_boxes,
    "median_fire_box_area": float(train_area.loc["fire", "median_area"]),
    "median_smoke_box_area": float(train_area.loc["smoke", "median_area"]),
}
(RUN_DIR / "reports" / "imbalance_diagnosis.json").write_text(
    json.dumps(imbalance_diagnosis, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Ảnh có khói/lửa:", train_smoke_images, train_fire_images)
print("Box khói/lửa:", train_smoke_boxes, train_fire_boxes)
print("Diện tích box trung vị lửa/khói:", imbalance_diagnosis["median_fire_box_area"],
      imbalance_diagnosis["median_smoke_box_area"])
print("Chưa kết luận nguyên nhân yếu của lớp lửa chỉ từ số ảnh; xem thêm recall theo kích thước ở cuối.")
''')

pretrain_md = md("""## Minh họa checkpoint COCO trước fine-tune

In tên lớp từ chính `yolo11n.pt` và thử trên một nhóm ảnh val cố định. Không tính mAP khói/lửa trực tiếp cho model COCO vì ID 80 lớp của nó khác ID hai lớp FinalSFD. `fire hydrant` là trụ cứu hỏa, không phải ngọn lửa.
""")
pretrain_code = code('''os.chdir(RUN_DIR)
pretrained_model = YOLO("yolo11n.pt")
coco_names = {int(key): str(value) for key, value in pretrained_model.names.items()}
target_names = {"smoke", "fire", "flame"}
present_targets = {key: name for key, name in coco_names.items() if name.lower() in target_names}
print("Số lớp của checkpoint COCO:", len(coco_names))
print("Lớp smoke/fire/flame đúng nghĩa:", present_targets or "không có")
print("Tên lớp chứa từ fire:", [name for name in coco_names.values() if "fire" in name.lower()])
(RUN_DIR / "reports" / "pretrained_class_names.json").write_text(
    json.dumps(coco_names, ensure_ascii=False, indent=2), encoding="utf-8"
)
val_records = image_records[image_records["split"] == "val"]
sample_rng = random.Random(SEED)
pretrained_paths = []
for category in ("fire_only", "smoke_only", "both", "background"):
    candidates = sorted(val_records.loc[val_records["category"] == category, "image"].tolist())
    pretrained_paths.extend(sample_rng.sample(candidates, min(2, len(candidates))))
pretrained_dir = RUN_DIR / "reports" / "pretrained_examples"
pretrained_dir.mkdir(parents=True, exist_ok=True)
pretrained_rows = []
for path in pretrained_paths:
    result = pretrained_model.predict(path, imgsz=IMAGE_SIZE, conf=VIS_CONF, iou=NMS_IOU,
                                      device="cpu", batch=1, verbose=False)[0]
    predicted_names = [result.names[int(class_id)] for class_id in result.boxes.cls.cpu().tolist()]
    output_path = pretrained_dir / Path(path).name
    result.save(filename=str(output_path))
    pretrained_rows.append({"image": Path(path).name,
                            "true_category": val_records.loc[val_records["image"] == path, "category"].iloc[0],
                            "coco_predictions": ", ".join(predicted_names), "annotated_file": output_path.name})
    del result
pretrained_table = pd.DataFrame(pretrained_rows)
pretrained_table.to_csv(RUN_DIR / "reports" / "pretrained_examples.csv", index=False)
display(pretrained_table)
for output_path in sorted(pretrained_dir.iterdir())[:4]:
    display(NotebookImage(filename=str(output_path), width=460))
del pretrained_model
gc.collect()
torch.cuda.empty_cache()
''')

replace(7, """## Huấn luyện YOLO11n transfer hoặc dùng lại checkpoint

Phiên đầu để `TRANSFER_RESULTS_ZIP=None`: train từ `yolo11n.pt` với `imgsz=640`, NMS IoU 0.5 khi chọn best.pt trên val. Nếu cần chạy scratch ở phiên Kaggle thứ hai, gắn ZIP kết quả phiên đầu vào Input, đặt đường dẫn `TRANSFER_RESULTS_ZIP` và bật `RUN_SCRATCH=True`. Notebook kiểm tra nguồn/cấu hình trước khi dùng lại checkpoint; ZIP cũ với IoU 0.7 không được chấp nhận cho so sánh kiểm soát.
""")

scratch_md = md("""## Đối chứng train từ đầu (tùy chọn)

Khi `RUN_SCRATCH=True`, YOLO11n được khởi tạo từ `yolo11n.yaml`, dùng cùng dữ liệu, seed, `imgsz=640`, batch, giới hạn epochs và patience như run fine-tune. Hai checkpoint đều chọn bằng validation.
""")
scratch_code = code('''SCRATCH_PT = None
scratch_training_seconds = None
if RUN_SCRATCH:
    scratch_started = time.perf_counter()
    scratch_model = YOLO("yolo11n.yaml")
    scratch_model.train(
        data=str(TRAIN_YAML), epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
        patience=PATIENCE, seed=SEED, deterministic=True, workers=WORKERS,
        iou=NMS_IOU, pretrained=False,
        device=0, project=str(RUN_DIR), name="scratch", exist_ok=True,
        plots=True, save=True, cache=False,
    )
    scratch_training_seconds = time.perf_counter() - scratch_started
    SCRATCH_PT = Path(scratch_model.trainer.save_dir) / "weights" / "best.pt"
    if not SCRATCH_PT.is_file():
        raise FileNotFoundError(f"Không có checkpoint scratch: {SCRATCH_PT}")
    print("Scratch best.pt:", SCRATCH_PT, "thời gian (giờ):", round(scratch_training_seconds / 3600, 2))
    del scratch_model
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("Chưa chạy scratch. Không kết luận transfer tốt hơn scratch từ run này.")
''')

curves_md = md("""## Biểu đồ quá trình train

Vẽ loss, precision, recall và mAP từ `results.csv`; đánh dấu epoch tốt nhất theo fitness. Scratch chỉ xuất hiện nếu run đối chứng đã chạy.
""")
curves_code = code('''def read_history(run_name: str) -> pd.DataFrame:
    history = pd.read_csv(RUN_DIR / run_name / "results.csv")
    history.columns = [column.strip() for column in history.columns]
    history["fitness"] = 0.1 * history["metrics/mAP50(B)"] + 0.9 * history["metrics/mAP50-95(B)"]
    return history


histories = {"transfer": read_history("train")}
if SCRATCH_PT is not None:
    histories["scratch"] = read_history("scratch")
history_rows = []
for name, history in histories.items():
    best_row = history.loc[history["fitness"].idxmax()]
    history_rows.append({"model": name, "epochs_run": len(history), "best_epoch": int(best_row["epoch"]),
                         "val_mAP50": float(best_row["metrics/mAP50(B)"]),
                         "val_mAP50_95": float(best_row["metrics/mAP50-95(B)"])})
    history.to_csv(RUN_DIR / "reports" / f"history_{name}.csv", index=False)
    columns = ["train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss",
               "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for axis, column in zip(axes.ravel(), columns):
        axis.plot(history["epoch"], history[column], linewidth=1)
        axis.plot(history["epoch"], history[column].rolling(5, min_periods=1).mean(), linewidth=2)
        axis.axvline(best_row["epoch"], color="gray", linestyle="--", linewidth=1)
        axis.set_title(column)
        axis.set_xlabel("Epoch")
    fig.suptitle(f"YOLO11n {name}: best epoch {int(best_row['epoch'])}")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "reports" / f"training_curves_{name}.png", dpi=150)
    display(fig)
    plt.close(fig)
history_table = pd.DataFrame(history_rows)
history_table.to_csv(RUN_DIR / "reports" / "training_history_summary.csv", index=False)
display(history_table)
''')

before_after_md = md("""## Cùng ảnh val: trước và sau fine-tune

So sánh dự đoán của COCO pretrained với `best.pt` trên đúng các ảnh val đã chọn ở trên. Bảng và ảnh chỉ là minh họa định tính; bằng chứng định lượng cho lợi ích của khởi tạo pretrained cần run scratch.
""")
before_after_code = code('''finetuned_model = YOLO(str(BEST_PT))
finetuned_dir = RUN_DIR / "reports" / "finetuned_val_examples"
finetuned_dir.mkdir(parents=True, exist_ok=True)
before_after_rows = []
for earlier, path in zip(pretrained_rows, pretrained_paths):
    result = finetuned_model.predict(path, imgsz=IMAGE_SIZE, conf=VIS_CONF,
                                     iou=NMS_IOU, device="cpu", batch=1, verbose=False)[0]
    predicted_names = [result.names[int(class_id)] for class_id in result.boxes.cls.cpu().tolist()]
    output_path = finetuned_dir / Path(path).name
    result.save(filename=str(output_path))
    before_after_rows.append({"image": Path(path).name,
                              "ground_truth": earlier["true_category"],
                              "pretrained_coco_predictions": earlier["coco_predictions"],
                              "finetuned_predictions": ", ".join(predicted_names)})
    del result
before_after_table = pd.DataFrame(before_after_rows)
before_after_table.to_csv(RUN_DIR / "reports" / "pretrained_vs_finetuned_val.csv", index=False)
display(before_after_table)
for path in pretrained_paths[:4]:
    name = Path(path).name
    print(name, "— COCO pretrained / FinalSFD fine-tuned")
    display(NotebookImage(filename=str(pretrained_dir / name), width=440))
    display(NotebookImage(filename=str(finetuned_dir / name), width=440))
del finetuned_model
gc.collect()
torch.cuda.empty_cache()
''')

evaluation = source(10)
evaluation = evaluation.replace('    evaluation[split_name] = summarize_metrics(metrics, split_name)', '''    evaluation[split_name] = summarize_metrics(metrics, split_name)
    if split_name == "val":
        confidence_grid = np.asarray(metrics.box.px, dtype=float)
        mean_f1 = np.asarray(metrics.box.f1_curve, dtype=float).mean(axis=0)
        chosen_index = int(np.nanargmax(mean_f1))
        DEPLOY_CONF = round(float(confidence_grid[chosen_index]), 3)
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot(confidence_grid, mean_f1)
        axis.axvline(DEPLOY_CONF, color="red", linestyle="--", label=f"val max F1: {DEPLOY_CONF:.3f}")
        axis.set(xlabel="Confidence", ylabel="Mean F1", title="Chọn confidence bằng validation")
        axis.legend()
        fig.tight_layout()
        fig.savefig(RUN_DIR / "reports" / "validation_confidence_selection.png", dpi=150)
        display(fig)
        plt.close(fig)
        print("Confidence chọn từ val:", DEPLOY_CONF)''')
evaluation = evaluation.replace('metrics_table = pd.DataFrame(metric_rows)', '''metrics_table = pd.DataFrame(metric_rows)
metrics_table["F1"] = 2 * metrics_table["precision"] * metrics_table["recall"] / (
    metrics_table["precision"] + metrics_table["recall"]
).replace(0, np.nan)''')
replace(10, evaluation)

scratch_eval_md = md("""## Bảng so sánh transfer và scratch

Nếu đã bật scratch, đánh giá hai checkpoint trên cùng val/test ở `NMS_IOU=0.5`, `conf=0.001`. Nếu chưa chạy scratch thì bảng chỉ ghi transfer và không đưa ra kết luận so sánh.
""")
scratch_eval_code = code('''comparison_rows = []
for split in ("val", "test"):
    comparison_rows.append({"model": "transfer", "split": split, "class": "all",
                            **evaluation[split]["overall"]})
    comparison_rows.extend({"model": "transfer", "split": split, "class": row["class_name"],
                            **{key: row[key] for key in ("precision", "recall", "mAP50", "mAP50_95")}}
                           for row in evaluation[split]["per_class"])
if SCRATCH_PT is not None:
    scratch_eval_model = YOLO(str(SCRATCH_PT))
    for split in ("val", "test"):
        scratch_metrics = scratch_eval_model.val(
            data=str(TRAIN_YAML), split=split, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
            conf=EVAL_CONF, iou=NMS_IOU, workers=WORKERS, device=0,
            plots=True, project=str(RUN_DIR), name=f"scratch_evaluation_{split}",
            exist_ok=True, verbose=False,
        )
        scratch_report = summarize_metrics(scratch_metrics, split)
        comparison_rows.append({"model": "scratch", "split": split, "class": "all",
                                **scratch_report["overall"]})
        comparison_rows.extend({"model": "scratch", "split": split, "class": row["class_name"],
                                **{key: row[key] for key in ("precision", "recall", "mAP50", "mAP50_95")}}
                               for row in scratch_report["per_class"])
        del scratch_metrics
        gc.collect()
        torch.cuda.empty_cache()
    del scratch_eval_model
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("Chưa chạy scratch; không suy ra ưu thế của transfer learning.")
comparison_table = pd.DataFrame(comparison_rows)
comparison_table.to_csv(RUN_DIR / "reports" / "transfer_vs_scratch.csv", index=False)
display(comparison_table)
''')

iou_md = md("""## Khảo sát ngưỡng NMS IoU

Giữ nguyên checkpoint, `imgsz=640` và `conf=0.001` để tính mAP; chỉ thay NMS IoU. Bảng val dùng để biện minh cho cấu hình chính 0.5; test chỉ kiểm tra độ nhạy sau khi đã cố định cấu hình. Một mẫu val gồm ảnh nền và ảnh có cả hai lớp được dự đoán ở confidence triển khai để quan sát báo nhầm và số box.
""")
iou_code = code('''iou_rows = []
sweep_model = YOLO(str(BEST_PT))
for split in ("val", "test"):
    for nms_iou in IOU_SWEEP:
        sweep_metrics = sweep_model.val(
            data=str(TRAIN_YAML), split=split, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
            conf=EVAL_CONF, iou=nms_iou, workers=WORKERS, device=0,
            plots=False, project=str(RUN_DIR), name=f"nms_{split}_{nms_iou:.1f}",
            exist_ok=True, verbose=False,
        )
        report = summarize_metrics(sweep_metrics, split)
        rows = [("all", report["overall"])] + [
            (row["class_name"], {key: row[key] for key in ("precision", "recall", "mAP50", "mAP50_95")})
            for row in report["per_class"]
        ]
        for class_name, values in rows:
            iou_rows.append({"split": split, "nms_iou": nms_iou, "class": class_name, **values})
        del sweep_metrics
        gc.collect()
        torch.cuda.empty_cache()
iou_table = pd.DataFrame(iou_rows)
iou_table.to_csv(RUN_DIR / "reports" / "nms_iou_sweep.csv", index=False)
display(iou_table)
for split in ("val", "test"):
    subset = iou_table[(iou_table["split"] == split) & (iou_table["class"] == "all")]
    fig, axis = plt.subplots(figsize=(7, 4))
    for metric in ("precision", "recall", "mAP50", "mAP50_95"):
        axis.plot(subset["nms_iou"], subset[metric], marker="o", label=metric)
    axis.axvline(NMS_IOU, color="gray", linestyle="--", label="chính: 0.5")
    axis.set(xlabel="NMS IoU", ylabel="Score", title=f"NMS IoU trên {split}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(RUN_DIR / "reports" / f"nms_iou_sweep_{split}.png", dpi=150)
    display(fig)
    plt.close(fig)

val_background = sorted(val_records.loc[val_records["category"] == "background", "image"].tolist())
val_both = sorted(val_records.loc[val_records["category"] == "both", "image"].tolist())
probe_rng = random.Random(SEED)
probe_groups = {
    "background": probe_rng.sample(val_background, min(80, len(val_background))),
    "both": probe_rng.sample(val_both, min(80, len(val_both))),
}
operational_rows = []
for nms_iou in IOU_SWEEP:
    for category, paths in probe_groups.items():
        detections, images_with_detection = 0, 0
        for path in paths:
            result = sweep_model.predict(path, conf=DEPLOY_CONF, iou=nms_iou,
                                         imgsz=IMAGE_SIZE, batch=1, device="cpu", verbose=False)[0]
            count = len(result.boxes)
            detections += count
            images_with_detection += int(count > 0)
            del result
        operational_rows.append({"nms_iou": nms_iou, "category": category,
                                 "confidence": DEPLOY_CONF, "images": len(paths),
                                 "images_with_detection": images_with_detection,
                                 "detections": detections})
operational_table = pd.DataFrame(operational_rows)
operational_table.to_csv(RUN_DIR / "reports" / "nms_iou_operational_val.csv", index=False)
display(operational_table)
del sweep_model
gc.collect()
torch.cuda.empty_cache()
''')

operational_md = md("""## Ma trận nhầm lẫn ở confidence triển khai

Đánh giá thêm val/test với confidence đã chọn từ val và NMS IoU chính 0.5 để tạo ma trận nhầm lẫn gần với lúc dự đoán. Không dùng mAP từ lần lọc confidence cao này làm mAP chuẩn; mAP chuẩn ở phần `conf=0.001`. Bảng mức ảnh ở phần kế tiếp phản ánh trực tiếp ngưỡng triển khai.
""")
operational_code = code('''operational_model = YOLO(str(BEST_PT))
for split in ("val", "test"):
    operational_metrics = operational_model.val(
        data=str(TRAIN_YAML), split=split, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
        conf=DEPLOY_CONF, iou=NMS_IOU, workers=WORKERS, device=0,
        plots=True, project=str(RUN_DIR), name=f"evaluation_{split}_deploy",
        exist_ok=True, verbose=False,
    )
    matrix_path = Path(operational_metrics.save_dir) / "confusion_matrix.png"
    if matrix_path.is_file():
        print(f"Ma trận {split} tại conf={DEPLOY_CONF}, NMS IoU={NMS_IOU}")
        display(NotebookImage(filename=str(matrix_path), width=700))
    del operational_metrics
    gc.collect()
    torch.cuda.empty_cache()
del operational_model
gc.collect()
torch.cuda.empty_cache()
''')

replace(11, """## Đánh giá toàn bộ test ở mức ảnh và kích thước box

Dự đoán từng ảnh trên CPU với confidence đã chọn từ val. Bảng mức ảnh trả lời ảnh có lửa/khói có được phát hiện không; bảng box theo diện tích dùng ngưỡng diện tích chuẩn hóa riêng của FinalSFD và IoU khớp box 0.5. Danh sách lỗi và ảnh minh họa giúp rà lại cả ảnh nền có nhãn gây tranh cãi.
""")
replace(12, '''test_records = image_records[image_records["split"] == "test"].reset_index(drop=True)
REPORT_DEVICE = "cpu"
REPORT_BATCH_SIZE = 1
report_model = YOLO(str(BEST_PT))


def predict_one(path: str):
    return report_model.predict(path, batch=REPORT_BATCH_SIZE, conf=DEPLOY_CONF,
                                iou=NMS_IOU, imgsz=IMAGE_SIZE, device=REPORT_DEVICE,
                                verbose=False)[0]


def pair_iou(box, others):
    if len(others) == 0:
        return np.empty(0, dtype=float)
    left_top = np.maximum(box[:2], others[:, :2])
    right_bottom = np.minimum(box[2:], others[:, 2:])
    intersection = np.prod(np.maximum(right_bottom - left_top, 0), axis=1)
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    other_areas = np.maximum(others[:, 2] - others[:, 0], 0) * np.maximum(others[:, 3] - others[:, 1], 0)
    return intersection / np.maximum(box_area + other_areas - intersection, 1e-12)


prediction_rows = []
size_counts = Counter()
for index, row in enumerate(test_records.itertuples(index=False), 1):
    result = predict_one(row.image)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    truth_fire = row.category in ("fire_only", "both")
    truth_smoke = row.category in ("smoke_only", "both")
    prediction_rows.append({
        "image": Path(row.image).name, "path": row.image, "category": row.category,
        "true_fire": truth_fire, "true_smoke": truth_smoke,
        "pred_fire": bool(np.any(classes == 1)), "pred_smoke": bool(np.any(classes == 0)),
        "detections": len(classes), "fire_detections": int(np.sum(classes == 1)),
        "smoke_detections": int(np.sum(classes == 0)),
        "max_confidence": float(scores.max()) if len(scores) else 0.0,
    })

    height, width = result.orig_shape
    label_path = LOCAL_DATASET_ROOT / "labels" / "test" / (Path(row.image).stem + ".txt")
    ground_truth = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, box_w, box_h = map(float, line.split())
        area = box_w * box_h
        size_group = "<1%" if area < 0.01 else "1–10%" if area < 0.10 else "≥10%"
        ground_truth.append({
            "class_id": int(class_id), "size_group": size_group,
            "box": np.array([(x - box_w / 2) * width, (y - box_h / 2) * height,
                             (x + box_w / 2) * width, (y + box_h / 2) * height]),
        })
    matched_gt = set()
    for pred_index in np.argsort(-scores):
        eligible = [gt_index for gt_index, gt in enumerate(ground_truth)
                    if gt["class_id"] == classes[pred_index] and gt_index not in matched_gt]
        if not eligible:
            continue
        overlaps = pair_iou(boxes[pred_index], np.stack([ground_truth[gt_index]["box"] for gt_index in eligible]))
        best_local = int(np.argmax(overlaps))
        if overlaps[best_local] >= 0.5:
            matched_gt.add(eligible[best_local])
    for gt_index, gt in enumerate(ground_truth):
        key = ("smoke" if gt["class_id"] == 0 else "fire", gt["size_group"])
        size_counts[(key, "ground_truth")] += 1
        size_counts[(key, "matched")] += int(gt_index in matched_gt)
    del result
    if index % 200 == 0 or index == len(test_records):
        print(f"Đã dự đoán test: {index}/{len(test_records)}")

pred_df = pd.DataFrame(prediction_rows)
pred_df.to_csv(RUN_DIR / "reports" / "test_image_predictions.csv", index=False)


def image_level_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    actual, predicted = actual.astype(bool), predicted.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall,
            "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


image_metric_rows = []
for class_name, actual, predicted in (
    ("fire", pred_df["true_fire"], pred_df["pred_fire"]),
    ("smoke", pred_df["true_smoke"], pred_df["pred_smoke"]),
    ("any", pred_df["true_fire"] | pred_df["true_smoke"],
     pred_df["pred_fire"] | pred_df["pred_smoke"]),
):
    image_metric_rows.append({"class": class_name, **image_level_metrics(actual, predicted)})
image_metric_table = pd.DataFrame(image_metric_rows)
image_metric_table.to_csv(RUN_DIR / "reports" / "test_image_metrics.csv", index=False)
display(image_metric_table)

size_rows = []
for class_name in ("smoke", "fire"):
    for size_group in ("<1%", "1–10%", "≥10%"):
        key = (class_name, size_group)
        total = size_counts[(key, "ground_truth")]
        matched = size_counts[(key, "matched")]
        size_rows.append({"class": class_name, "size_group": size_group,
                          "ground_truth_boxes": total, "matched_boxes": matched,
                          "recall_at_conf_iou05": matched / total if total else None})
size_recall_table = pd.DataFrame(size_rows)
size_recall_table.to_csv(RUN_DIR / "reports" / "test_box_recall_by_size.csv", index=False)
display(size_recall_table)

background_table = pred_df[pred_df["category"] == "background"].copy()
background_table[["image", "detections", "fire_detections", "smoke_detections", "max_confidence"]].to_csv(
    RUN_DIR / "reports" / "background_test_predictions.csv", index=False
)
background_false_positive_images = int((background_table["detections"] > 0).sum())
background_report = {
    "split": "test", "confidence": DEPLOY_CONF, "nms_iou": NMS_IOU,
    "device": REPORT_DEVICE, "batch": REPORT_BATCH_SIZE,
    "background_images": len(background_table),
    "images_with_detection": background_false_positive_images,
    "image_false_positive_rate": (
        background_false_positive_images / len(background_table) if len(background_table) else None
    ),
}
(RUN_DIR / "reports" / "background_test_summary.json").write_text(
    json.dumps(background_report, ensure_ascii=False, indent=2), encoding="utf-8"
)
display(pd.DataFrame([background_report]))

error_df = pred_df[(pred_df["category"].eq("background") & (pred_df["pred_fire"] | pred_df["pred_smoke"]))
                   | (pred_df["true_fire"] & ~pred_df["pred_fire"])
                   | (pred_df["true_smoke"] & ~pred_df["pred_smoke"])].copy()
error_df["background_false_alarm"] = error_df["category"].eq("background") & (
    error_df["pred_fire"] | error_df["pred_smoke"]
)
error_df["missed_fire"] = error_df["true_fire"] & ~error_df["pred_fire"]
error_df["missed_smoke"] = error_df["true_smoke"] & ~error_df["pred_smoke"]
error_df.to_csv(RUN_DIR / "reports" / "test_image_errors.csv", index=False)
display(error_df[["background_false_alarm", "missed_fire", "missed_smoke"]].sum().to_frame("images"))

sample_paths = []
sample_rng = random.Random(SEED)
for category in ("fire_only", "smoke_only", "both", "background"):
    candidates = sorted(pred_df.loc[pred_df["category"] == category, "path"].tolist())
    sample_paths.extend(sample_rng.sample(candidates, min(2, len(candidates))))
for column in ("missed_fire", "missed_smoke", "background_false_alarm"):
    candidates = sorted(error_df.loc[error_df[column], "path"].tolist())
    sample_paths.extend(sample_rng.sample(candidates, min(2, len(candidates))))
sample_paths = list(dict.fromkeys(sample_paths))
sample_dir = RUN_DIR / "reports" / "sample_predictions"
sample_dir.mkdir(parents=True, exist_ok=True)
sample_rows = []
for path in sample_paths:
    result = predict_one(path)
    output_path = sample_dir / Path(path).name
    result.save(filename=str(output_path))
    row = pred_df.loc[pred_df["path"] == path].iloc[0]
    sample_rows.append({"image": Path(path).name, "ground_truth_category": row["category"],
                        "detections": len(result.boxes), "annotated_file": output_path.name})
    del result
sample_table = pd.DataFrame(sample_rows)
sample_table.to_csv(RUN_DIR / "reports" / "sample_predictions.csv", index=False)
display(sample_table)
for output_path in sorted(sample_dir.iterdir())[:6]:
    display(NotebookImage(filename=str(output_path), width=480))
del report_model
gc.collect()
''')

report = source(14)
report = report.replace('    "dataset_images": dataset_rows,', '''    "dataset_images": dataset_rows,
    "transfer_checkpoint_reused": transfer_reused,
    "box_area_by_class": area_table.to_dict(orient="records"),
    "nms_iou_main": NMS_IOU,
    "nms_iou_sweep": list(IOU_SWEEP),
    "deploy_conf_selected_on_val": DEPLOY_CONF,
    "scratch_completed": SCRATCH_PT is not None,
    "scratch_training_seconds": scratch_training_seconds,
    "image_level_test": image_metric_rows,
    "box_recall_by_size_test": size_rows,''')
report = report.replace('"Ngưỡng 0.25 chỉ dùng để minh họa và đếm ảnh nền, chưa tối ưu cho triển khai.",',
                        '"Confidence chọn bằng max F1 trên val; cần thử thêm trên video thực tế trước triển khai.",')
report = report.replace('"Tập test chưa được chứng minh độc lập hoàn toàn theo cảnh/video.",',
                        '"Tập test chưa được chứng minh độc lập hoàn toàn theo cảnh/video.",\n'
                        '        "Bảng recall theo kích thước dùng diện tích box chuẩn hóa và confidence triển khai, không phải COCO AP_small/medium/large.",')
report = report.replace('"Ảnh mẫu và bảng dự đoán ở `reports/sample_predictions`. "',
                        '"Ảnh mẫu, ảnh sai, bảng mức ảnh và khảo sát NMS ở `reports/`. "')
replace(14, report)

packaging = source(16).replace(
    '"reports/metrics_comparison.csv"}',
    '"reports/metrics_comparison.csv", "reports/nms_iou_sweep.csv", '
    '"reports/test_image_metrics.csv", "reports/test_box_recall_by_size.csv"}'
)
replace(16, packaging)

nb["cells"] = (
    nb["cells"][:7] + [size_md, size_code, pretrain_md, pretrain_code]
    + nb["cells"][7:9] + [scratch_md, scratch_code, curves_md, curves_code,
                          before_after_md, before_after_code]
    + nb["cells"][9:11] + [scratch_eval_md, scratch_eval_code, iou_md, iou_code,
                           operational_md, operational_code]
    + nb["cells"][11:]
)

import re
for index, cell in enumerate(nb["cells"]):
    cell.setdefault("id", hashlib.sha1(f"finalsfd-{index}".encode()).hexdigest()[:8])
    if cell["cell_type"] == "markdown":
        cell.pop("outputs", None)
        cell.pop("execution_count", None)
        content = "".join(cell["source"])
        cell["source"] = re.sub(r"^## [0-9]+\. ", "## ", content, flags=re.MULTILINE).splitlines(keepends=True)
    else:
        cell["execution_count"] = None
        cell["outputs"] = []
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(NOTEBOOK)
