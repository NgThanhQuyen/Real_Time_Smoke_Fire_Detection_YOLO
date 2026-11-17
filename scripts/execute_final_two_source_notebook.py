"""Chạy cell phân tích và lưu output notebook khi kernel Jupyter không khởi động."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import base64
import os

import matplotlib
matplotlib.use("Agg")
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/data_preparation/16_build_final_two_source_dataset.ipynb"
FIGURES = ROOT / "reports/final_two_source_dataset/figures"


def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)


def main() -> None:
    os.chdir(ROOT)
    notebook = nbf.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__main__", "display": display}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        before = set(FIGURES.glob("*.png")) if FIGURES.exists() else set()
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            exec(compile(cell.source, f"notebook_cell_{count}", "exec"), namespace)
        cell.execution_count = count
        cell.outputs = []
        if output.getvalue():
            cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=output.getvalue()))
        for path in sorted(set(FIGURES.glob("*.png")) - before):
            cell.outputs.append(nbf.v4.new_output(
                "display_data", data={"image/png": base64.b64encode(path.read_bytes()).decode()},
                metadata={}
            ))
        nbf.write(notebook, NOTEBOOK)
        print(f"Executed cell {count}", flush=True)
    nbf.validate(notebook)
    print(f"Notebook complete: {NOTEBOOK}", flush=True)


if __name__ == "__main__":
    main()
