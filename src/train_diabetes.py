"""Run:  python -m src.train_diabetes"""
from src.training import run_from_disk, _f, dl

if __name__ == "__main__":
    try:
        r = run_from_disk("diabetes"); print("Selected:", r["meta"]["selected_model"], "test F1", _f(r["final_test"]["f1"]))
    except dl.DatasetError as e:
        print("NOT TRAINED:", e)
