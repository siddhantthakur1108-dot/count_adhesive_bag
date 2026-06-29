from multiprocessing import freeze_support
from ultralytics import YOLO


def train():
    model = YOLO(r"C:\Users\siddh\Desktop\adhesive_bag\old_best_model_single bag\best (1).pt")

    model.train(
        data=r"C:\Users\siddh\Desktop\adhesive_bag\Tile_adhesive.v5-improved_overlapped.yolov8\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        freeze=10,
        lr0=0.0001,
        lrf=0.00001,
        optimizer="auto",
        patience=8,
        cos_lr=True,
        workers=4,
        device="0",
        project="improved_overlapped",
        name="train_head_improved_overlapped",
        exist_ok=True,
    )


if __name__ == "__main__":
    freeze_support()
    train()