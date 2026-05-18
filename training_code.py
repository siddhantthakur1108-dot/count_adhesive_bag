
#  Potato Bag Detection — 3-Phase Training

from ultralytics import YOLO

# ── CONFIG for YOUR dataset ──────────────────────────────────
CONFIG = {
    "data_yaml"        : r"C:\Users\siddh\Desktop\adhesive_bag\Tile_adhesive.v2i.yolov8\data.yaml",
    "pretrained_model" : "yolov8s.pt",   # ← small model for tiny dataset
                                          #   yolov8m if you have RTX 3060+
    "imgsz"            : 640,            # 640 is fine for conveyor belt
                                          # use 1280 only if bags are very small
    "project_dir"      : "runs/adhesive_bag",
    "experiment_name"  : "v1",

    # ── Phase 1: HEAD only ─────────────────────────────────
    # Short — single class head converges fast
    "phase1": {
        "epochs"        : 30,
        "freeze"        : 10,      # full backbone frozen
        "lr0"           : 0.001,   # lower than usual — small dataset
        "lrf"           : 0.0001,
        "batch"         : 8,
        "optimizer"     : "AdamW",
        "warmup_epochs" : 3,
        "patience"      : 8,       # stop early if no improvement for 8 epochs
    },

    # ── Phase 2: NECK + HEAD ───────────────────────────────
    # Let neck adapt to conveyor belt object scales
    "phase2": {
        "epochs"        : 10,
        "freeze"        : 3,       # only freeze first 3 backbone layers
        "lr0"           : 0.0005,
        "lrf"           : 0.00005,
        "batch"         : 8,
        "optimizer"     : "AdamW",
        "warmup_epochs" : 2,
        "patience"      : 5,
    },

    # ── Phase 3: Full unfreeze ─────────────────────────────
    # Short + very low LR — overfit risk is high with 550 images
    "phase3": {
        "epochs"        : 20,
        "freeze"        : 0,
        "lr0"           : 0.00005,  # very conservative
        "lrf"           : 0.000005,
        "batch"         : 16,
        "optimizer"     : "AdamW",
        "warmup_epochs" : 1,
        "patience"      : 6,        # stop quickly if overfitting starts
    },

    "device"     : "0",
    "workers"    : 4,
    "amp"        : True,
    "multi_scale": False,  # OFF for small dataset — adds training noise
}


def train_adhesiv_bag():

    # ── PHASE 1 ────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Training HEAD only (30 epochs)")
    print("Backbone + Neck: FROZEN | LR = 0.005")
    print("="*60)

    model = YOLO(r"runs\detect\runs\adhesive_bag\phase1\weights\last.pt")

    p2 = CONFIG["phase2"]
    model.train(
        data          = CONFIG["data_yaml"],
        epochs        = p2["epochs"],          # 30 epochs
        imgsz         = CONFIG["imgsz"],
        batch         = p2["batch"],
        freeze        = p2["freeze"],          # freeze layers 0–9
        lr0           = p2["lr0"],
        lrf           = p2["lrf"],
        optimizer     = p2["optimizer"],
        warmup_epochs = p2["warmup_epochs"],
        cos_lr        = True,
        patience      = p2["patience"],        # early stop at 8 epochs no improve
        weight_decay  = 0.0005,

        # ── Augmentation — aggressive for small dataset ──
        mosaic        = 1.0,
        copy_paste    = 0.3,   # paste bags onto empty belt images ← key
        mixup         = 0.15,
        degrees       = 10.0,
        translate     = 0.15,
        scale         = 0.8,
        fliplr        = 0.5,
        flipud        = 0.0,
        hsv_h         = 0.02,
        hsv_s         = 0.8,
        hsv_v         = 0.5,
        perspective   = 0.001,

        device        = CONFIG["device"],
        workers       = CONFIG["workers"],
        amp           = CONFIG["amp"],
        multi_scale   = CONFIG["multi_scale"],
        plots         = True,
        save_period   = 5,
        project       = CONFIG["project_dir"],
        name          = "phase1",
        exist_ok      = True,
    )
    print("Phase 1 complete. Best weights → runs/potato_bag/phase1/weights/best.pt")



if __name__ == "__main__":
    train_adhesiv_bag()