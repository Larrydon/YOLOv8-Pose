from ultralytics import YOLO
import os


# 強制排除 Polars 可能產生的 CPU 警告（保險起見）
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"


# 載入 Small 版本模型
model = YOLO("yolov8s-pose.pt")
# model = YOLO('runs/pose/license_plate_project/v8s_pose_plate/weights/best.pt') =>train3

# 開始訓練
model.train(
    data="configs/license_plate_pose.yaml",
    epochs=200,  # 建議稍微增加 epoch 讓 s 版本充分收斂
    imgsz=640,  # 標準尺寸，若車牌很小可設為 1280
    batch=8,  # 根據顯存調整，8 或 16 皆可
    workers=0,  # <--- 先改為 0 試試看，這會關閉多執行緒讀取
    save=True,  # 自動儲存最佳權重
    device=0,  # 使用第一張顯卡
    # device="cpu",
    # project="license_plate_project",
    # name="v8s_pose_plate",
    degrees=20,  # 旋轉角度，增加對傾斜車牌的魯棒性
    perspective=0.001,  # 透視變換增強
    shear=10.0,  # 剪切變換
    mosaic=1.0,  # 確保開啟 Mosaic，對偵測小目標很有幫助
    flipud=0.0,  # (上下翻轉): 務必設為 0.0。車牌不會倒過來，開了反而會干擾模型。
)

# # 進行微調 (Fine-tuning)
# model.train(
#     data='configs/license_plate_pose.yaml',
#     epochs=50,         # 再練 50 輪就好
#     imgsz=640,
#     batch=8,    # 根據顯存調整，8 或 16 皆可
#     workers=0,  # <--- 先改為 0 試試看，這會關閉多執行緒讀取
#     save=True,  # 自動儲存最佳權重
#     device=0,   # 使用第一張顯卡
#     #device="cpu",
#     degrees=15,        # 加入更強的旋轉增強
#     perspective=0.0005,
#     shear=10.0,
#     mosaic=1.0,
# )
