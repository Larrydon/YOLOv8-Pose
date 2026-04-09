# YOLOv8-Pose 找車牌4點
標註檔一定要貼合車牌4頂點，效果才會越好，可以利用 PaddleOCR 訓練用的時候，透過 PPOCRLabel 標註完成的 Label.txt(這時候標註就要貼合了)
再透過 convert_ppocr_to_yolo_pose.py 自動轉成YOLO格式
(src\convert_ppocr_to_yolo_pose.py)
接著 refine_label.py 利用OpenCV自動找4頂點，自動修正更好的4頂點座標
(src\refine_label.py)


## 環境 ##
Python 3.10.19
<br>
<br>

### CPU版本Torch ###
> pip install ultralytics opencv-python

直接 pip install ultralytics 預設通常會安裝 CPU 版本 的 PyTorch。
<br>
<br>

### GPU版本Torch ###
如果你要跑 YOLOv8 訓練，建議先去 PyTorch 官網<br>
 根據你的 CUDA 版本複製指令安裝 torch，然後再裝 ultralytics。<br>
 NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4<br>
 驅動程式支援 CUDA 12.4<br>
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

版本衝突： 如果你之前已經裝過 CPU 版，建議先
> pip uninstall torch torchvision<br>
> 
刪乾淨後，再執行上面的 CUDA 12.4 安裝指令。


## 開始訓練 ##
遇到錯誤:
>
	You said
	The following required CPU features were not detected:

		avx2, fma, bmi1, bmi2, lzcnt, movbe

	Continuing to use this version of Polars on this processor will likely result in a crash.

	Install `polars[rtcompat]` instead of `polars` to run Polars with better compatibility.



	Hint: If you are on an Apple ARM machine (e.g. M1) this is likely due to running Python under Rosetta.

	It is recommended to install a native version of Python that does not run under Rosetta x86-64 emulation.



	If you believe this warning to be a false positive, you can set the `POLARS_SKIP_CPU_CHECK` environment variable to bypass this check.

這個錯誤訊息是因為你安裝了 Polars（一個高效能資料處理庫），<br>
而你的 CPU 比較舊，不支援 Polars 預設要求的現代指令集（如 AVX2、FMA 等）。

安裝相容版 Polars<br>
按照錯誤訊息的提示，安裝專為舊款 CPU 或相容模式設計的版本：<br>

> 	pip uninstall polars<br>
	pip install "polars[rtcompat]"<br>

這會自動幫你處理剛才提到的 polars-runtime-32 相關依賴，讓 Polars 可以在沒有 AVX2 指令集的 CPU 上跑。




