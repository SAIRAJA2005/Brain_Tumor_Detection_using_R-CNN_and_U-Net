# Brain_Tumor_Detection_using_R-CNN_and_U-Net



## U NET

# 🧠 Brain Tumor Detection & Segmentation  
## Full Pipeline – From Zero to Trained Model

This guide walks you step-by-step through:

1️⃣ Environment Setup  
2️⃣ Preprocessing BraTS Dataset  
3️⃣ Training U-Net  
4️⃣ Saving Weights  
5️⃣ (Optional) Prediction  

---

# ✅ STEP 1 — Environment Setup (Only Once)

## 🔹 1. Open Command Prompt

Navigate to your project root:

E:\Final_Year_Project\Brain-Tumor-Detection-and-Segmentation-using-Deep-Learning

---

## 🔹 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

You should now see:

```
(venv)
```

---

## 🔹 3. Install Required Libraries

```bash
pip install tensorflow numpy matplotlib scikit-image simpleitk opencv-python tqdm
```

Wait until installation finishes.

---

# ✅ STEP 2 — Dataset Structure Check

Your dataset must look like this:

```
BraTS_small/
 ├── HGG/
 ├── LGG/
 ├── processed/      (can be empty)
 └── weights/        (can be empty)
```

Inside `LGG/` or `HGG/`:

```
BraTS19_2013_x_x/
   ├── *_flair.nii
   ├── *_seg.nii
   ├── *_t1.nii
   ├── *_t2.nii
```

If this structure is correct → proceed.

---

# ✅ STEP 3 — Run Preprocessing

Go into the U-Net folder:

```bash
cd U-Net
```

---

## 🔧 Edit `preprocess.py`

Make sure this path is correct:

```python
src_root = r"E:\Final_Year_Project\BraTS_small\LGG"
```

---

## ▶ Run Preprocessing

```bash
python preprocess.py
```

### ✔ Expected Output

```
Saved 800 slices to y_120.npy
Saved 800 slices to x_120.npy
```

You should now see in `U-Net/`:

```
x_120.npy
y_120.npy
```

✅ Preprocessing is DONE.

---

# ✅ STEP 4 — Train the U-Net

Open `train.py`.

---

## 🔴 IMPORTANT — Modify Path

Change this line:

```python
data_root = r"E:/Final_Year_Project/BraTS_small/processed"
```

To:

```python
data_root = r"."
```

This tells the script to load data from the current folder.

---

## ▶ Run Training

```bash
python train.py
```

---

### ✔ Expected Console Output

```
Loading dataset...
Original X shape: (800, 1, 120, 120)
Converted X shape: (800, 120, 120, 1)
Loading the model...
Starting training...
Epoch 1/30
```

Training will run for **30 epochs**.

---

## 💾 After Training Completes

Model weights will be saved to:

```
E:/Final_Year_Project/BraTS_small/weights/dice_weights_120_30.h5
```

🎉 Congratulations — your model is now trained!

---

# ✅ STEP 5 — (Optional) Test / Predict

After training:

1. Load saved weights  
2. Run prediction  
3. Visualize tumor mask  

Run:

```bash
python predict.py
```

This will generate predicted tumor segmentation masks.

---

# 🧩 Full Command Summary (Quick Version)

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Go to U-Net folder
cd U-Net

# 3. Preprocess data
python preprocess.py

# 4. Train model
python train.py
```

---

# 🏁 End of Pipeline

You now have:

✔ Preprocessed MRI slices  
✔ Trained U-Net model  
✔ Saved model weights  
✔ Optional prediction pipeline  
