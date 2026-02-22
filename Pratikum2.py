import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ==============================
# FUNGSI KUANTISASI
# ==============================

def uniform_quantization(img, levels=16):
    step = 256 // levels
    q = (img // step) * step + step // 2
    return np.clip(q, 0, 255).astype(np.uint8)

def nonuniform_quantization(img, k=16):
    Z = img.reshape((-1,1)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 3,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape(img.shape)

# ==============================
# PARAMETER TEKNIS
# ==============================

def memory_size(img):
    return img.size * img.itemsize

def compression_ratio(before, after):
    return memory_size(before) / memory_size(after)

# ==============================
# FILE CITRA (3 kondisi cahaya)
# ==============================

files = ["image3.jpeg", "image4.jpeg", "image5.jpeg"]

print("Folder kerja:", os.getcwd())

for fname in files:
    print("\n====================")
    print("Analisis:", fname)

    if not os.path.exists(fname):
        print("File tidak ada:", fname)
        continue

    img_bgr = cv2.imread(fname)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ==========================
    # REPRESENTASI MATRIX & VEKTOR
    # ==========================
    print("\n--- Representasi Citra ---")

    matrix_rgb = img_rgb[:5, :5, :]
    print("\nMatriks RGB 5x5:")
    print(matrix_rgb)

    vector_rgb = img_rgb.flatten()
    print("Ukuran vektor:", vector_rgb.shape)
    print("10 nilai pertama:", vector_rgb[:10])

    # ==========================
    # KONVERSI WARNA
    # ==========================
    start = time.time()

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    conv_time = time.time() - start

    hsv_v = hsv[:,:,2]
    lab_l = lab[:,:,0]

    # ==========================
    # KUANTISASI
    # ==========================
    gray_u = uniform_quantization(gray,16)
    gray_n = nonuniform_quantization(gray,16)

    hsv_u = uniform_quantization(hsv_v,16)
    hsv_n = nonuniform_quantization(hsv_v,16)

    lab_u = uniform_quantization(lab_l,16)
    lab_n = nonuniform_quantization(lab_l,16)

    # ==========================
    # SEGMENTASI (Otsu)
    # ==========================
    _, seg_gray = cv2.threshold(gray_u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, seg_hsv  = cv2.threshold(hsv_u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, seg_lab  = cv2.threshold(lab_u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    area_gray = np.sum(seg_gray == 255)
    area_hsv  = np.sum(seg_hsv == 255)
    area_lab  = np.sum(seg_lab == 255)

    print("\n--- Analisis Segmentasi ---")
    print("Area objek Gray:", area_gray)
    print("Area objek HSV:", area_hsv)
    print("Area objek LAB:", area_lab)

    # ==========================
    # PARAMETER TEKNIS
    # ==========================

    # Gray
    mem_gray_before = memory_size(gray)
    mem_gray_u = memory_size(gray_u)
    mem_gray_n = memory_size(gray_n)

    cr_gray_u = compression_ratio(gray, gray_u)
    cr_gray_n = compression_ratio(gray, gray_n)

    # HSV
    mem_hsv_before = memory_size(hsv_v)
    mem_hsv_u = memory_size(hsv_u)
    mem_hsv_n = memory_size(hsv_n)

    cr_hsv_u = compression_ratio(hsv_v, hsv_u)
    cr_hsv_n = compression_ratio(hsv_v, hsv_n)

    # LAB
    mem_lab_before = memory_size(lab_l)
    mem_lab_u = memory_size(lab_u)
    mem_lab_n = memory_size(lab_n)

    cr_lab_u = compression_ratio(lab_l, lab_u)
    cr_lab_n = compression_ratio(lab_l, lab_n)

    print("\n--- Parameter Teknis ---")
    print("Waktu konversi warna:", round(conv_time,6), "detik")

    print("\nGray:")
    print("Memori sebelum:", mem_gray_before)
    print("Uniform:", mem_gray_u)
    print("Non-uniform:", mem_gray_n)
    print("CR Uniform:", round(cr_gray_u,2))
    print("CR Non-uniform:", round(cr_gray_n,2))

    print("\nHSV:")
    print("Memori sebelum:", mem_hsv_before)
    print("Uniform:", mem_hsv_u)
    print("Non-uniform:", mem_hsv_n)
    print("CR Uniform:", round(cr_hsv_u,2))
    print("CR Non-uniform:", round(cr_hsv_n,2))

    print("\nLAB:")
    print("Memori sebelum:", mem_lab_before)
    print("Uniform:", mem_lab_u)
    print("Non-uniform:", mem_lab_n)
    print("CR Uniform:", round(cr_lab_u,2))
    print("CR Non-uniform:", round(cr_lab_n,2))

    # ==========================
    # VISUALISASI
    # ==========================
    plt.figure(figsize=(14,10))

    # Gray
    plt.subplot(4,4,1); plt.imshow(img_rgb); plt.title("RGB"); plt.axis("off")
    plt.subplot(4,4,2); plt.imshow(gray,cmap="gray"); plt.title("Gray"); plt.axis("off")
    plt.subplot(4,4,3); plt.imshow(gray_u,cmap="gray"); plt.title("Gray U"); plt.axis("off")
    plt.subplot(4,4,4); plt.imshow(gray_n,cmap="gray"); plt.title("Gray N"); plt.axis("off")

    # HSV
    plt.subplot(4,4,6); plt.imshow(hsv_v,cmap="gray"); plt.title("HSV-V"); plt.axis("off")
    plt.subplot(4,4,7); plt.imshow(hsv_u,cmap="gray"); plt.title("HSV U"); plt.axis("off")
    plt.subplot(4,4,8); plt.imshow(hsv_n,cmap="gray"); plt.title("HSV N"); plt.axis("off")

    # LAB
    plt.subplot(4,4,10); plt.imshow(lab_l,cmap="gray"); plt.title("LAB-L"); plt.axis("off")
    plt.subplot(4,4,11); plt.imshow(lab_u,cmap="gray"); plt.title("LAB U"); plt.axis("off")
    plt.subplot(4,4,12); plt.imshow(lab_n,cmap="gray"); plt.title("LAB N"); plt.axis("off")

    # Segmentasi
    plt.subplot(4,4,14); plt.imshow(seg_gray,cmap="gray"); plt.title("Seg Gray"); plt.axis("off")
    plt.subplot(4,4,15); plt.imshow(seg_hsv,cmap="gray"); plt.title("Seg HSV"); plt.axis("off")
    plt.subplot(4,4,16); plt.imshow(seg_lab,cmap="gray"); plt.title("Seg LAB"); plt.axis("off")

    plt.suptitle(f"Hasil Analisis - {fname}")
    plt.tight_layout()
    plt.show()

    # ==========================
    # HISTOGRAM
    # ==========================

    # Gray
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.hist(gray.ravel(),256,[0,256])
    plt.title("Gray Asli")

    plt.subplot(1,3,2)
    plt.hist(gray_u.ravel(),256,[0,256])
    plt.title("Gray Uniform")

    plt.subplot(1,3,3)
    plt.hist(gray_n.ravel(),256,[0,256])
    plt.title("Gray Non-uniform")
    plt.show()

    # HSV
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.hist(hsv_v.ravel(),256,[0,256])
    plt.title("HSV Asli")

    plt.subplot(1,3,2)
    plt.hist(hsv_u.ravel(),256,[0,256])
    plt.title("HSV Uniform")

    plt.subplot(1,3,3)
    plt.hist(hsv_n.ravel(),256,[0,256])
    plt.title("HSV Non-uniform")
    plt.show()

    # LAB
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.hist(lab_l.ravel(),256,[0,256])
    plt.title("LAB Asli")

    plt.subplot(1,3,2)
    plt.hist(lab_u.ravel(),256,[0,256])
    plt.title("LAB Uniform")

    plt.subplot(1,3,3)
    plt.hist(lab_n.ravel(),256,[0,256])
    plt.title("LAB Non-uniform")
    plt.show()

print("\nSELESAI SEMUA GAMBAR DIPROSES")