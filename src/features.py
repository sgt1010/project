"""
features.py -- turn an image into a numeric feature vector.

WHY hand-crafted features?  KNN, SVM and Decision Trees work on tabular
vectors, not raw pixels. We describe each road image with a compact vector
that captures what makes a pothole / crack different from smooth asphalt:

  GROUP             WHAT IT CAPTURES                       #FEATURES
  ----------------  -------------------------------------  ---------
  Intensity stats   potholes are darker / higher variance       6
  Histogram         full brightness distribution               16
  Edge stats        cracks & pothole rims = many edges          4
  Dark-blob stats   size/shape of the largest dark region       5
  LBP texture       micro-texture (rough vs smooth asphalt)    10
  HOG (coarse)      local gradient orientation structure       ~72

Pipeline:  image -> grayscale -> resize -> CLAHE -> features
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from config import IMG_SIZE


def preprocess(img):
    """Gray + resize + CLAHE (contrast equalisation) so lighting differences
    between photos matter less."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_features(img):
    g = preprocess(img)
    f = []

    # 1) Intensity statistics (6)
    f += [g.mean(), g.std(), np.median(g), g.min(), g.max(), np.percentile(g, 10)]

    # 2) Normalised 16-bin histogram (16)
    hist = cv2.calcHist([g], [0], None, [16], [0, 256]).ravel()
    f += list(hist / hist.sum())

    # 3) Edge statistics (4): Canny density + Sobel gradient magnitude stats
    edges = cv2.Canny(g, 60, 150)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    f += [edges.mean() / 255, mag.mean(), mag.std(), np.percentile(mag, 95)]

    # 4) Largest dark blob (5): Otsu threshold -> contours -> biggest dark area
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True) + 1e-6
        circ = 4 * np.pi * a / (p ** 2)              # 1.0 = perfect circle
        x, y, w, h = cv2.boundingRect(c)
        f += [a / float(g.size), circ, w / (h + 1e-6), len(cnts), dark.mean() / 255]
    else:
        f += [0, 0, 0, 0, 0]

    # 5) Local Binary Pattern texture histogram (10)
    lbp = local_binary_pattern(g, P=8, R=1, method="uniform")
    lh, _ = np.histogram(lbp, bins=10, range=(0, 10))
    f += list(lh / lh.sum())

    # 6) HOG on a 64x64 version with coarse cells -> keeps vector small
    small = cv2.resize(g, (64, 64))
    f += list(hog(small, orientations=8, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), feature_vector=True))

    return np.asarray(f, dtype=np.float32)
