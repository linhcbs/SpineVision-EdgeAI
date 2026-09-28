# Báo Cáo Huấn Luyện & Đánh Giá Lightweight Posture Classifier Head

> **Thời gian huấn luyện:** 2026-09-28 23:14:43
> **Tập dữ liệu huấn luyện:** Kết hợp cả 2 bộ dữ liệu: Sitting Posture v4 COCO + Sitting Posture v2 Multiclass
> **Tổng số mẫu:** 1887 train | 242 valid | 121 test
> **Mô hình tốt nhất được chọn:** `LightweightMLP`

---

## 1. So Sánh Các Kiến Trúc Classifier Nhẹ

| Mô hình | Split | Accuracy | F1-Score | Recall (Bad) | Precision (Bad) | COCO F1 | Multiclass F1 |
|---------|-------|----------|----------|--------------|-----------------|---------|---------------|
| **LogisticRegression** | Train | 81.24% | **81.58%** | 78.71% | 84.67% | 87.6% | 78.2% |
| **LogisticRegression** | Valid | 86.78% | **86.44%** | 80.95% | 92.73% | 78.1% | 89.5% |
| **LogisticRegression** | Test | 85.95% | **87.02%** | 83.82% | 90.48% | 83.9% | 88.0% |
| **LightweightMLP** | Train | 85.21% | **85.37%** | 81.73% | 89.35% | 89.5% | 83.0% |
| **LightweightMLP** | Valid | 90.08% | **90.24%** | 88.10% | 92.50% | 81.8% | 93.3% |
| **LightweightMLP** | Test | 90.91% | **91.85%** | 91.18% | 92.54% | 90.3% | 92.3% |
| **RandomForest** | Train | 91.15% | **91.41%** | 89.26% | 93.68% | 94.1% | 89.9% |
| **RandomForest** | Valid | 90.50% | **90.76%** | 89.68% | 91.87% | 81.8% | 94.0% |
| **RandomForest** | Test | 91.74% | **92.75%** | 94.12% | 91.43% | 90.3% | 93.5% |

---

## 2. Thông Số Kiến Trúc Đầu Phân Loại Được Chọn

- **Loại mô hình:** `LightweightMLP`
- **Kích thước vector đặc trưng:** 12 chiều (CVA, Trunk slump, Shoulder tilt, Spine lateral offset, Eye distance deficit, Ear-Shoulder ratio, Nose-Shoulder ratio, Torso lean angle)
- **Thời gian thực thi suy luận (Inference Latency):** < **0.02 ms** trên CPU (thuần NumPy matrix operations)
- **Vị trí lưu trọng số:** `/media/linhcbs/DATA/NCKH/NCKH HÈ/code/models/weights/posture_classifier_weights.json` và `/media/linhcbs/DATA/NCKH/NCKH HÈ/code/configs/posture_classifier_weights.json`
