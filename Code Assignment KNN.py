import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset dari file CSV
color_data = pd.read_csv('dataset/color-detection-data-set/colors.csv')
X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# Normalisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediksi data test dan menghitung akurasi
y_pred = knn.predict(X_test)
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {overall_accuracy * 100:.2f}%")

cap = cv2.VideoCapture(0)

detected_colors_1 = []
detected_true_labels_1 = []
detected_colors_2 = []
detected_true_labels_2 = []

def find_nearest_color(avg_color):
    distances = np.linalg.norm(X - avg_color, axis=1)
    nearest_idx = np.argmin(distances)
    return y[nearest_idx]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Ambil ukuran frame
    height, width, _ = frame.shape
    
    # Tentukan koordinat bounding box di tengah layar
    box_size = 100
    x_start_1, y_start_1 = (width // 4 - box_size // 2, height // 2 - box_size // 2)
    x_end_1, y_end_1 = (x_start_1 + box_size, y_start_1 + box_size)
    
    x_start_2, y_start_2 = (3 * width // 4 - box_size // 2, height // 2 - box_size // 2)
    x_end_2, y_end_2 = (x_start_2 + box_size, y_start_2 + box_size)
    
    # Ambil warna dari area tengah pertama
    roi_1 = frame[y_start_1:y_end_1, x_start_1:x_end_1]
    avg_color_1 = np.mean(roi_1, axis=(0, 1)).astype(int)  # Ambil warna rata-rata
    
    # Ambil warna dari area tengah kedua
    roi_2 = frame[y_start_2:y_end_2, x_start_2:x_end_2]
    avg_color_2 = np.mean(roi_2, axis=(0, 1)).astype(int)  # Ambil warna rata-rata
    
    # Normalisasi warna
    avg_color_scaled_1 = scaler.transform([avg_color_1])
    avg_color_scaled_2 = scaler.transform([avg_color_2])
    
    # Prediksi warna
    color_pred_1 = knn.predict(avg_color_scaled_1)[0]
    true_color_1 = find_nearest_color(avg_color_1)
    
    color_pred_2 = knn.predict(avg_color_scaled_2)[0]
    true_color_2 = find_nearest_color(avg_color_2)
    
    # Simpan prediksi dan warna asli untuk perhitungan akurasi
    detected_colors_1.append(color_pred_1)
    detected_true_labels_1.append(true_color_1)
    
    detected_colors_2.append(color_pred_2)
    detected_true_labels_2.append(true_color_2)
    
    # Hitung akurasi deteksi warna secara real-time
    if len(detected_colors_1) > 50:
        detected_colors_1.pop(0)
        detected_true_labels_1.pop(0)
    
    if len(detected_colors_2) > 50:
        detected_colors_2.pop(0)
        detected_true_labels_2.pop(0)
    
    color_accuracy_1 = accuracy_score(detected_true_labels_1, detected_colors_1) * 100 if detected_colors_1 else 0.0
    color_accuracy_2 = accuracy_score(detected_true_labels_2, detected_colors_2) * 100 if detected_colors_2 else 0.0
    
    # Cetak akurasi ke terminal
    print(f"Deteksi Warna 1: {color_pred_1} | Warna Asli 1: {true_color_1} | Akurasi Deteksi Warna 1: {color_accuracy_1:.2f}%")
    print(f"Deteksi Warna 2: {color_pred_2} | Warna Asli 2: {true_color_2} | Akurasi Deteksi Warna 2: {color_accuracy_2:.2f}%")
    
    # Gambar bounding box
    cv2.rectangle(frame, (x_start_1, y_start_1), (x_end_1, y_end_1), (0, 255, 0), 2)
    cv2.rectangle(frame, (x_start_2, y_start_2), (x_end_2, y_end_2), (0, 255, 0), 2)
    
    # Tambahkan label warna pada bounding box
    cv2.putText(frame, f'{color_pred_1}', (x_start_1, y_start_1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'{color_pred_2}', (x_start_2, y_start_2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Tampilkan informasi warna dan akurasi
    cv2.putText(frame, f'Color 1: {color_pred_1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f'True Color 1: {true_color_1}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f'Accuracy 1: {color_accuracy_1:.2f}%', (x_start_1 - 50, y_end_1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.putText(frame, f'Color 2: {color_pred_2}', (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f'True Color 2: {true_color_2}', (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f'Accuracy 2: {color_accuracy_2:.2f}%', (x_start_2 - 50, y_end_2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()