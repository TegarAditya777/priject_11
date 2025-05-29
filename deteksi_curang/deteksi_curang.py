import cv2
import time
import os
from datetime import datetime

def log_event(event):
    with open("log_deteksi_curang.txt", "a") as log_file:
        log_file.write(f"{datetime.now()} - {event}\n")

def save_screenshot(frame, event):
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
    filename = f"screenshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event}.png"
    cv2.imwrite(filename, frame)

def main():
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam tidak dapat diakses.")
        return

    wajah_terakhir_terdeteksi = time.time()
    wajah_tidak_terdeteksi_durasi = 3  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame dari webcam.")
            break

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        wajah = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        fokus_terdeteksi = True

        
        for (x, y, w, h) in wajah:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            
            mata = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))

            if len(mata) == 0:
                fokus_terdeteksi = False
            else:
                
                for (ex, ey, ew, eh) in mata:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        
        jumlah_wajah = len(wajah)
        if jumlah_wajah > 1:
            cv2.putText(frame, "Peringatan: Lebih dari satu wajah terdeteksi!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            log_event("Lebih dari satu wajah terdeteksi")
            save_screenshot(frame, "lebih_dari_satu_wajah")
        elif jumlah_wajah == 0:
            
            if time.time() - wajah_terakhir_terdeteksi > wajah_tidak_terdeteksi_durasi:
                cv2.putText(frame, "Peringatan: Wajah tidak terlihat selama > 3 detik!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                log_event("Wajah tidak terlihat selama > 3 detik")
                save_screenshot(frame, "wajah_tidak_terlihat")
        else:
            
            wajah_terakhir_terdeteksi = time.time()
            if not fokus_terdeteksi:
                cv2.putText(frame, "Peringatan: Wajah tidak fokus ke layar!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                log_event("Wajah tidak fokus ke layar")
                save_screenshot(frame, "wajah_tidak_fokus")

            
            frame_center_x = frame.shape[1] // 2
            for (x, y, w, h) in wajah:
                wajah_center_x = x + w // 2
                
                if abs(wajah_center_x - frame_center_x) > w // 2:
                    cv2.putText(frame, "Peringatan: Kepala menoleh ke samping!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    log_event("Kepala menoleh ke samping")
                    save_screenshot(frame, "kepala_menoleh")
                    break

        
        cv2.imshow('Deteksi Curang - Tekan q untuk keluar', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()