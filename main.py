import os
import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import face_recognition

# code here
is_capturing = False
cap = None
image_count = 0
max_images = 10 
current_frame = None
encoded_faces = []
student_names = []

def start_capture():
    global is_capturing, cap, folder, progress_var
    student_name = entry_name.get()
    if not student_name:
        messagebox.showwarning("Peringatan", "Masukkan nama mahasiswa terlebih dahulu!")
        return
 
    folder = f"dataset/{student_name}"
    os.makedirs(folder, exist_ok=True)
 
    cap = cv2.VideoCapture(0)
 
    if not cap.isOpened():
        messagebox.showerror("Error", "Kamera tidak dapat dibuka!")
        return
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
 
    is_capturing = True
    messagebox.showinfo("Info", "Mulai merekam gambar. Total 10 gambar akan diambil.")
 
    progress_var.set(0)
    progress_bar['maximum'] = max_images  
    
    capture_images()

def capture_images():
    global is_capturing, current_frame, image_count, max_images, cap, progress_var
    if not is_capturing:
        return

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Gagal membaca frame dari kamera!")
        return

    current_frame = frame
    img_name = f"{folder}/{entry_name.get()}_{image_count + 1}.jpg"
    cv2.imwrite(img_name, frame)
    image_count += 1

    progress_var.set(image_count)
    percentage = (image_count / max_images) * 100
    percentage_label.config(text=f"{percentage:.0f}%")

    if image_count < max_images:
        root.after(1000, capture_images)
    else:
        cap.release()
        encode_faces()
        messagebox.showinfo("Selesai", f"Pengambilan gambar selesai. {max_images} gambar telah diambil dan encoding wajah selesai!")
        is_capturing = False

def encode_faces():
    global encoded_faces, student_names

    if not os.path.exists('dataset'):
        messagebox.showwarning("Peringatan", "Dataset tidak ditemukan! Silakan ambil gambar terlebih dahulu.")
        return

    for student_folder in os.listdir('dataset'):
        folder_path = f'dataset/{student_folder}'
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                img_path = f'{folder_path}/{image_file}'
                img = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(img)

                if face_encodings:
                    encoded_faces.append(face_encodings[0])
                    student_names.append(student_folder)

root = tk.Tk()
root.title("Sistem Absensi Pendeteksi Wajah")
root.geometry("800x600")

bg_color = "#2f2f2f"  
button_bg = "#ffffff"  
button_fg = "#000000"  
input_bg = "#ffffff" 
input_fg = "#000000"

root.configure(bg=bg_color)
 
padding_frame = tk.Frame(root, padx=20, pady=20, bg=bg_color)
padding_frame.pack(expand=True)

label_name = tk.Label(padding_frame, text="Masukkan Nama Mahasiswa:", bg=bg_color, font=("Arial", 14, "bold"), fg="#ffffff")
label_name.pack(pady=10)
 
entry_name = tk.Entry(padding_frame, font=("Arial", 14), justify='center', bg=input_bg, fg=input_fg, width=40)
entry_name.pack(pady=5, padx=10, fill=tk.X)
 
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(padding_frame, variable=progress_var, maximum=max_images, length=650, style="TProgressbar")
progress_bar.pack(pady=5)

percentage_label = tk.Label(padding_frame, text="0%", bg=bg_color, font=("Arial", 14, "bold"), fg="#ffffff")
percentage_label.pack(pady=(0, 10))  
 
button_start_capture = tk.Button(padding_frame, text="Scan Wajah", command=start_capture, width=90, height=3, font=("Arial", 12, "bold"), bg=button_bg, fg=button_fg)
button_start_capture.pack(pady=5)

root.mainloop()