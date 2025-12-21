import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import os

class MosaicApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI 얼굴 인식 모자이크 by ESC")
        self.geometry("500x400")
        
       
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # UI 구성
        ctk.CTkLabel(self, text="얼굴 모자이크", font=("Arial", 24, "bold")).pack(pady=20)
        
        self.slider_blur = ctk.CTkSlider(self, from_=5, to=100)
        self.slider_blur.set(30)
        self.slider_blur.pack(pady=10)
        ctk.CTkLabel(self, text="모자이크 강도").pack()

        self.btn_run = ctk.CTkButton(self, text="사진 선택하기", command=self.process_image)
        self.btn_run.pack(pady=30)
        
        self.status_label = ctk.CTkLabel(self, text="준비 완료", text_color="gray")
        self.status_label.pack()

    def process_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return

        self.status_label.configure(text="얼굴 찾는 중...", text_color="yellow")
        self.update()

        try:
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. 얼굴 탐지 실행
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # 얼굴 영역만 추출
                    roi = img[y:y+h, x:x+w]
                    # 모자이크 처리 
                    level = int(self.slider_blur.get())
                    roi = cv2.resize(roi, (w//level, h//level))
                    roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
                    # 원본에 덮어쓰기
                    img[y:y+h, x:x+w] = roi
                msg = f"{len(faces)}명의 얼굴을 찾았습니다."
            else:
                msg = "얼굴을 찾지 못했습니다."

            # 3. 저장
            out_path = file_path.replace(".", "_anon.")
            cv2.imwrite(out_path, img)
            messagebox.showinfo("성공", f"{msg}\n저장 위치: {out_path}")
            self.status_label.configure(text="완료", text_color="green")

        except Exception as e:
            messagebox.showerror("오류", str(e))

if __name__ == "__main__":
    app = MosaicApp()
    app.mainloop()