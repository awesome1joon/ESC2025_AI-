import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import os

# Set modern appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MosaicApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI 얼굴 인식 모자이크 by ESC")
        self.geometry("500x400")
        
       
        # Robustly load Haar cascade; some Windows user paths with non-ASCII
        # characters cause OpenCV's FileStorage to fail. Try multiple locations
        # and fall back to copying the XML to a safe ASCII path.
        cascade_name = 'haarcascade_frontalface_default.xml'
        candidates = []
        try:
            candidates.append(os.path.join(cv2.data.haarcascades, cascade_name))
        except Exception:
            pass
        try:
            candidates.append(os.path.join(os.path.dirname(cv2.__file__), 'data', cascade_name))
        except Exception:
            pass
        try:
            candidates.append(os.path.join(os.path.dirname(cv2.__file__), cascade_name))
        except Exception:
            pass

        cascade_path = None
        for p in candidates:
            try:
                if p and os.path.exists(p):
                    cascade_path = p
                    break
            except Exception:
                continue

        if cascade_path is None:
            # last-resort: search inside the cv2 package directory
            try:
                base = os.path.dirname(cv2.__file__)
                for root, dirs, files in os.walk(base):
                    if cascade_name in files:
                        cascade_path = os.path.join(root, cascade_name)
                        break
            except Exception:
                cascade_path = None

        face_cascade = None
        if cascade_path:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                # try copying to current working directory (ASCII path) and reload
                try:
                    with open(cascade_path, 'rb') as rf:
                        data = rf.read()
                    alt_path = os.path.join(os.getcwd(), cascade_name)
                    with open(alt_path, 'wb') as wf:
                        wf.write(data)
                    face_cascade = cv2.CascadeClassifier(alt_path)
                except Exception:
                    face_cascade = cv2.CascadeClassifier()
        else:
            face_cascade = cv2.CascadeClassifier()

        self.face_cascade = face_cascade

        # UI 구성
        ctk.CTkLabel(self, text="얼굴 모자이크", font=("Arial", 24, "bold")).pack(pady=20)
        
        self.slider_blur = ctk.CTkSlider(self, from_=5, to=100)
        self.slider_blur.set(30)
        self.slider_blur.pack(pady=10)
        ctk.CTkLabel(self, text="모자이크 강도").pack()

        self.btn_run = ctk.CTkButton(self, text="사진 선택하기", command=self.process_image)
        self.btn_run.pack(pady=10)
        
        self.btn_video = ctk.CTkButton(self, text="비디오 선택하기", command=self.process_video)
        self.btn_video.pack(pady=10)
        
        self.status_label = ctk.CTkLabel(self, text="준비 완료", text_color="gray")
        self.status_label.pack()

    def process_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return

        self.status_label.configure(text="얼굴 찾는 중...", text_color="orange")
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

    def process_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not file_path: return

        self.status_label.configure(text="비디오 처리 중...", text_color="orange")
        self.update()

        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("비디오 파일을 열 수 없습니다.")

            # 비디오 속성 가져오기
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # 출력 파일 경로
            out_path = file_path.replace(".", "_anon.")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            total_faces = 0
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        roi = frame[y:y+h, x:x+w]
                        level = int(self.slider_blur.get())
                        roi = cv2.resize(roi, (w//level, h//level))
                        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
                        frame[y:y+h, x:x+w] = roi
                    total_faces += len(faces)

                out.write(frame)

                # 진행 상황 업데이트 (예: 10프레임마다)
                if frame_count % 10 == 0:
                    self.status_label.configure(text=f"처리 중... 프레임 {frame_count}", text_color="orange")
                    self.update()

            cap.release()
            out.release()

            msg = "비디오 처리 완료."
            messagebox.showinfo("성공", f"{msg}\n저장 위치: {out_path}")
            self.status_label.configure(text="완료", text_color="green")

        except Exception as e:
            messagebox.showerror("오류", str(e))
            self.status_label.configure(text="오류 발생", text_color="red")

if __name__ == "__main__":
    app = MosaicApp()
    app.mainloop()