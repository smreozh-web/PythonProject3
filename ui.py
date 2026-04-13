from tkinter import Tk, filedialog, Toplevel, Label, Frame
import tkinter as tk

def select_video():
    root=Tk()
    root.withdraw()

    source=filedialog.askopenfilename(
        title="영상 파일 선택",
        filetypes=[("Video files","*.mp4 *.mov *.avi *.mkv")]
    )
    return root, source

def select_speed(root):
    speed_window = Toplevel(root)
    speed_window.title("속도 선택")
    speed_window.geometry("300x200")

    selected_speed = {"value": None}

    def set_speed(val):
        selected_speed["value"] = val
        speed_window.destroy()

    Label(speed_window, text="러닝 속도 선택", font=("Arial", 14)).pack(pady=20)

    tk.Button(speed_window, text="8 km/h", width=10, command=lambda: set_speed(8)).pack(pady=5)
    tk.Button(speed_window, text="10 km/h", width=10, command=lambda: set_speed(10)).pack(pady=5)
    tk.Button(speed_window, text="12 km/h", width=10, command=lambda: set_speed(12)).pack(pady=5)

    root.wait_window(speed_window)

    return selected_speed["value"]