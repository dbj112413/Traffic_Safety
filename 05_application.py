from program_functions import *
import tkinter as tk
from tkinter import ttk

if __name__ == "__main__":
    root = tk.Tk()
    
    # 解決 Windows 模糊問題的高 DPI 設定 (強烈建議加入)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    style = ttk.Style()
    
    # 使用 clam 主題，它比預設的 Windows 主題更容易客製化顏色
    if "clam" in style.theme_names():
        style.theme_use("clam")
        
    # ==========================================
    # 全局字體平滑設定
    # ==========================================
    default_font = ("Microsoft JhengHei", 11)
    style.configure('.', font=default_font) 
    style.configure('TLabelframe.Label', font=("Microsoft JhengHei", 11, "bold"))
    style.configure('TLabel', font=default_font)
    root.option_add('*Listbox.font', default_font) # 修正下拉選單內部的字體

    # ==========================================
    # 👇 徹底解決 Combobox 藍底反白問題 👇
    # ==========================================
    # 重新映射 TCombobox 的顏色，當它處於 focus 或 select 狀態時，
    # 強制將背景色 (fieldbackground) 與選取色 (selectbackground) 設回白色
    style.map('TCombobox', 
              fieldbackground=[('readonly', 'white'), ('focus', 'white'), ('active', 'white')],
              selectbackground=[('readonly', 'white'), ('focus', 'white'), ('active', 'white')],
              selectforeground=[('readonly', 'black'), ('focus', 'black'), ('active', 'black')],
              background=[('readonly', '#f0f0f0'), ('active', '#e5e5e5')]) # 這是旁邊下拉按鈕的顏色
    # ==========================================
    
    app = TrafficRiskApp(root)
    root.mainloop()
    
    