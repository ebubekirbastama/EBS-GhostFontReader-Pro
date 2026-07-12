#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from ghost_reader import Config, analyze, default_roi, resize_keep


class RoiDialog(tk.Toplevel):
    def __init__(self, parent, frame):
        super().__init__(parent)
        self.title("Yazı alanını seçin")
        self.transient(parent)
        self.grab_set()
        self.result = None
        self.start = None
        self.rect = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.scale = min(1.0, 1150/image.width, 720/image.height)
        if self.scale < 1:
            image = image.resize((round(image.width*self.scale), round(image.height*self.scale)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        ttk.Label(self, text="Yalnızca hareketli yazının bulunduğu kutuyu seçin.").pack(padx=10, pady=8)
        self.canvas = tk.Canvas(self, width=image.width, height=image.height, cursor="cross")
        self.canvas.pack(padx=10)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        f = ttk.Frame(self); f.pack(fill="x", padx=10, pady=10)
        ttk.Button(f, text="Seçimi Kullan", command=self.accept).pack(side="left")
        ttk.Button(f, text="Varsayılan", command=self.use_default).pack(side="left", padx=8)
        ttk.Button(f, text="İptal", command=self.destroy).pack(side="right")

    def press(self, e):
        self.start = (e.x, e.y)
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=3)
    def drag(self, e):
        if self.start: self.canvas.coords(self.rect, self.start[0], self.start[1], e.x, e.y)
    def release(self, e):
        if not self.start: return
        x1,y1=self.start; x2,y2=e.x,e.y
        x,y=min(x1,x2),min(y1,y2); w,h=abs(x2-x1),abs(y2-y1)
        if w>8 and h>8: self.result=(round(x/self.scale),round(y/self.scale),round(w/self.scale),round(h/self.scale))
    def accept(self):
        if not self.result:
            messagebox.showwarning("Seçim", "Önce yazı alanını seçin.", parent=self); return
        self.destroy()
    def use_default(self): self.result="default"; self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ghost Font Reader v4")
        self.geometry("820x520")
        self.video=tk.StringVar(); self.output=tk.StringVar(value=str(Path.cwd()/"ghost_output"))
        self.ocr=tk.BooleanVar(value=False); self.lang=tk.StringVar(value="eng"); self.vertical=tk.BooleanVar(value=True)
        self.status=tk.StringVar(value="Video seçin."); self.roi=None; self.result=None
        self.row("Video",self.video,self.pick_video,0); self.row("Çıktı",self.output,self.pick_output,1)
        opts=ttk.Frame(self); opts.grid(row=2,column=1,sticky="w",padx=8,pady=6)
        ttk.Checkbutton(opts,text="Dikey harf sürüklenmesini düzelt",variable=self.vertical).pack(side="left")
        ttk.Checkbutton(opts,text="Tesseract OCR uygula",variable=self.ocr).pack(side="left",padx=(14,0))
        ttk.Label(opts,text="Dil:").pack(side="left",padx=(14,4)); ttk.Combobox(opts,textvariable=self.lang,values=["eng","tur","tur+eng"],width=9,state="readonly").pack(side="left")
        buttons=ttk.Frame(self); buttons.grid(row=3,column=1,sticky="w",padx=8,pady=10)
        ttk.Button(buttons,text="Alan Seç ve Gelişmiş Analiz",command=self.prepare).pack(side="left")
        self.open_btn=ttk.Button(buttons,text="Sonuç Klasörünü Aç",command=self.open_output,state="disabled"); self.open_btn.pack(side="left",padx=8)
        self.bar=ttk.Progressbar(self,maximum=100); self.bar.grid(row=4,column=0,columnspan=3,sticky="ew",padx=12,pady=8)
        ttk.Label(self,textvariable=self.status,wraplength=780).grid(row=5,column=0,columnspan=3,sticky="w",padx=12,pady=8)
        self.preview=ttk.Label(self,anchor="center"); self.preview.grid(row=6,column=0,columnspan=3,sticky="nsew",padx=12,pady=8)
        self.columnconfigure(1,weight=1); self.rowconfigure(6,weight=1)

    def row(self,label,var,cmd,r):
        ttk.Label(self,text=label).grid(row=r,column=0,padx=8,pady=8,sticky="w")
        ttk.Entry(self,textvariable=var).grid(row=r,column=1,padx=8,pady=8,sticky="ew")
        ttk.Button(self,text="Seç",command=cmd).grid(row=r,column=2,padx=8,pady=8)
    def pick_video(self):
        p=filedialog.askopenfilename(filetypes=[("Video","*.webm *.mp4 *.avi *.mov *.mkv"),("Tümü","*.*")])
        if p:self.video.set(p)
    def pick_output(self):
        p=filedialog.askdirectory()
        if p:self.output.set(p)
    def prepare(self):
        p=Path(self.video.get())
        if not p.exists(): messagebox.showerror("Hata","Geçerli video seçin."); return
        cap=cv2.VideoCapture(str(p)); ok,frame=cap.read(); cap.release()
        if not ok: messagebox.showerror("Hata","Video okunamadı."); return
        frame,_=resize_keep(frame,1200)
        d=RoiDialog(self,frame); self.wait_window(d)
        if d.result is None:return
        self.roi=default_roi(frame) if d.result=="default" else d.result
        self.run()
    def run(self):
        self.bar["value"]=0; self.status.set("Analiz başlatılıyor..."); self.open_btn["state"]="disabled"
        def cb(n,s): self.after(0,lambda:self.update_progress(n,s))
        def job():
            try:
                r=analyze(Config(input=self.video.get(),output=self.output.get(),roi=self.roi,vertical_compensate=self.vertical.get(),ocr=self.ocr.get(),lang=self.lang.get()),cb)
                self.after(0,lambda:self.done(r))
            except Exception as e:self.after(0,lambda e=e:self.fail(e))
        threading.Thread(target=job,daemon=True).start()
    def update_progress(self,n,s): self.bar["value"]=n; self.status.set(s)
    def done(self,r):
        self.result=r; self.open_btn["state"]="normal"
        text=f"Tamamlandı: {r['frames_analyzed']} kare. Ana sonuç: 12_BEST_READABLE.png\nDikey kayma: {r.get('vertical_shift_min',0):.1f} ile {r.get('vertical_shift_max',0):.1f} piksel"
        if r.get("ocr_text"): text+=f"\nOCR adayı: {r['ocr_text']}"
        if r.get("ocr_error"): text+=f"\nOCR uyarısı: {r['ocr_error']}"
        self.status.set(text); self.show_preview(r["best_image"]); messagebox.showinfo("Tamamlandı",text)
    def fail(self,e): self.status.set(f"Hata: {e}"); messagebox.showerror("Hata",str(e))
    def show_preview(self,path):
        im=Image.open(path); im.thumbnail((780,260),Image.Resampling.LANCZOS); self.preview_img=ImageTk.PhotoImage(im); self.preview.configure(image=self.preview_img)
    def open_output(self):
        p=self.output.get()
        try: os.startfile(p)
        except AttributeError: subprocess.Popen(["xdg-open",p])

if __name__=="__main__": App().mainloop()
