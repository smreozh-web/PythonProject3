from tkinter import Toplevel, Label, Frame
import tkinter as tk

def show_report(root, data, SPEED, VO_MIN, VO_MAX, THIGH_MIN, THIGH_MAX):
    avg_lean = data["summary"]["lean"]["average"]
    avg_knee = data["summary"]["knee"]["average"]
    avg_arm = data["summary"]["arm"]["average"]

    avg_vo = data["summary"]["vo"]["average"]
    avg_thigh = data["summary"]["thigh"]["average"]

    report=Toplevel(root)
    report.title("러닝 기록지")
    report.geometry("450x700")

    # =========================
    # 스크롤 구조 생성
    # =========================
    canvas = tk.Canvas(report, bg="#eef3f6")
    scrollbar = tk.Scrollbar(report, orient="vertical", command=canvas.yview)

    scrollable_frame = tk.Frame(canvas, bg="#eef3f6")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # =========================
    # 헤더
    # =========================
    Label(scrollable_frame,text="MY RECORD",
          bg="#eef3f6",fg="#3aaed8",
          font=("Arial",12,"bold")).pack(anchor="w",padx=20,pady=(20,0))

    Label(scrollable_frame,text="OO님의 기록",
          bg="#eef3f6",
          font=("Arial",28,"bold")).pack(anchor="w",padx=20,pady=(0,20))


    # =========================
    # 총평 박스
    # =========================
    summary_box=Frame(scrollable_frame,bg="white")
    summary_box.pack(fill="x",padx=20,pady=10)

    inner=Frame(summary_box,bg="white")
    inner.pack(fill="both",expand=True,padx=18,pady=18)

    Label(inner,text="총평",bg="white",
          font=("Arial",18,"bold")).pack(anchor="w")

    Label(inner,
          text="전반적으로 안정적인 러닝 자세를 유지하고 있습니다.",
          bg="white",
          wraplength=360,
          justify="left",
          font=("Arial",12)).pack(anchor="w",pady=(6,0))


    # =========================
    # 카드 영역 (2x2)
    # =========================
    cards_frame=Frame(scrollable_frame,bg="#eef3f6")
    cards_frame.pack(fill="both",expand=True,padx=14,pady=10)

    cards_frame.columnconfigure(0,weight=1)
    cards_frame.columnconfigure(1,weight=1)


    def make_card(parent,row,col,title,value,min_v,max_v):

        card=Frame(parent,bg="#ffffff")
        card.grid(row=row,column=col,padx=10,pady=10,sticky="nsew")

        inner=Frame(card,bg="#ffffff")
        inner.pack(fill="both",expand=True,padx=18,pady=18)

        Label(inner,text=title,bg="white",
              font=("Arial",16,"bold")).pack(anchor="w")

        Label(inner,text=f"권장: {min_v}~{max_v}",
              bg="white",fg="#6b7280",
              font=("Arial",11)).pack(anchor="w",pady=(0,8))

        top=Frame(inner,bg="white")
        top.pack(fill="x")

        Label(top,text="100M/H",bg="white",fg="#6b7280",
              font=("Arial",12)).pack(side="left")

        Label(top,text=f"{value:.1f}",bg="white",fg="#1aa3b0",
              font=("Arial",30,"bold")).pack(side="right")

        Label(top,text="",bg="white",fg="#1aa3b0",
              font=("Arial",16,"bold")).pack(side="right")

        # 진행바
        bar_bg=Frame(inner,bg="#d1d5db",height=10)
        bar_bg.pack(fill="x",pady=10)

        if title=="팔 스윙":
            ratio = (value + 10) / 17
        else:
            ratio=(value-min_v)/(max_v-min_v)

        ratio = max(0, min(1, ratio))

        width=260
        bar_fg=Frame(bar_bg,bg="#020617",height=10,width=int(width*ratio))
        bar_fg.place(x=0,y=0)

        # min / max
        mm=Frame(inner,bg="white")
        mm.pack(fill="x")
        Label(mm,text=f"{min_v}",bg="white",fg="#6b7280").pack(side="left")
        Label(mm,text=f"{max_v}",bg="white",fg="#6b7280").pack(side="right")

        # 상태 배지
        ok=min_v<=value<=max_v

        if ok:
            txt="↗ 양호"
            fg="#2563eb"
            bg="#e5efff"
        else:
            txt="⚠ 개선 필요"
            fg="#dc2626"
            bg="#fee2e2"

        badge=Frame(inner,bg=bg)
        badge.pack(pady=12)

        Label(badge,text=txt,bg=bg,fg=fg,
              font=("Arial",11,"bold"),
              padx=10,pady=4).pack()

        return ok


    # 카드 4개
    ok1=make_card(cards_frame,0,0,"상체 기울기",avg_lean,0,10)
    ok2=make_card(cards_frame,0,1,"무릎 각도",avg_knee,150,176)
    ok4=make_card(cards_frame,1,1,"팔 스윙",avg_arm,-10,7)
    ok5=make_card(cards_frame,1,0,"수직 진폭",avg_vo,VO_MIN,VO_MAX)
    ok6 = make_card(cards_frame,2,0,"무릎 높이",avg_thigh,THIGH_MIN,THIGH_MAX)

    # =========================
    # 개선포인트
    # =========================
    tips=[]
    if not ok1: tips.append("상체를 조금 더 세우면 효율이 좋아집니다.")
    if not ok2: tips.append("무릎 각도를 조정하면 착지 안정성이 좋아집니다.")
    if not ok4: tips.append("팔 스윙 범위를 조정하면 러닝 효율이 좋아집니다.")

    if not tips:
        msg="전반적으로 우수한 자세를 유지하고 있습니다. 효율적인 러닝을 하고 있습니다."
    else:
        msg=" ".join(tips)

    improve=Frame(scrollable_frame,bg="#dbeafe")
    improve.pack(fill="x",padx=20,pady=20)

    inner=Frame(improve,bg="#dbeafe")
    inner.pack(fill="both",expand=True,padx=16,pady=16)

    Label(inner,text="개선 포인트",
          bg="#dbeafe",
          font=("Arial",16,"bold")).pack(anchor="w")

    Label(inner,text=msg,
          wraplength=360,
          justify="left",
          bg="#dbeafe",
          font=("Arial",12)).pack(anchor="w",pady=(6,0))
