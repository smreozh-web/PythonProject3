from ui import select_video, select_speed
from analysis import run_analysis
from report import show_report
from config import get_vo_range, get_thigh_range
from db import db

def main():

    # =========================
    # 영상 선택
    # =========================
    root, source = select_video()

    if not source:
        print("영상 선택 안됨")
        return

    # =========================
    # 속도 선택
    # =========================
    SPEED = select_speed(root)

    if SPEED is None:
        print("속도 선택 안됨")
        return

    # =========================
    # 분석 실행
    # =========================
    data = run_analysis(source, SPEED)

    # Firebase 저장
    db.collection("users").document(data["userId"]).set(data)
    # =========================
    # 기준값 가져오기
    # =========================
    VO_MIN, VO_MAX = get_vo_range(SPEED)
    THIGH_MIN, THIGH_MAX = get_thigh_range(SPEED)

    # =========================
    # 리포트 출력
    # =========================
    show_report(
        root,
        data,
        SPEED,
        VO_MIN,
        VO_MAX,
        THIGH_MIN,
        THIGH_MAX
    )

    # =========================
    # UI 유지
    # =========================
    root.mainloop()


# =========================
# 프로그램 시작
# =========================
if __name__ == "__main__":
    main()