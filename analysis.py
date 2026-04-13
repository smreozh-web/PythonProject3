from ultralytics import YOLO
import cv2
import numpy as np
from metrics import *

def run_analysis(source, SPEED):

    model=YOLO("yolo26n-pose.pt")

    KNEE_MIN,KNEE_MAX=150,176

    ARM_FWD_MIN, ARM_FWD_MAX = 0.0, 0.14
    ARM_BWD_MIN, ARM_BWD_MAX = -0.2, -0.04

    if SPEED == 8:
        VO_MIN, VO_MAX = 6, 11
    elif SPEED == 10:
        VO_MIN, VO_MAX = 6, 10
    elif SPEED == 12:
        VO_MIN, VO_MAX = 5, 9

    if SPEED == 8:
        THIGH_MIN, THIGH_MAX = 35, 50
    elif SPEED == 10:
        THIGH_MIN, THIGH_MAX = 40, 55
    elif SPEED == 12:
        THIGH_MIN, THIGH_MAX = 45, 60

    total=0
    warns={"knee":0,"lean":0,"arm":0}

    lean_list=[]; knee_list=[]; arm_list=[]
    vo_list=[]; thigh_list=[]

    BUFFER_SIZE = 60
    frame_buffer=[]

    hip_y_buffer = []
    VO_BUFFER_SIZE = BUFFER_SIZE
    vo_value_buffer = []
    VO_VALUE_BUFFER_SIZE = BUFFER_SIZE

    best_clip=[];best_box=None;tail_frames=0;max_lean_error=-1
    best_knee_clip=[];best_knee_box=None;knee_tail_frames=0;max_knee_error=-1

    results=model(source,stream=True)

    for r in results:

        frame=r.plot()
        if frame is None or frame.size==0:
            continue

        frame_buffer.append(frame.copy())
        if len(frame_buffer)>BUFFER_SIZE:
            frame_buffer.pop(0)

        if r.keypoints is not None:

            for person in r.keypoints.data:

                pts=[(int(k[0]),int(k[1])) for k in person]
                conf=[k[2] for k in person]

                left=min(conf[5],conf[7],conf[9],conf[11],conf[13],conf[15])
                right=min(conf[6],conf[8],conf[10],conf[12],conf[14],conf[16])

                if left>right:
                    shoulder,elbow,wrist,hip,knee,ankle=pts[5],pts[7],pts[9],pts[11],pts[13],pts[15]
                else:
                    shoulder,elbow,wrist,hip,knee,ankle=pts[6],pts[8],pts[10],pts[12],pts[14],pts[16]

                if min(left,right)<0.5:
                    continue

                knee_angle = angle(hip, knee, ankle)
                lean=body_lean_angle(shoulder,hip)

                thigh_ang = thigh_angle(shoulder, hip, knee)
                knee_lift = 180 - thigh_ang

                body_pixel = distance(shoulder, hip)
                scale = 35 / (body_pixel + 1e-6)

                hip_center = (
                    (pts[11][0] + pts[12][0]) // 2,
                    (pts[11][1] + pts[12][1]) // 2
                )

                hip_y_buffer.append(hip_center[1])
                if len(hip_y_buffer) > VO_BUFFER_SIZE:
                    hip_y_buffer.pop(0)

                if len(hip_y_buffer) >= 10:
                    vo_pixel = max(hip_y_buffer) - min(hip_y_buffer)
                    vertical_osc_cm = vo_pixel * scale
                else:
                    vertical_osc_cm = 0

                vo_value_buffer.append(vertical_osc_cm)
                if len(vo_value_buffer) > VO_VALUE_BUFFER_SIZE:
                    vo_value_buffer.pop(0)

                if vo_value_buffer:
                    vo_avg = sum(vo_value_buffer) / len(vo_value_buffer)
                else:
                    vo_avg = 0

                elbow_hip_ratio = abs(elbow[0] - hip[0]) / (distance(shoulder, hip) + 1e-6)

                elbow_body_pixel = point_to_line_distance(elbow, shoulder, hip)
                elbow_body_cm = elbow_body_pixel * scale

                forward_vec = np.array(shoulder) - np.array(hip)
                elbow_vec = np.array(elbow) - np.array(hip)

                direction = np.sign(np.dot(elbow_vec, forward_vec))
                arm_swing_val = elbow_body_cm * direction

                upper_arm_cm = distance(shoulder, elbow) * scale
                lower_arm_cm = distance(elbow, wrist) * scale

                # ================= 색상 =================
                if arm_swing_val >= 0:
                    arm_ok = 0 <= arm_swing_val <= 7
                else:
                    arm_ok = -10 <= arm_swing_val <= -2

                knee_ok = KNEE_MIN <= knee_angle <= KNEE_MAX

                knee_color = (0,255,0) if knee_ok else (0,0,255)
                arm_color = (0,255,0) if arm_ok else (0,0,255)

                thigh_ok = THIGH_MIN <= knee_lift <= THIGH_MAX
                thigh_color = (0,255,0) if thigh_ok else (0,0,255)

                vo_ok = VO_MIN <= vo_avg <= VO_MAX
                vo_color = (0,255,0) if vo_ok else (0,0,255)

                if not arm_ok: warns["arm"]+=1
                if not knee_ok: warns["knee"]+=1
                if lean>10: warns["lean"]+=1

                # ================= 그리기 =================
                for p in [shoulder,elbow,wrist,hip,knee,ankle]:
                    cv2.circle(frame,p,6,(0,255,0),-1)

                cv2.line(frame, shoulder, elbow, arm_color, 3)
                cv2.line(frame, elbow, wrist, arm_color, 3)
                direction_text = "FWD" if arm_swing_val >= 0 else "BWD"

                cv2.putText(frame,
                            f"{direction_text}: {arm_swing_val:.1f}cm",
                            (elbow[0], elbow[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            arm_color,
                            2)

                cv2.line(frame, hip, knee, knee_color, 2)
                cv2.line(frame, ankle, knee, knee_color, 2)

                cv2.line(frame, shoulder, hip, (0,255,0), 4)

                # ================= 텍스트 =================
                cv2.putText(frame,f"{lean:.1f}",shoulder,
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                cv2.putText(frame,
                    f"SPEED: {SPEED} km/h",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)

                cv2.putText(frame,
                    f"VO(avg): {vo_avg:.1f}cm",
                    (hip_center[0], hip_center[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    vo_color,
                    2)

                cv2.putText(frame,
                    f"{int(knee_angle)}deg",
                    knee,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    knee_color,
                    2)

                cv2.putText(frame,
                    f"TH: {int(knee_lift)}",
                    (hip[0], hip[1]-40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    thigh_color,
                    2)

                knee_list.append(knee_angle)
                lean_list.append(lean)
                arm_list.append(arm_swing_val)
                vo_list.append(vo_avg)
                thigh_list.append(knee_lift)
                total+=1

        if tail_frames>0: best_clip.append(frame.copy()); tail_frames-=1
        if knee_tail_frames>0: best_knee_clip.append(frame.copy()); knee_tail_frames-=1

        cv2.imshow("Running Coach",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cv2.destroyAllWindows()

    def play_slow(clip,box,title):
        if not clip or not box: return
        x1,y1,x2,y2=box
        for f in clip:
            h,w=f.shape[:2]
            crop=f[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
            if crop.size==0: continue
            crop=cv2.resize(crop,None,fx=2,fy=2)
            cv2.imshow(title,crop)
            if cv2.waitKey(60)&0xFF==ord('q'): break
        cv2.destroyWindow(title)

    # (기존 코드 끝)
    if max_lean_error > 0: play_slow(best_clip, best_box, "Worst Lean")
    if max_knee_error > 0: play_slow(best_knee_clip, best_knee_box, "Worst Knee")

    # =========================
    # 평균 계산
    # =========================
    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0

    arm_avg = safe_avg(arm_list)
    knee_avg = safe_avg(knee_list)
    lean_avg = safe_avg(lean_list)

    # =========================
    # 상태 판단
    # =========================
    def get_status(val, min_v, max_v):
        return "good" if min_v <= val <= max_v else "bad"

    arm_status = "good" if (-10 <= arm_avg <= 7) else "bad"
    knee_status = get_status(knee_avg, KNEE_MIN, KNEE_MAX)
    lean_status = "good" if lean_avg <= 10 else "bad"

    # =========================
    # 🔥 여기다 넣는거다
    # =========================
    import datetime

    result = {
        "userId": "jinwook_test",
        "userName": "진욱",
        "speed": f"{SPEED}km/h",
        "totalScore": int(100 - (warns["arm"] + warns["knee"] + warns["lean"]) * 2),
        "createdAt": datetime.datetime.utcnow().isoformat(),

        "summary": {
            "arm": {
                "average": round(arm_avg, 1),
                "status": arm_status,
                "range": [-10, 7]
            },
            "knee": {
                "average": round(knee_avg, 1),
                "status": knee_status,
                "range": [KNEE_MIN, KNEE_MAX]
            },
            "lean": {
                "average": round(lean_avg, 1),
                "status": lean_status,
                "range": [0, 10]
            }
        },

        "details": {
            "overallFeedback":
                "상체가 앞으로 쏠림" if lean_avg > 10 else
                "무릎이 낮음" if knee_avg < KNEE_MIN else
                "자세 양호 👍",

            "fullVideoLink": "https://firebasestorage...",

            "parts": {
                "arm": {
                    "evaluation": "팔치기 부족" if arm_status == "bad" else "양호",
                    "videoLink": "https://firebasestorage.../arm.mp4"
                },
                "knee": {
                    "evaluation": "무릎 낮음" if knee_status == "bad" else "양호",
                    "videoLink": "https://firebasestorage.../knee.mp4"
                },
                "lean": {
                    "evaluation": "상체 과도" if lean_status == "bad" else "양호",
                    "videoLink": "https://firebasestorage.../lean.mp4"
                }
            }
        }
    }

    return result