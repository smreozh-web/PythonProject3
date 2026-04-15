from ultralytics import YOLO
import cv2
import numpy as np
import imageio_ffmpeg as ffmpeg
import subprocess
import os
from metrics import *
from openai import OpenAI
client = OpenAI(api_key="키값")



def generate_all_feedback(
    arm, knee, lean, vo, thigh,
    arm_avg, knee_avg, lean_avg, vo_avg, thigh_avg,
    knee_min, knee_max, vo_min, vo_max, thigh_min, thigh_max
):
    prompt = f"""
너는 러닝 자세를 분석해주는 전문 코치야.

다음 분석 결과를 바탕으로
1) 전체 총평(overallFeedback)
2) 관절별 평가(parts.arm, parts.knee, parts.lean, parts.vo, parts.thigh)
를 한 번에 만들어줘.

[상태]
- 팔 스윙: {arm}
- 무릎 각도: {knee}
- 상체 기울기: {lean}
- 수직 진폭: {vo}
- 무릎 높이: {thigh}

[평균값]
- 팔 스윙 평균: {arm_avg}
- 무릎 각도 평균: {knee_avg}
- 상체 기울기 평균: {lean_avg}
- 수직 진폭 평균: {vo_avg}
- 무릎 높이 평균: {thigh_avg}

[권장 범위]
- 팔 스윙: -10 ~ 7
- 무릎 각도: {knee_min} ~ {knee_max}
- 상체 기울기: 0 ~ 10
- 수직 진폭: {vo_min} ~ {vo_max}
- 무릎 높이: {thigh_min} ~ {thigh_max}

작성 규칙:
- 한국어로 작성
- overallFeedback는 3~5문장
- 자세가 좋으면 좋은 점과 유지 시 이점을 설명
- 자세가 나쁘면 문제점, 부상 위험, 교정 시 이점을 설명
- parts의 각 evaluation은 2문장 이내
- 코치처럼 자연스럽고 이해하기 쉽게 작성
- 반드시 아래 JSON 형식으로만 답변해

JSON 형식:
{{
  "overallFeedback": "문장",
  "parts": {{
    "arm": {{"evaluation": "문장"}},
    "knee": {{"evaluation": "문장"}},
    "lean": {{"evaluation": "문장"}},
    "vo": {{"evaluation": "문장"}},
    "thigh": {{"evaluation": "문장"}}
  }}
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    import json
    return json.loads(response.choices[0].message.content)

def convert_to_h264(input_path, output_path):
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    subprocess.run(command, check=True)

def crop_and_resize(frame, points, output_size=(480, 640), margin=100):
    h, w = frame.shape[:2]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(w, max(xs) + margin)
    y2 = min(h, max(ys) + margin)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return cv2.resize(frame, output_size)

    return cv2.resize(crop, output_size)


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

    arm_count = 0
    knee_count = 0
    lean_count = 0
    vo_count = 0
    thigh_count = 0

    BUFFER_SIZE = 60
    frame_buffer=[]

    hip_y_buffer = []
    VO_BUFFER_SIZE = BUFFER_SIZE
    vo_value_buffer = []
    VO_VALUE_BUFFER_SIZE = BUFFER_SIZE

    best_clip=[];best_box=None;tail_frames=0;max_lean_error=-1
    best_knee_clip=[];best_knee_box=None;knee_tail_frames=0;max_knee_error=-1

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    highlight_fps = 8
    highlight_size = (480, 640)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    arm_writer = cv2.VideoWriter("arm_highlight_raw.mp4", fourcc, highlight_fps, highlight_size)
    knee_writer = cv2.VideoWriter("knee_highlight_raw.mp4", fourcc, highlight_fps, highlight_size)
    lean_writer = cv2.VideoWriter("lean_highlight_raw.mp4", fourcc, highlight_fps, highlight_size)
    vo_writer = cv2.VideoWriter("vo_highlight_raw.mp4", fourcc, highlight_fps, highlight_size)
    thigh_writer = cv2.VideoWriter("thigh_highlight_raw.mp4", fourcc, highlight_fps, highlight_size)

    print("fps:", fps, "highlight_fps:", highlight_fps, "highlight_size:", highlight_size)
    print("arm_writer:", arm_writer.isOpened())
    print("knee_writer:", knee_writer.isOpened())
    print("lean_writer:", lean_writer.isOpened())
    print("vo_writer:", vo_writer.isOpened())
    print("thigh_writer:", thigh_writer.isOpened())

    results=model(source,stream=True)

    for r in results:

        frame = r.plot()
        if frame is None or frame.size == 0:
            continue

        print(frame.shape)

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

                if min(left,right)<0.75:
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

                # ================= 하이라이트 저장 =================
                if not arm_ok:
                    arm_crop = crop_and_resize(
                        frame,
                        [shoulder, elbow, wrist, hip],
                        output_size=highlight_size,
                        margin=100
                    )
                    arm_writer.write(arm_crop)
                    arm_count += 1

                if not knee_ok:
                    knee_crop = crop_and_resize(
                        frame,
                        [hip, knee, ankle],
                        output_size=highlight_size,
                        margin=120
                    )
                    knee_writer.write(knee_crop)
                    knee_count += 1

                if lean > 10:
                    lean_crop = crop_and_resize(
                        frame,
                        [shoulder, hip, knee],
                        output_size=highlight_size,
                        margin=120
                    )
                    lean_writer.write(lean_crop)
                    lean_count += 1

                if not vo_ok:
                    vo_crop = crop_and_resize(
                        frame,
                        [shoulder, hip, knee, ankle],
                        output_size=highlight_size,
                        margin=140
                    )
                    vo_writer.write(vo_crop)
                    vo_count += 1

                if not thigh_ok:
                    thigh_crop = crop_and_resize(
                        frame,
                        [hip, knee, ankle],
                        output_size=highlight_size,
                        margin=120
                    )
                    thigh_writer.write(thigh_crop)
                    thigh_count += 1

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

    print("arm frames:", arm_count)
    print("knee frames:", knee_count)
    print("lean frames:", lean_count)
    print("vo frames:", vo_count)
    print("thigh frames:", thigh_count)

    arm_writer.release()
    knee_writer.release()
    lean_writer.release()
    vo_writer.release()
    thigh_writer.release()

    convert_to_h264("arm_highlight_raw.mp4", "arm_highlight.mp4")
    convert_to_h264("knee_highlight_raw.mp4", "knee_highlight.mp4")
    convert_to_h264("lean_highlight_raw.mp4", "lean_highlight.mp4")
    convert_to_h264("vo_highlight_raw.mp4", "vo_highlight.mp4")
    convert_to_h264("thigh_highlight_raw.mp4", "thigh_highlight.mp4")

    print("arm final size:", os.path.getsize("arm_highlight.mp4"))
    print("knee final size:", os.path.getsize("knee_highlight.mp4"))
    print("lean final size:", os.path.getsize("lean_highlight.mp4"))
    print("vo final size:", os.path.getsize("vo_highlight.mp4"))
    print("thigh final size:", os.path.getsize("thigh_highlight.mp4"))

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
    vo_avg = safe_avg(vo_list)
    thigh_avg = safe_avg(thigh_list)

    # =========================
    # 상태 판단
    # =========================
    def get_status(val, min_v, max_v):
        return "good" if min_v <= val <= max_v else "bad"

    arm_status = "good" if (-10 <= arm_avg <= 7) else "bad"
    knee_status = get_status(knee_avg, KNEE_MIN, KNEE_MAX)
    lean_status = "good" if lean_avg <= 10 else "bad"
    vo_status = "good" if VO_MIN <= vo_avg <= VO_MAX else "bad"
    thigh_status = "good" if THIGH_MIN <= thigh_avg <= THIGH_MAX else "bad"

    feedback_data = generate_all_feedback(
        arm_status,
        knee_status,
        lean_status,
        vo_status,
        thigh_status,
        round(arm_avg, 1),
        round(knee_avg, 1),
        round(lean_avg, 1),
        round(vo_avg, 1),
        round(thigh_avg, 1),
        KNEE_MIN,
        KNEE_MAX,
        VO_MIN,
        VO_MAX,
        THIGH_MIN,
        THIGH_MAX
    )

    score = 100

    if arm_status == "bad":
        score -= 20

    if knee_status == "bad":
        score -= 20

    if lean_status == "bad":
        score -= 20

    if vo_status == "bad":
        score -= 20

    if thigh_status == "bad":
        score -= 20

    score = max(0, score)

    import datetime
    from storage_utils import upload_file_and_get_url

    user_id = "jinwook_test"
    record_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    full_video_url = upload_file_and_get_url(
        source,
        f"videos/{user_id}/{record_id}/full.mp4"
    )

    arm_video_url = upload_file_and_get_url(
        "arm_highlight.mp4",
        f"videos/{user_id}/{record_id}/arm.mp4"
    )

    knee_video_url = upload_file_and_get_url(
        "knee_highlight.mp4",
        f"videos/{user_id}/{record_id}/knee.mp4"
    )

    lean_video_url = upload_file_and_get_url(
        "lean_highlight.mp4",
        f"videos/{user_id}/{record_id}/lean.mp4"
    )

    vo_video_url = upload_file_and_get_url(
        "vo_highlight.mp4",
        f"videos/{user_id}/{record_id}/vo.mp4"
    )

    thigh_video_url = upload_file_and_get_url(
        "thigh_highlight.mp4",
        f"videos/{user_id}/{record_id}/thigh.mp4"
    )

    result = {
        "userId": user_id,
        "userName": "진욱",
        "speed": f"{SPEED}km/h",
        "totalScore": score,
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
            },
            "vo": {
                "average": round(vo_avg, 1),
                "status": "good" if VO_MIN <= vo_avg <= VO_MAX else "bad",
                "range": [VO_MIN, VO_MAX]
            },
            "thigh": {
                "average": round(thigh_avg, 1),
                "status": "good" if THIGH_MIN <= thigh_avg <= THIGH_MAX else "bad",
                "range": [THIGH_MIN, THIGH_MAX]
            }
        },

        "details": {
            "overallFeedback": feedback_data["overallFeedback"],
            "fullVideoLink": full_video_url,

            "parts": {
                "arm": {
                    "evaluation": feedback_data["parts"]["arm"]["evaluation"],
                    "videoLink": arm_video_url
                },
                "knee": {
                    "evaluation": feedback_data["parts"]["knee"]["evaluation"],
                    "videoLink": knee_video_url
                },
                "lean": {
                    "evaluation": feedback_data["parts"]["lean"]["evaluation"],
                    "videoLink": lean_video_url
                },
                "vo": {
                    "evaluation": feedback_data["parts"]["vo"]["evaluation"],
                    "videoLink": vo_video_url
                },
                "thigh": {
                    "evaluation": feedback_data["parts"]["thigh"]["evaluation"],
                    "videoLink": thigh_video_url
                }
            }
        }
    }

    return result
