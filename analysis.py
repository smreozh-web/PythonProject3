from ultralytics import YOLO
import cv2
import numpy as np
import imageio_ffmpeg as ffmpeg
import subprocess
import os
from metrics import *
from openai import OpenAI
client = OpenAI(api_key="키")



def generate_all_feedback(
    arm, knee, lean, vo, thigh,
    arm_avg, knee_avg, lean_avg, vo_avg, thigh_avg,
    knee_min, knee_max, vo_min, vo_max, thigh_min, thigh_max
):
    prompt = f"""
너는 러닝 자세를 분석해주는 전문 코치이자 스포츠 손상 예방 전문가야.

다음 분석 결과를 바탕으로
1) 전체 총평(overallFeedback)
2) 관절별 평가(parts.arm, parts.knee, parts.lean, parts.vo, parts.thigh)

를 JSON으로 만들어줘.

[판정 기준]
- good: 평균값이 권장 범위 안에 들어온 경우
- bad: 평균값이 권장 범위를 벗어난 경우
- 반드시 이 기준만 사용해서 판단해

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
- 각 관절은 반드시 아래 규칙을 따를 것
  1) good이면:
     - evaluation: 현재 상태를 1~2문장으로 요약
     - expectedBenefit: 이 자세를 유지했을 때 기대할 수 있는 효과를 1문장
     - injuryRisk: ""
  2) bad이면:
     - evaluation: 어떤 점이 범위를 벗어났는지 1~2문장으로 설명
     - injuryRisk: 잘못된 자세가 반복될 때 생길 수 있는 대표적 부상/통증 위험을 1문장
     - expectedBenefit: 교정했을 때 얻을 수 있는 효과를 1문장
- 부상명은 과장하지 말고 러닝 자세와 관련된 흔한 위험만 설명
- 자연스럽고 이해하기 쉬운 코치 말투로 작성
- 반드시 아래 JSON 형식으로만 답변해

JSON 형식:
{{
  "overallFeedback": "문장",
  "parts": {{
    "arm": {{
      "evaluation": "문장",
      "injuryRisk": "문장 또는 빈 문자열",
      "expectedBenefit": "문장"
    }},
    "knee": {{
      "evaluation": "문장",
      "injuryRisk": "문장 또는 빈 문자열",
      "expectedBenefit": "문장"
    }},
    "lean": {{
      "evaluation": "문장",
      "injuryRisk": "문장 또는 빈 문자열",
      "expectedBenefit": "문장"
    }},
    "vo": {{
      "evaluation": "문장",
      "injuryRisk": "문장 또는 빈 문자열",
      "expectedBenefit": "문장"
    }},
    "thigh": {{
      "evaluation": "문장",
      "injuryRisk": "문장 또는 빈 문자열",
      "expectedBenefit": "문장"
    }}
  }}
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
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

def expand_box(x1, y1, x2, y2, frame_w, frame_h, pad=40):
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame_w, x2 + pad)
    y2 = min(frame_h, y2 + pad)
    return int(x1), int(y1), int(x2), int(y2)

def make_fixed_clip(frames, center_idx, box, output_path, fps=8, duration_sec=4, output_size=(480, 640)):
    if center_idx is None or box is None or not frames:
        return

    total_needed = int(fps * duration_sec)   # 예: 8fps * 4초 = 32프레임
    half = total_needed // 2

    start = max(0, center_idx - half)
    end = min(len(frames), start + total_needed)

    # 뒤가 부족하면 앞쪽으로 보정
    start = max(0, end - total_needed)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    for i in range(start, end):
        crop = crop_box_and_resize(frames[i], box, output_size)
        writer.write(crop)

    writer.release()

def get_box_from_points(points, frame_w, frame_h, pad=40, min_size=140):
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]

    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))

    w = max(max(xs) - min(xs), min_size)
    h = max(max(ys) - min(ys), min_size)

    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2

    return expand_box(x1, y1, x2, y2, frame_w, frame_h, pad=pad)

def crop_box_and_resize(frame, box, output_size=(480, 640)):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return cv2.resize(frame, output_size)

    return cv2.resize(crop, output_size)

def get_part_boxes(frame, shoulder, elbow, wrist, hip, knee, ankle):
    h, w = frame.shape[:2]

    return {
        "arm": get_box_from_points([shoulder, elbow, wrist], w, h, pad=50, min_size=180),
        "knee": get_box_from_points([hip, knee, ankle], w, h, pad=60, min_size=220),
        "lean": get_box_from_points([shoulder, hip, knee], w, h, pad=70, min_size=260),
        "vo": get_box_from_points([shoulder, hip, knee, ankle], w, h, pad=80, min_size=320),
        "thigh": get_box_from_points([hip, knee], w, h, pad=60, min_size=180),
    }


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

    all_frames = []

    hip_y_buffer = []
    VO_BUFFER_SIZE = BUFFER_SIZE
    vo_value_buffer = []
    VO_VALUE_BUFFER_SIZE = BUFFER_SIZE

    best_clip=[];best_box=None;tail_frames=0;max_lean_error=-1
    best_knee_clip=[];best_knee_box=None;knee_tail_frames=0;max_knee_error=-1

    worst_arm_error = -1
    worst_knee_error = -1
    worst_lean_error = -1
    worst_vo_error = -1
    worst_thigh_error = -1

    worst_arm_idx = None
    worst_knee_idx = None
    worst_lean_idx = None
    worst_vo_idx = None
    worst_thigh_idx = None

    worst_arm_box = None
    worst_knee_box = None
    worst_lean_box = None
    worst_vo_box = None
    worst_thigh_box = None

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    highlight_fps = 8
    highlight_size = (480, 640)

    full_writer = None

    print("fps:", fps, "highlight_fps:", highlight_fps, "highlight_size:", highlight_size)


    results=model(source,stream=True)

    for r in results:

        frame = r.plot()

        if frame is None or frame.size == 0:
            continue

        all_frames.append(frame.copy())

        # 🔥 writer 생성 (한 번만)
        if full_writer is None:
            h, w = frame.shape[:2]
            full_writer = cv2.VideoWriter(
                "full_with_skeleton_raw.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                highlight_fps,  # 🔥 속도 핵심
                (w, h)
            )

        # 🔥 저장
        full_writer.write(frame)

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

                part_boxes = get_part_boxes(frame, shoulder, elbow, wrist, hip, knee, ankle)

                current_idx = len(all_frames) - 1

                # arm error
                if arm_swing_val > 7:
                    arm_error = arm_swing_val - 7
                elif arm_swing_val < -10:
                    arm_error = -10 - arm_swing_val
                else:
                    arm_error = 0

                if arm_error > worst_arm_error:
                    worst_arm_error = arm_error
                    worst_arm_idx = current_idx
                    worst_arm_box = part_boxes["arm"]

                # knee error
                if knee_angle < KNEE_MIN:
                    knee_error = KNEE_MIN - knee_angle
                elif knee_angle > KNEE_MAX:
                    knee_error = knee_angle - KNEE_MAX
                else:
                    knee_error = 0

                if knee_error > worst_knee_error:
                    worst_knee_error = knee_error
                    worst_knee_idx = current_idx
                    worst_knee_box = part_boxes["knee"]

                # lean error
                lean_error = max(0, lean - 10)
                if lean_error > worst_lean_error:
                    worst_lean_error = lean_error
                    worst_lean_idx = current_idx
                    worst_lean_box = part_boxes["lean"]

                # vo error
                if vo_avg < VO_MIN:
                    vo_error = VO_MIN - vo_avg
                elif vo_avg > VO_MAX:
                    vo_error = vo_avg - VO_MAX
                else:
                    vo_error = 0

                if vo_error > worst_vo_error:
                    worst_vo_error = vo_error
                    worst_vo_idx = current_idx
                    worst_vo_box = part_boxes["vo"]

                # thigh error
                if knee_lift < THIGH_MIN:
                    thigh_error = THIGH_MIN - knee_lift
                elif knee_lift > THIGH_MAX:
                    thigh_error = knee_lift - THIGH_MAX
                else:
                    thigh_error = 0

                if thigh_error > worst_thigh_error:
                    worst_thigh_error = thigh_error
                    worst_thigh_idx = current_idx
                    worst_thigh_box = part_boxes["thigh"]



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

    if full_writer:
        full_writer.release()

    make_fixed_clip(all_frames, worst_arm_idx, worst_arm_box, "arm_highlight_raw.mp4", fps=highlight_fps,
                    duration_sec=4, output_size=highlight_size)
    make_fixed_clip(all_frames, worst_knee_idx, worst_knee_box, "knee_highlight_raw.mp4", fps=highlight_fps,
                    duration_sec=4, output_size=highlight_size)
    make_fixed_clip(all_frames, worst_lean_idx, worst_lean_box, "lean_highlight_raw.mp4", fps=highlight_fps,
                    duration_sec=4, output_size=highlight_size)
    make_fixed_clip(all_frames, worst_vo_idx, worst_vo_box, "vo_highlight_raw.mp4", fps=highlight_fps, duration_sec=4,
                    output_size=highlight_size)
    make_fixed_clip(all_frames, worst_thigh_idx, worst_thigh_box, "thigh_highlight_raw.mp4", fps=highlight_fps,
                    duration_sec=4, output_size=highlight_size)

    convert_to_h264("full_with_skeleton_raw.mp4", "full_with_skeleton.mp4")
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
    #if max_lean_error > 0: play_slow(best_clip, best_box, "Worst Lean")
    #if max_knee_error > 0: play_slow(best_knee_clip, best_knee_box, "Worst Knee")

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
        "full_with_skeleton.mp4",
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
                    "injuryRisk": feedback_data["parts"]["arm"]["injuryRisk"],
                    "expectedBenefit": feedback_data["parts"]["arm"]["expectedBenefit"],
                    "videoLink": arm_video_url
                },
                "knee": {
                    "evaluation": feedback_data["parts"]["knee"]["evaluation"],
                    "injuryRisk": feedback_data["parts"]["knee"]["injuryRisk"],
                    "expectedBenefit": feedback_data["parts"]["knee"]["expectedBenefit"],
                    "videoLink": knee_video_url
                },
                "lean": {
                    "evaluation": feedback_data["parts"]["lean"]["evaluation"],
                    "injuryRisk": feedback_data["parts"]["lean"]["injuryRisk"],
                    "expectedBenefit": feedback_data["parts"]["lean"]["expectedBenefit"],
                    "videoLink": lean_video_url
                },
                "vo": {
                    "evaluation": feedback_data["parts"]["vo"]["evaluation"],
                    "injuryRisk": feedback_data["parts"]["vo"]["injuryRisk"],
                    "expectedBenefit": feedback_data["parts"]["vo"]["expectedBenefit"],
                    "videoLink": vo_video_url
                },
                "thigh": {
                    "evaluation": feedback_data["parts"]["thigh"]["evaluation"],
                    "injuryRisk": feedback_data["parts"]["thigh"]["injuryRisk"],
                    "expectedBenefit": feedback_data["parts"]["thigh"]["expectedBenefit"],
                    "videoLink": thigh_video_url
                }
            }
        }
    }

    return result
