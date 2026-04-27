from flask import Flask, request, jsonify
from analysis import run_analysis
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Server is running"

@app.route("/run", methods=["POST"])
def run():
    data = request.json

    video_url = data.get("videoUrl")
    speed = data.get("speed")

    print("🔥 요청 받음:", data)

    # ❗ 지금은 테스트용 (나중에 다운로드 로직 추가해야 함)
    local_path = "test.mp4"

    try:
        result = run_analysis(local_path, speed)
        return jsonify(result)

    except Exception as e:
        print("❌ 에러:", e)
        return jsonify({"error": str(e)}), 500


# 🔥 Render용 실행 코드 (중요)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)