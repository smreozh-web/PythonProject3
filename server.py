from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Server is running"

@app.route("/run", methods=["POST"])
def run():
    print("🔥 요청 들어옴")
    return jsonify({
        "result": "성공",
        "speed": 10
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
