import firebase_admin
from firebase_admin import credentials, firestore, storage

# 1. JSON 키 파일 경로
cred = credentials.Certificate("running-form-coach-firebase-adminsdk-fbsvc-83dcdc795b.json")

# 2. Firebase 초기화 (스토리지 버킷 필수)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'running-form-coach.firebasestorage.app'
})

# 3. DB & 스토리지 객체 생성
db = firestore.client()
bucket = storage.bucket()

print("🔥 파이어베이스 AI 연동 완료!")