# =========================
# 무릎 각도 기준
# =========================
KNEE_MIN = 150
KNEE_MAX = 176


# =========================
# 팔 스윙 기준
# =========================
ARM_FWD_MIN = 0.0
ARM_FWD_MAX = 0.14
ARM_BWD_MIN = -0.2
ARM_BWD_MAX = -0.04


# =========================
# 수직 진폭 기준
# =========================
def get_vo_range(speed):
    if speed == 8:
        return 6, 11
    elif speed == 10:
        return 6, 10
    elif speed == 12:
        return 5, 9


# =========================
# 무릎 높이 기준 (허벅지 각도)
# =========================
def get_thigh_range(speed):
    if speed == 8:
        return 35, 50
    elif speed == 10:
        return 40, 55
    elif speed == 12:
        return 45, 60
