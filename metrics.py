import numpy as np

def angle(a,b,c):
    a,b,c=np.array(a),np.array(b),np.array(c)
    ba=a-b
    bc=c-b
    cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

def thigh_angle(shoulder, hip, knee):
    body = np.array(shoulder) - np.array(hip)
    thigh = np.array(knee) - np.array(hip)
    cos = np.dot(body, thigh) / (np.linalg.norm(body) * np.linalg.norm(thigh) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def body_lean_angle(shoulder,hip):
    body=np.array(shoulder)-np.array(hip)
    vertical=np.array([0,-1])
    cos=np.dot(body,vertical)/(np.linalg.norm(body)+1e-6)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

def shoulder_tilt_angle(ls,rs):
    dx=rs[0]-ls[0]
    dy=rs[1]-ls[1]
    return abs(np.degrees(np.arctan2(dy,dx)))

def point_to_line_distance(p, a, b):
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    ap = p - a
    ab = b - a
    distance = np.abs(np.cross(ab, ap)) / (np.linalg.norm(ab) + 1e-6)
    return distance

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

