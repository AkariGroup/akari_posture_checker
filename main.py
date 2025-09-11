#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DepthAI + MediaPipe Pose で姿勢チェックを行う完全版スクリプト（関数化版・修正済み）
機能1. 左右自動判定で鼻‐肩‐腰・膝‐腰‐肩の角度を計算
2. 猫背を 3 回検知すると siren_long.wav をループ再生し続ける ★EDIT
3. 立ち姿勢へ変化した瞬間だけサウンドを再生（reset.wav）
4. 1時間以上連続着座すると「Stretch Time!」を画面に表示して siren_long.wav をループ再生
5. 人物が検出されない間はタイマーとカウンタをリセット
6. 右下に座りタイマー（HH:MM:SS）を常時表示"""

# ─────────────────────────────────────────────────────────────
#  インポート
# ─────────────────────────────────────────────────────────────
import os
import sys
import math
import time
import cv2
import mediapipe as mp
import depthai as dai
import absl.logging
from playsound import playsound
from akari_client import AkariClient
import threading

# ─────────────────────────────────────────────────────────────
#  MediaPipe Pose モジュール
# ─────────────────────────────────────────────────────────────
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

# ─────────────────────────────────────────────────────────────
#  定数
# ─────────────────────────────────────────────────────────────
POSTURE_THRESHOLD   = 130         # 猫背判定（鼻‐肩‐腰）
STAND_THRESHOLD     = 170         # 立ち姿勢判定（膝‐腰‐肩）
MAX_COUNTER         =   3         # 猫背許容回数
DETECTION_COOLDOWN  =   5.0       # 猫背サウンドのクールダウン [s]
VIS_THRESH          =   0.5       # ランドマーク visibility しきい値
SIT_LIMIT_SEC       = 3600          # 1時間以上の連続着座でストレッチ促し

# ─────────────────────────────────────────────────────────────
#  ループ音再生用関数
# ─────────────────────────────────────────────────────────────
def play_ping_loop(state):
    """ siren_long.wav をループ再生するためのスレッド関数 """
    while state['ping_active']:
        playsound("sound/siren_long.wav")

# ─────────────────────────────────────────────────────────────
#  ユーティリティ関数群
# ─────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    """ 3点の座標から角度を計算する """
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) -
        math.atan2(a[1]-b[1], a[0]-b[0])
    )
    return abs(ang if ang < 180 else 360-ang)

def get_angle(i, j, k, lms, v=VIS_THRESH):
    """ ランドマークのインデックスから角度を取得する """
    try:
        la, lb, lc = lms[i], lms[j], lms[k]
    except IndexError:
        return 0.0
    if min(la.visibility, lb.visibility, lc.visibility) < v:
        return 0.0
    return calculate_angle([la.x,la.y], [lb.x,lb.y], [lc.x,lc.y])

def choose_side(lms, v=VIS_THRESH):
    """ 可視性や深度から、体のどちら側を向いているかを判定する """
    Ls, Rs = lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value],  lms[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    Lh, Rh = lms[mp_pose.PoseLandmark.LEFT_HIP.value],        lms[mp_pose.PoseLandmark.RIGHT_HIP.value]
    vis_left  = Ls.visibility + Lh.visibility
    vis_right = Rs.visibility + Rh.visibility
    if vis_left  >= v*2 and vis_left  > vis_right: return 'left'
    if vis_right >= v*2 and vis_right > vis_left: return 'right'
    if Ls.z < Rs.z: return 'left'
    if Rs.z < Ls.z: return 'right'
    nose = lms[mp_pose.PoseLandmark.NOSE.value]
    dL = (nose.x-Ls.x)**2 + (nose.y-Ls.y)**2
    dR = (nose.x-Rs.x)**2 + (nose.y-Rs.y)**2
    return 'left' if dL < dR else 'right'

def draw_text(img, text, pos, col=(255,255,255), bg=(0,0,0)):
    """ 背景付きテキストを描画する """
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3
    sz = cv2.getTextSize(text, font, scale, th)[0]; x, y = pos
    cv2.rectangle(img, (x-10, y-sz[1]-10), (x+sz[0]+10, y+10), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, col, th)

def hms(sec):
    """ 秒数を HH:MM:SS 形式の文字列に変換する """
    return f"{int(sec//3600):02d}:{int(sec%3600//60):02d}:{int(sec%60):02d}"

# ─────────────────────────────────────────────────────────────
#  初期化関数群
# ─────────────────────────────────────────────────────────────
def initialize_akari():
    """ Akari ロボットを初期化し、接続する """
    print("🤖  Akari ロボットと接続しています ...")
    akari = AkariClient()
    joints = akari.joints
    joints.enable_all_servo()
    print("🌀  カメラの頭部を初期位置へ（pan=0, tilt=0）")
    joints.move_joint_positions(sync=True, pan=0, tilt=0)
    return akari

def suppress_logs():
    """ TensorFlow と MediaPipe の冗長なログ出力を抑制する """
    absl.logging.set_verbosity(absl.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, sys.stderr.fileno())
    except Exception:
        pass

def setup_depthai_pipeline():
    """ DepthAI のパイプラインを構築する """
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)
    return pipeline

# ─────────────────────────────────────────────────────────────
#  状態管理・更新関数
# ─────────────────────────────────────────────────────────────
def manage_sound_loop(state, should_be_active):
    """ サウンドループの状態を管理する（開始・停止） """
    if should_be_active and not state['ping_active']:
        state['ping_active'] = True
        state['ping_thread'] = threading.Thread(target=play_ping_loop, args=(state,), daemon=True)
        state['ping_thread'].start()
    elif not should_be_active and state['ping_active']:
        state['ping_active'] = False
        if state['ping_thread']:
            state['ping_thread'].join()
            state['ping_thread'] = None
    return state

def process_landmarks(img, pose_landmarks, state): # <- 変更点: 引数を `lms` から `pose_landmarks` に変更
    """ 検出されたランドマークを処理し、姿勢を判定・描画する """
    lms = pose_landmarks.landmark # <- 変更点: 描画で使う親オブジェクトから座標リストを抽出

    now = time.time()
    side = choose_side(lms)

    # 角度計算
    sit_angle_points = (
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value if side=='right' else mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value if side=='right' else mp_pose.PoseLandmark.LEFT_HIP.value,
    )
    stand_angle_points = (
        mp_pose.PoseLandmark.RIGHT_KNEE.value if side=='right' else mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value if side=='right' else mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value if side=='right' else mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    )
    sit = get_angle(*sit_angle_points, lms)
    std = get_angle(*stand_angle_points, lms)
    if std > 180:
        std = 360 - std

    # 姿勢判定
    is_stand = std >= STAND_THRESHOLD
    is_sit = sit != 0 and not is_stand

    # 着座タイマー
    if is_sit and state['sitting_start_time'] is None:
        state['sitting_start_time'], state['stretch_prompted'] = now, False
    if is_stand:
        state['sitting_start_time'], state['stretch_prompted'] = None, False

    # 立ち姿勢への遷移を検出
    if is_stand and not state['prev_is_standing']:
        playsound("sound/reset.wav")
        state['stand_up_time'] = now
        state['bad_posture_count'] = 0

    # 猫背判定
    if state['stand_up_time'] and now - state['stand_up_time'] < 10:
        draw_text(img, "Posture Check Paused", (10,70), (255,255,255), (90,90,90))
    else:
        if sit < POSTURE_THRESHOLD and sit != 0:
            state['consecutive_bad_posture'] += 1
            if state['consecutive_bad_posture'] >= 3 and now - state['last_detection_time'] >= DETECTION_COOLDOWN:
                state['bad_posture_count'] += 1
                state['last_detection_time'] = now
                playsound("sound/siren_short.wav")
                state['consecutive_bad_posture'] = 0
            draw_text(img, f"Bad Posture! count:{state['bad_posture_count']}", (10,70), (255,255,255), (0,0,200))
        else:
            state['consecutive_bad_posture'] = 0
            if is_stand:
                draw_text(img, "Standing!", (10,70), (255,255,255), (150,0,0))
            else:
                draw_text(img, f"Good Posture! count:{state['bad_posture_count']}", (10,70), (255,255,255), (0,150,0))

    # Stretch Time 判定
    is_stretch_time = is_sit and state['sitting_start_time'] and now - state['sitting_start_time'] >= SIT_LIMIT_SEC
    if is_stretch_time:
        if not state['stretch_prompted']:
            print("🧘  1時間以上座り続けています。ストレッチしましょう！")
            state['stretch_prompted'] = True
        draw_text(img, "Stretch Time!", (10,110), (0,0,0), (0,255,255))
    # bad_posture 3 回超過処理 ★EDIT
    is_bad_posture_exceeded = state['bad_posture_count'] >= MAX_COUNTER
    if is_bad_posture_exceeded:
        if not state['ping_active']:
            print("⚠️  猫背が 3 回連続で検出されました。正しい姿勢に戻してください！")
        draw_text(img, "    PLEASE STRETCH    ", (10,110), (0,0,0), (0,0,255))

    # サウンドループの制御
    should_ping = is_stretch_time or is_bad_posture_exceeded
    state = manage_sound_loop(state, should_ping)
    
    # 角度とランドマーク描画
    draw_text(img, f"{side[0].upper()}-deg1:{'N/A' if sit==0 else int(sit)}", (400,70), (20,20,20), (200,200,200) if sit==0 else (250,250,250))
    draw_text(img, f"{side[0].upper()}-deg2:{'N/A' if std==0 else int(std)}", (400,110), (20,20,20), (200,200,200) if std==0 else (250,250,250))
    mp_drawing.draw_landmarks(img, pose_landmarks, mp_pose.POSE_CONNECTIONS) # <- 変更点: `res.pose_landmarks` を `pose_landmarks` に変更
    
    state['prev_is_standing'] = is_stand
    return state

def handle_no_detection(img, state):
    """ 人物が検出されなかった場合の処理 """
    draw_text(img, "No detection", (10,70), (200,200,200), (150,0,0))
    # 状態をリセット
    state['sitting_start_time'] = None
    state['stretch_prompted'] = False
    state['consecutive_bad_posture'] = 0
    state['prev_is_standing'] = False
    # サウンドループを停止
    state = manage_sound_loop(state, False)
    return state

# ─────────────────────────────────────────────────────────────
#  メイン関数
# ─────────────────────────────────────────────────────────────
def main():
    """ メイン処理 """
    # 初期化
    # initialize_akari()
    suppress_logs()
    pipeline = setup_depthai_pipeline()

    # 状態変数を辞書で管理
    state = {
        'bad_posture_count': 0,
        'consecutive_bad_posture': 0,
        'last_detection_time': 0.0,
        'sitting_start_time': None,
        'stretch_prompted': False,
        'prev_is_standing': False,
        'stand_up_time': None,
        'ping_thread': None,
        'ping_active': False
    }

    print("📷  DepthAI カメラ起動中。'q' で終了")
    
    with dai.Device(pipeline) as device, mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)

        while True:
            in_rgb = rgb_queue.tryGet()
            if in_rgb is None:
                cv2.waitKey(1)
                continue

            frame = in_rgb.getCvFrame()
            if frame is None:
                continue
                
            # MediaPipeでの処理
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgRGB.flags.writeable = False
            results = pose.process(imgRGB)
            imgRGB.flags.writeable = True
            img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

            # 姿勢ランドマークが検出されたかどうかで処理を分岐
            if results.pose_landmarks:
                state = process_landmarks(img, results.pose_landmarks, state) # <- 変更点: `.landmark` を付けずにオブジェクト全体を渡す
            else:
                state = handle_no_detection(img, state)

            # タイマーを常時表示
            h, w, _ = img.shape
            if state['sitting_start_time']:
                elapsed = time.time() - state['sitting_start_time']
                draw_text(img, f"Sitting {hms(elapsed)}", (w-270, h-20), (0,0,0), (255,255,255))
            else:
                draw_text(img, "Standing 00:00:00", (w-270, h-20), (0,0,0), (255,255,255))
            
            # 画面表示と終了処理
            cv2.imshow("Posture Checker", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 終了処理
    state = manage_sound_loop(state, False)
    cv2.destroyAllWindows()
    print("プログラム終了")

if __name__ == '__main__':
    main()