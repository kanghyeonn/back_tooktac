import cv2  # OpenCV 사용을 위한 모듈 임포트
import mediapipe as mp  # MediaPipe 포즈/얼굴 랜드마크 분석을 위한 모듈 임포트
import numpy as np  # 수치 계산을 위한 NumPy 임포트
from typing import Dict  # 타입 힌트를 위해 Dict 임포트

class PostureCoreModel:
    def __init__(self):
        # MediaPipe의 솔루션 핸들 준비
        self.mp_face_mesh = mp.solutions.face_mesh  # FaceMesh 클래스 네임스페이스
        self.mp_pose = mp.solutions.pose  # Pose 클래스 네임스페이스
        # 무거운 초기화: 모델/그래프 로드는 한번만 수행하기 위함
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)  # 얼굴 랜드마크 모델 로드
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # 포즈 모델 로드

        # 얼굴 및 포즈에서 사용할 인덱스(상수) 지정
        self.LEFT_EYE_OUTER = 33  # 왼쪽 눈 바깥쪽 랜드마크 인덱스
        self.LEFT_IRIS_LEFT = 471  # 왼쪽 홍채 좌측
        self.LEFT_IRIS_RIGHT = 469  # 왼쪽 홍채 우측
        self.LEFT_EYE_CENTER = 468  # 왼쪽 눈 중심
        self.RIGHT_EYE_OUTER = 263  # 오른쪽 눈 바깥쪽
        self.RIGHT_IRIS_LEFT = 476  # 오른쪽 홍채 좌측
        self.RIGHT_IRIS_RIGHT = 474  # 오른쪽 홍채 우측
        self.RIGHT_EYE_CENTER = 473  # 오른쪽 눈 중심
        self.LEFT_EAR = self.mp_pose.PoseLandmark.LEFT_EAR  # 왼쪽 귀 포즈 랜드마크
        self.RIGHT_EAR = self.mp_pose.PoseLandmark.RIGHT_EAR  # 오른쪽 귀 포즈 랜드마크
        self.LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER  # 왼쪽 어깨
        self.RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER  # 오른쪽 어깨
        self.LEFT_INDEX = self.mp_pose.PoseLandmark.LEFT_INDEX  # 왼손 집게손가락 끝
        self.RIGHT_INDEX = self.mp_pose.PoseLandmark.RIGHT_INDEX  # 오른손 집게손가락 끝

        # solvePnP용 3D 모델 포인트(고정값)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),       # 코 끝
            (0.0, -63.6, -12.5),   # 턱
            (-43.3, 32.7, -26.0),  # 왼쪽 눈 구석
            (43.3, 32.7, -26.0),   # 오른쪽 눈 구석
            (-28.9, -28.9, -24.1), # 왼쪽 입꼬리
            (28.9, -28.9, -24.1)   # 오른쪽 입꼬리
        ], dtype=np.float32)  # float32 배열로 고정

    def get_camera_matrix(self, w: int, h: int) -> np.ndarray:
        # 카메라 내부 파라미터 행렬 생성 (초점거리=focal_length=w 가정)
        focal_length = w  # 초점거리로 프레임 폭 사용
        center = (w / 2, h / 2)  # 영상 중심
        return np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)  # 카메라 행렬 반환

    def infer_once(self, frame: np.ndarray) -> Dict[str, object]:
        # 한 프레임을 받아 시선/피치/회전/어깨/손 등 즉시 결과만 계산
        frame = cv2.flip(frame, 1)  # 좌우 반전으로 거울 효과
        h, w = frame.shape[:2]  # 프레임 높이/너비 획득
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe는 RGB 입력을 기대하므로 변환

        face_result = self.face_mesh.process(rgb)  # 얼굴 랜드마크 추론
        pose_result = self.pose.process(rgb)  # 포즈 랜드마크 추론

        # 기본값 초기화
        lm = None  # 얼굴 랜드마크 배열
        gaze_h = "CENTER"  # 수평 시선 기본값
        pitch_dir = "CENTER"  # 상하 고개 방향 기본값
        turn_dir = "CENTER"  # 좌우 회전 기본값
        shoulder_dir = "CENTER"  # 어깨 기울기 기본값
        hand_dir = "NONE"  # 손 등장 기본값

        if face_result.multi_face_landmarks:
            lm = face_result.multi_face_landmarks[0].landmark  # 첫 얼굴의 랜드마크 사용

            # 왼쪽 눈의 수평 시선 지표 계산
            left_eye_outer = np.array([lm[self.LEFT_EYE_OUTER].x * w, lm[self.LEFT_EYE_OUTER].y * h])
            left_iris_left = np.array([lm[self.LEFT_IRIS_LEFT].x * w, lm[self.LEFT_IRIS_LEFT].y * h])
            left_iris_right = np.array([lm[self.LEFT_IRIS_RIGHT].x * w, lm[self.LEFT_IRIS_RIGHT].y * h])
            left_ratio = np.linalg.norm(left_iris_left - left_eye_outer) / (np.linalg.norm(left_iris_right - left_iris_left) + 1e-6)

            # 오른쪽 눈의 수평 시선 지표 계산
            right_eye_outer = np.array([lm[self.RIGHT_EYE_OUTER].x * w, lm[self.RIGHT_EYE_OUTER].y * h])
            right_iris_left = np.array([lm[self.RIGHT_IRIS_LEFT].x * w, lm[self.RIGHT_IRIS_LEFT].y * h])
            right_iris_right = np.array([lm[self.RIGHT_IRIS_RIGHT].x * w, lm[self.RIGHT_IRIS_RIGHT].y * h])
            right_ratio = np.linalg.norm(right_eye_outer - right_iris_right) / (np.linalg.norm(right_iris_right - right_iris_left) + 1e-6)

            # 간단한 임계치 기반 수평 시선 판정
            gaze_h = "LEFT" if left_ratio < 0.42 else "RIGHT" if right_ratio < 0.40 else "CENTER"

            # PnP로 피치(상하) 추정
            image_points = np.array([[lm[i].x * w, lm[i].y * h] for i in [1, 152, 33, 263, 78, 308]], dtype=np.float32)
            cam_mtx = self.get_camera_matrix(w, h)  # 카메라 행렬 생성
            success, rvec, _ = cv2.solvePnP(self.model_points, image_points, cam_mtx, np.zeros((4, 1)))  # PnP 해
            if success:
                rmat, _ = cv2.Rodrigues(rvec)  # 회전 벡터를 회전 행렬로 변환
                pitch = np.degrees(np.arcsin(-rmat[2][1]))  # 피치 각 추출
                pitch_dir = "UP" if pitch < -12 else "DOWN" if pitch > 9 else "CENTER"  # 임계치로 라벨링

        if pose_result.pose_landmarks:
            plm = pose_result.pose_landmarks.landmark  # 포즈 랜드마크 배열
            if lm is not None:
                # 좌우 회전: 눈 중심과 귀의 x 거리 비교
                le, re = lm[self.LEFT_EYE_CENTER], lm[self.RIGHT_EYE_CENTER]  # 눈 중심 좌표
                le2e = abs(le.x - plm[self.LEFT_EAR].x)  # 왼눈-왼귀 x거리
                re2e = abs(re.x - plm[self.RIGHT_EAR].x)  # 오른눈-오른귀 x거리
                turn_dir = "LEFT" if le2e > re2e + 0.035 else "RIGHT" if re2e > le2e + 0.055 else "CENTER"  # 임계치 판정

            # 어깨 기울기: 좌우 어깨 y 차이를 이용
            diff = plm[self.LEFT_SHOULDER].y - plm[self.RIGHT_SHOULDER].y  # y 차이
            shoulder_dir = "LEFT UP" if diff > 0.04 else "RIGHT UP" if diff < -0.04 else "CENTER"  # 임계치 판정

            # 손 등장 여부: 두 손의 가시성 중 하나라도 높으면 등장으로 판단
            hand_dir = "VISIBLE" if plm[self.LEFT_INDEX].visibility > 0.5 or plm[self.RIGHT_INDEX].visibility > 0.5 else "NONE"

        # 프레임별 즉시 피드백용 boolean 묶음 반환
        return {
            "gaze_h": gaze_h,  # 수평 시선 라벨
            "pitch_dir": pitch_dir,  # 상하 라벨
            "turn_dir": turn_dir,  # 좌우 회전 라벨
            "shoulder_dir": shoulder_dir,  # 어깨 라벨
            "hand_dir": hand_dir,  # 손 라벨
            "ok_gaze": (gaze_h == "CENTER"),  # 정면 응시 여부
            "ok_pitch": (pitch_dir == "CENTER"),  # 상하 정면 여부
            "ok_head": (turn_dir == "CENTER"),  # 좌우 정면 여부
            "ok_shoulder": (shoulder_dir == "CENTER"),  # 어깨 수평 여부
            "ok_hand": (hand_dir == "NONE"),  # 손 비등장 여부
        }

class PostureSessionState:
    def __init__(self):
        # 이 질문(WebSocket 연결) 동안만 유지할 누적값
        self.total_frames = 0  # 총 프레임 수
        self.center_frames = 0  # 완전 정면 프레임 수
        self.shoulder_warning_count = 0  # 어깨 경고 횟수
        self.hand_warning_count = 0  # 손 경고 횟수
        self.shoulder_prev_state = "CENTER"  # 이전 어깨 상태
        self.hand_prev_state = "NONE"  # 이전 손 상태

    def update(self, step: Dict[str, object]):
        # 프레임별 결과를 받아 누적값 갱신
        self.total_frames += 1  # 프레임 수 증가
        if step["ok_gaze"] and step["ok_pitch"] and step["ok_head"]:
            self.center_frames += 1  # 완전 정면이면 카운트 증가

        # 어깨 경고: CENTER에서 CENTER가 아닌 상태로 변할 때 카운트
        if step["shoulder_dir"] != self.shoulder_prev_state and self.shoulder_prev_state == "CENTER":
            self.shoulder_warning_count += 1  # 경고 증가
        self.shoulder_prev_state = step["shoulder_dir"]  # 이전 상태 갱신

        # 손 경고: NONE에서 VISIBLE로 변할 때 카운트
        if step["hand_dir"] != self.hand_prev_state and self.hand_prev_state == "NONE":
            self.hand_warning_count += 1  # 경고 증가
        self.hand_prev_state = step["hand_dir"]  # 이전 상태 갱신

    def finalize(self) -> Dict[str, int]:
        # 누적 결과로 최종 점수 산출
        gaze_score = int((self.center_frames / self.total_frames) * 100) if self.total_frames > 0 else 0  # 정면 응시율 점수
        shoulder_score = max(0, 50 - 5 * self.shoulder_warning_count)  # 어깨 경고 기반 점수
        hand_score = max(0, 50 - 5 * self.hand_warning_count)  # 손 경고 기반 점수
        total_score = gaze_score + shoulder_score + hand_score - 100  # 합산 후 0~100 스케일 유사화

        return {
            "gaze_rate_score": gaze_score,  # 정면 응시 점수
            "shoulder_posture_warning_count": self.shoulder_warning_count,  # 어깨 경고 수
            "hand_posture_warning_count": self.hand_warning_count,  # 손 경고 수
            "shoulder_hand_score": shoulder_score + hand_score,  # 어깨+손 합산 점수
            "video_score": total_score  # 최종 비디오 점수
        }

class PostureAnalyzer:
    def __init__(self, core_model: PostureCoreModel):
        # 전역으로 재사용될 무거운 코어 모델 주입
        self.core = core_model  # 코어 참조 저장

    def analyze_frame(self, frame: np.ndarray, state: PostureSessionState) -> Dict[str, bool]:
        # 한 프레임 추론 수행
        step = self.core.infer_once(frame)  # 코어로부터 프레임별 결과 획득
        state.update(step)  # 질문 상태에 누적
        # 즉시 피드백용 boolean만 반환
        return {
            "gaze": step["ok_gaze"],  # 정면 응시 여부
            "pitch": step["ok_pitch"],  # 상하 정면 여부
            "head": step["ok_head"],  # 좌우 정면 여부
            "shoulder": step["ok_shoulder"],  # 어깨 수평 여부
            "hand": step["ok_hand"],  # 손 비등장 여부
        }
