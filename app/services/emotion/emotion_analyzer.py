from ultralytics import YOLO  # YOLO 모델 사용
from collections import defaultdict  # 감정 카운트 누적을 위한 defaultdict

class EmotionCoreModel:
    def __init__(self, model_path="best.pt"):
        # 무거운 YOLO 분류/검출 모델 로드를 1회만 수행
        self.model = YOLO(model_path)  # 모델 가중치 로드

    def infer_once(self, frame):
        # 단일 프레임 예측 수행
        return self.model.predict(source=frame, save=False, conf=0.5, verbose=False)  # 프레임 추론 결과 반환

class EmotionSessionState:
    def __init__(self):
        # 질문(WebSocket 연결) 구간 동안의 감정 누적 상태
        self.class_names = ['긍정', '중립', '부정', '긴장']  # 클래스 명시
        self.emotion_counter = defaultdict(int)  # 감정별 카운터
        self.total_frames = 0  # 감정이 탐지된 총 프레임 수

    def update(self, results):
        # YOLO 결과를 받아 감정 카운터 누적
        boxes = results[0].boxes  # 첫 결과의 바운딩 박스
        if boxes is None:
            return  # 탐지 없으면 종료
        for box in boxes:
            class_id = int(box.cls[0])  # 클래스 인덱스
            emotion = self.class_names[class_id]  # 감정 라벨
            self.emotion_counter[emotion] += 1  # 해당 감정 증가
            self.total_frames += 1  # 총 카운트 증가

    def current_top(self):
        # 현재까지 가장 많이 나온 감정 반환
        if not self.emotion_counter:
            return "분석중"  # 데이터 없으면 분석중 표시
        return max(self.emotion_counter.items(), key=lambda x: x[1])[0]  # 최빈 감정 반환

    def summary(self) -> dict:
        # 감정 비율과 점수 요약 반환
        if self.total_frames == 0:
            # 데이터 없으면 0으로 채움
            return {k: 0 for k in self.class_names} | {"best": "없음", "score": 0}  # 기본 요약
        # 비율 계산
        rate = {k: 0 for k in self.class_names}  # 초기화
        for k, v in self.emotion_counter.items():
            rate[k] = round((v / self.total_frames) * 100)  # 백분율 반올림
        # 총합 보정(반올림으로 인한 편차)
        total = sum(rate.values())  # 합계
        if total != 100:
            best = max(rate, key=rate.get)  # 최빈 감정
            rate[best] += (100 - total)  # 최빈 감정에 차이 보정
        # 점수 계산
        rate["best"] = max(self.emotion_counter, key=self.emotion_counter.get)  # 최빈 감정 라벨
        rate["score"] = self._score(rate)  # 감정 점수
        return rate  # 요약 반환

    def _score(self, emotion_distribution: dict) -> int:
        # 감정 분포에 기반한 점수화
        weights = {'긍정': 1.2, '중립': 0.8, '긴장': -0.5, '부정': -1.0}  # 감정 가중치
        raw = sum(emotion_distribution.get(k, 0) * w for k, w in weights.items())  # 가중 합
        return max(0, min(100, round(raw)))  # 0~100 범위로 클램프

class EmotionAnalyzer:
    def __init__(self, core_model: EmotionCoreModel):
        # 전역으로 재사용될 무거운 코어 모델 주입
        self.core = core_model  # 코어 참조

    def analyze_frame(self, frame, state: EmotionSessionState):
        # 한 프레임 감정 추론 수행 후 상태에 누적
        results = self.core.infer_once(frame)  # YOLO 추론 결과
        state.update(results)  # 세션 상태 업데이트
        return state.current_top()  # 현재 최빈 감정 즉시 반환(옵션)
