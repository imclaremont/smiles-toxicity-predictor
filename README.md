# 약물 독성 예측 프로젝트

## 프로젝트 구조
```
├── data/            # 데이터 파일
├── models/          # 학습된 모델
├── src/             # 소스 코드
├── README.md        # 이 파일
├── requirements.txt # 의존성 패키지
```

## 설치 방법
1. 저장소 복제
2. 필요한 패키지 설치
```
pip install -r requirements.txt
```

## 사용 방법
예측 실행
```
python drug_toxicity_prediction.py
```

결과는 `data/processed/predict_output_하동헌.csv`에 저장됩니다.

<br>

## 결과
<strong>[모델 성능 지표]<strong/>
- 정확도: 70%
- 클래스별 정밀도(Precision), 재현율(Recall), F1 점수 확인 가능
