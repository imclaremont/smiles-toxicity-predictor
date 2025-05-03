# 약물 독성 예측 프로젝트

## EDA
https://claremont.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D-%EC%8B%A0%EC%95%BD-%EA%B0%9C%EB%B0%9C%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%99%94%ED%95%A9%EB%AC%BC-%EB%8F%85%EC%84%B1-%EC%98%88%EC%B8%A1-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8EDA

<br>

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
- 리더보드 점수: 0.824
- "SK C&C + SKALA 1기" 289개 제출 중 48등
- 클래스별 정밀도(Precision), 재현율(Recall), F1 점수 확인 가능
