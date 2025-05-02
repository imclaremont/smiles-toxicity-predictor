import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging

# 0. 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. SMILES → 분자 특성 생성 함수
def featurize(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            features.append({
                "MolWt": Descriptors.MolWt(mol),
                "clogp": Descriptors.MolLogP(mol),
                "sa_score": Descriptors.TPSA(mol), # 대신 TPSA 사용
                "qed": Descriptors.qed(mol)
            })
        else:
            features.append({
                "MolWt": 0, "clogp": 0, "sa_score": 0, "qed": 0
            })
    return pd.DataFrame(features)

# 2. 학습 데이터 로드
train = pd.read_csv("../data/raw/train.csv")
X = featurize(train["SMILES"])
y = train["label"]

# 3. 학습/검증 분할 및 모델 학습
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 4. 검증 결과 출력
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# 5. 예측 데이터 로드 및 전처리
predict_input = pd.read_csv("../data/raw/predict_input.csv")
X_pred = featurize(predict_input["SMILES"])
predicted = model.predict(X_pred)

# 6. 결과 저장
predict_output = predict_input.copy()
predict_output["output"] = predicted
predict_output.to_csv("../data/processed/predict_output_하동헌.csv", index=False)
logger.info("Prediction results saved to ../data/processed/predict_output_하동헌.csv")

# 7. 모델 저장
import os
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/drug_toxicity_model.pkl")
logger.info("Model saved to ../models/drug_toxicity_model.pkl")