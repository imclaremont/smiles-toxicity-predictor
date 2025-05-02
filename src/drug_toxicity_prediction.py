import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import logging
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 분자 특성 생성 함수 (확장된 버전)
def featurize(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            features.append({
                "MolWt": Descriptors.MolWt(mol),
                "clogp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "qed": Descriptors.qed(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "RingCount": Descriptors.RingCount(mol),
                "FractionCSP3": Descriptors.FractionCSP3(mol)
            })
        else:
            features.append({
                "MolWt": 0, "clogp": 0, "tpsa": 0, "qed": 0,
                "NumHAcceptors": 0, "NumHDonors": 0, "RingCount": 0, "FractionCSP3": 0
            })
    return pd.DataFrame(features)

# 데이터 로드
train = pd.read_csv("../data/raw/train.csv")
X = featurize(train["SMILES"])
y = train["label"]

# 클래스 불균형 처리 (SMOTE)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# 교차 검증 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1
)

# 모델 학습
model.fit(X_res, y_res)
logger.info(f"Best parameters: {model.best_params_}")

# 검증 세트 평가
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_pred = model.predict(X_val)

logger.info("\nValidation Report:")
logger.info(classification_report(y_val, y_pred))
logger.info(f"F1 Score: {f1_score(y_val, y_pred, average='weighted'):.4f}")

# 예측 및 결과 저장
predict_input = pd.read_csv("../data/raw/predict_input.csv")
predict_features = featurize(predict_input["SMILES"])
predictions = model.predict(predict_features)

# 결과 저장 (SMILES와 output만 저장)
output_df = pd.DataFrame({
    'SMILES': predict_input['SMILES'],
    'output': predictions
})
output_df.to_csv("../data/processed/predict_output_하동헌.csv", index=False)
logger.info(f"Prediction results saved to ../data/processed/predict_output_하동헌.csv")

# 모델 저장
joblib.dump(model, "../models/drug_toxicity_model.pkl")