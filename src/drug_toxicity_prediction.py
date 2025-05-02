import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import logging
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def featurize(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # 기본 분자 서술자
            desc = {
                "MolWt": float(Descriptors.MolWt(mol)),
                "clogp": float(Descriptors.MolLogP(mol)),
                "tpsa": float(Descriptors.TPSA(mol)),
                "qed": float(Descriptors.qed(mol)),
                "NumHAcceptors": float(Descriptors.NumHAcceptors(mol)),
                "NumHDonors": float(Descriptors.NumHDonors(mol)),
                "RingCount": float(Descriptors.RingCount(mol)),
                "FractionCSP3": float(Descriptors.FractionCSP3(mol))
            }
            
            # Morgan 지문 추가 (2048비트)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            desc.update({f"fp_{i}": int(b) for i, b in enumerate(fp)})
            
            features.append(desc)
        else:
            # 유효하지 않은 SMILES 처리
            default_features = {k: 0.0 for k in desc.keys()}
            default_features.update({f"fp_{i}": 0 for i in range(2048)})
            features.append(default_features)
    
    return pd.DataFrame(features)

# 데이터 로드 및 전처리
train = pd.read_csv("../data/raw/train.csv")
X = featurize(train["SMILES"])
y = train["label"].values

# 파이프라인 설정 (SMOTE + XGBoost)
model = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    ))
])

# 하이퍼파라미터 튜닝 공간
params = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 6, 9],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__subsample': [0.6, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0],
    'xgb__gamma': [0, 0.1, 0.2]
}

# 교차 검증 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    model, params, 
    n_iter=50,
    scoring='f1_weighted',
    cv=cv,
    verbose=3,
    random_state=42,
    n_jobs=-1
)

# 모델 학습
search.fit(X, y)
best_model = search.best_estimator_

# 검증 세트 평가
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_pred = best_model.predict(X_val)

logger.info("\nValidation Report:")
logger.info(classification_report(y_val, y_pred))
logger.info(f"F1 Score: {f1_score(y_val, y_pred, average='weighted'):.4f}")

# 예측 및 결과 저장
predict_input = pd.read_csv("../data/raw/predict_input.csv")
predict_features = featurize(predict_input["SMILES"])
predictions = best_model.predict(predict_features)

output_df = pd.DataFrame({
    'SMILES': predict_input['SMILES'],
    'output': predictions
})
output_df.to_csv("../data/processed/predict_output_하동헌.csv", index=False)
logger.info("Improved prediction results saved")

# 모델 저장
joblib.dump(best_model, "../models/toxicity_model_xgb.pkl")