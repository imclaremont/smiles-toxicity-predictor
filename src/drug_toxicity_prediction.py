import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import logging
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 분자 특성 생성 함수 (수정된 버전)
def featurize(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            features.append({
                "MolWt": float(Descriptors.MolWt(mol)),
                "clogp": float(Descriptors.MolLogP(mol)),
                "tpsa": float(Descriptors.TPSA(mol)),
                "qed": float(Descriptors.qed(mol)),
                "NumHAcceptors": float(Descriptors.NumHAcceptors(mol)),
                "NumHDonors": float(Descriptors.NumHDonors(mol)),
                "RingCount": float(Descriptors.RingCount(mol)),
                "FractionCSP3": float(Descriptors.FractionCSP3(mol)),
                "HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
                "NumRotatableBonds": float(Descriptors.NumRotatableBonds(mol)),
                "NumAromaticRings": float(Descriptors.NumAromaticRings(mol)),
                "BalabanJ": float(Descriptors.BalabanJ(mol)),
                "BertzCT": float(Descriptors.BertzCT(mol))
            })
        else:
            features.append({
                "MolWt": 0.0, "clogp": 0.0, "tpsa": 0.0, "qed": 0.0, 
                "NumHAcceptors": 0.0, "NumHDonors": 0.0, "RingCount": 0.0,
                "FractionCSP3": 0.0, "HeavyAtomCount": 0.0, "NumRotatableBonds": 0.0,
                "NumAromaticRings": 0.0, "BalabanJ": 0.0, "BertzCT": 0.0
            })
    return pd.DataFrame(features)

# 데이터 로드
train = pd.read_csv("../data/raw/train.csv")
X = featurize(train["SMILES"])
y = train["label"].values  # numpy 배열로 변환

# 특성 선택 (상관관계 높은 특성 제거)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
X = X.drop(to_drop, axis=1)
logger.info(f"Dropped highly correlated features: {to_drop}")

# 데이터 분할 (특성 선택 후 수행)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# SMOTE 적용 (숫자형 데이터에만 적용)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 하이퍼파라미터 튜닝 (최적화된 버전)
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# 교차 검증 설정 (5-fold로 줄임)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # 50개 조합만 시도
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# 모델 학습
model.fit(X_res, y_res)
logger.info(f"Best parameters: {model.best_params_}")

# 검증 세트 평가
y_pred = model.predict(X_val)

logger.info("\nValidation Report:")
logger.info(classification_report(y_val, y_pred))
logger.info(f"F1 Score: {f1_score(y_val, y_pred, average='weighted'):.4f}")

# 예측 및 결과 저장
predict_input = pd.read_csv("../data/raw/predict_input.csv")
predict_features = featurize(predict_input["SMILES"])

# 학습 시와 동일한 피처 제거 적용
predict_features = predict_features.drop(['HeavyAtomCount', 'BertzCT'], axis=1, errors='ignore')

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