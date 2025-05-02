import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, Crippen
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import logging
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhanced_featurize(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # 기본 분자 서술자
            desc = {
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Crippen.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "HBA": Lipinski.NumHAcceptors(mol),
                "HBD": Lipinski.NumHDonors(mol),
                "RotatableBonds": Lipinski.NumRotatableBonds(mol),
                "AromaticRings": Lipinski.NumAromaticRings(mol),
                "FractionCSP3": Lipinski.FractionCSP3(mol),
                "RingCount": Lipinski.RingCount(mol)
            }
            
            # 추가 물리화학적 특성
            desc.update({
                "BalabanJ": Descriptors.BalabanJ(mol),
                "BertzCT": Descriptors.BertzCT(mol),
                "MolMR": Crippen.MolMR(mol)
            })
            
            # 지문 추가 (2048비트)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            desc.update({f"fp_{i}": int(b) for i, b in enumerate(fp)})
            
            features.append(desc)
        else:
            features.append({k:0 for k in desc.keys()})
    
    return pd.DataFrame(features)

# 데이터 로드
train = pd.read_csv("../data/raw/train.csv")
X = enhanced_featurize(train["SMILES"])
y = train["label"].values

# 베이스 모델 정의
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05
)

lgbm = LGBMClassifier(
    objective='binary',
    random_state=42,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1
)

# 스태킹 모델 구성
estimators = [
    ('xgb', xgb),
    ('lgbm', lgbm)
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.01
    ),
    cv=5
)

# 파이프라인 설정
model = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('stack', stacking_model)
])

# 모델 학습
model.fit(X, y)

# 검증 세트 평가
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_pred = model.predict(X_val)

logger.info(f"\nEnhanced Model F1 Score: {f1_score(y_val, y_pred, average='weighted'):.4f}")

# 예측 및 결과 저장
predict_input = pd.read_csv("../data/raw/predict_input.csv")
predict_features = enhanced_featurize(predict_input["SMILES"])
predictions = model.predict(predict_features)

output_df = pd.DataFrame({
    'SMILES': predict_input['SMILES'],
    'output': predictions
})
output_df.to_csv("../data/processed/predict_output_하동헌.csv", index=False)
logger.info("Enhanced prediction results saved")

# 모델 저장
joblib.dump(model, "../models/drug_toxicity_model.pkl")