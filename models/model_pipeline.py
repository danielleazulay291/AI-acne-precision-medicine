
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_pipeline():
    return Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("feature_selection", SelectKBest(score_func=f_classif, k=100)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="saga",
            l1_ratio=0.5,
            C=0.1,
            max_iter=5000,
            random_state=42
        ))
    ])
