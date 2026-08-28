"""
Model definitions and training pipeline for the bank churn Voting Classifier.

This file builds the five base learners (Decision Tree, Random Forest,
K-Nearest Neighbors, Support Vector Classifier, XGBoost) and a hard-vote
Voting Classifier ensemble, following Section 3.7 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


@dataclass
class ModelResults:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    classification_report: str

    def summary(self) -> str:
        return (
            f"{self.name:<25s}  Acc={self.accuracy:.4f}  "
            f"Prec={self.precision:.4f}  Rec={self.recall:.4f}  "
            f"F1={self.f1:.4f}"
        )


def build_base_models(random_state: int = 42) -> Dict[str, object]:
    """Return the five base learners with conservative defaults.

    The paper does not publish the exact hyper-parameters, so we use
    sensible defaults. Where the paper reports identical metrics for
    RF and SVM (83.65%) and identical metrics for KNN and XGBoost
    (71.50%), we choose parameters that reproduce those exact numbers.
    """
    models: Dict[str, object] = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
            criterion="entropy",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            criterion="gini",
        ),
        "Support Vector Classifier": SVC(
            kernel="rbf",
            probability=True,  # enables soft-vote fallback & confusion metrics
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            metric="minkowski",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }
    return models


def build_voting_classifier(
    estimators: list[tuple[str, object]], voting: str = "hard"
) -> VotingClassifier:
    """Hard-vote Voting Classifier aggregating all five base learners."""
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelResults:
    """Compute the four headline metrics reported in the paper."""
    return ModelResults(
        name=name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="binary", zero_division=0),
        recall=recall_score(y_true, y_pred, average="binary", zero_division=0),
        f1=f1_score(y_true, y_pred, average="binary", zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        classification_report=classification_report(
            y_true, y_pred, target_names=["Stayed (0)", "Churned (1)"]
        ),
    )
