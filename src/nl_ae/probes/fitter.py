"""Fit a single probe cell (one ``(label, layer)``) over fp32 activations (C07).

Tests inject a deterministic fake :class:`ProbeFitter`; the only shipped
implementation is :class:`SklearnLogisticFitter`, which lazy-imports
``sklearn.linear_model`` so the module surface stays free of sklearn at import
time.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .errors import FitFailedError, InsufficientLabelDataError
from .models import SklearnKwargs

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@dataclass(frozen=True)
class FitResult:
    """Output of one probe fit. ``train_proba`` / ``val_proba`` / ``test_proba``
    are shape ``(N,)`` for binary (positive-class probability) and ``(N, n_classes)``
    for multi-class (columns aligned with ``classes``)."""

    coef: np.ndarray
    intercept: np.ndarray
    classes: tuple[str, ...]
    train_pred: np.ndarray
    train_proba: np.ndarray
    val_pred: np.ndarray
    val_proba: np.ndarray
    test_pred: np.ndarray
    test_proba: np.ndarray
    standardize_mean: np.ndarray | None
    standardize_std: np.ndarray | None
    converged: bool
    n_iter: int


class ProbeFitter(Protocol):
    """Backend slice C07 depends on. Tests inject deterministic fakes."""

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_binary: bool,
        kwargs: SklearnKwargs,
        random_state: int,
    ) -> FitResult: ...


class SklearnLogisticFitter:
    """sklearn-backed :class:`ProbeFitter`. Imports sklearn lazily."""

    def fit_predict(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_binary: bool,
        kwargs: SklearnKwargs,
        random_state: int,
    ) -> FitResult:
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression

        if y_train.shape[0] < 2:
            raise InsufficientLabelDataError(
                f"train sub-fold has only {y_train.shape[0]} sample(s); need ≥2"
            )
        unique_train = np.unique(y_train)
        if unique_train.shape[0] < 2:
            raise InsufficientLabelDataError(
                f"train sub-fold has only {unique_train.shape[0]} distinct class(es); need ≥2"
            )

        if kwargs.standardize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train_use = scaler.fit_transform(X_train)
            X_val_use = scaler.transform(X_val) if X_val.shape[0] else X_val
            X_test_use = scaler.transform(X_test) if X_test.shape[0] else X_test
            mean: np.ndarray | None = np.asarray(scaler.mean_, dtype=np.float32)
            std: np.ndarray | None = np.asarray(scaler.scale_, dtype=np.float32)
        else:
            X_train_use, X_val_use, X_test_use = X_train, X_val, X_test
            mean = None
            std = None

        lr = LogisticRegression(
            penalty=kwargs.penalty,
            C=kwargs.C,
            solver=kwargs.solver,
            max_iter=kwargs.max_iter,
            fit_intercept=kwargs.fit_intercept,
            class_weight=None if kwargs.class_weight == "none" else kwargs.class_weight,
            random_state=random_state,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                lr.fit(X_train_use, y_train)
            except ValueError as exc:
                # E.g., NaNs in X, single-class y that slipped past our check.
                raise FitFailedError(f"LogisticRegression.fit raised: {exc}") from exc
            converged = not any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )
        n_iter = int(np.asarray(lr.n_iter_).max())

        classes = tuple(str(c) for c in lr.classes_)
        train_pred = lr.predict(X_train_use)
        val_pred = lr.predict(X_val_use) if X_val_use.shape[0] else np.empty((0,), dtype=object)
        test_pred = lr.predict(X_test_use) if X_test_use.shape[0] else np.empty((0,), dtype=object)

        if is_binary:
            if "1" not in classes:
                raise InsufficientLabelDataError(
                    f"binary train fold lacks positive class '1' in {classes!r}"
                )
            pos_idx = classes.index("1")
            train_proba = lr.predict_proba(X_train_use)[:, pos_idx]
            val_proba = (
                lr.predict_proba(X_val_use)[:, pos_idx]
                if X_val_use.shape[0]
                else np.empty((0,), dtype=np.float64)
            )
            test_proba = (
                lr.predict_proba(X_test_use)[:, pos_idx]
                if X_test_use.shape[0]
                else np.empty((0,), dtype=np.float64)
            )
            coef = np.asarray(lr.coef_, dtype=np.float32).squeeze(0)
        else:
            train_proba = lr.predict_proba(X_train_use)
            val_proba = (
                lr.predict_proba(X_val_use)
                if X_val_use.shape[0]
                else np.empty((0, len(classes)), dtype=np.float64)
            )
            test_proba = (
                lr.predict_proba(X_test_use)
                if X_test_use.shape[0]
                else np.empty((0, len(classes)), dtype=np.float64)
            )
            coef = np.asarray(lr.coef_, dtype=np.float32)

        intercept = np.asarray(lr.intercept_, dtype=np.float32)
        return FitResult(
            coef=coef,
            intercept=intercept,
            classes=classes,
            train_pred=train_pred,
            train_proba=train_proba,
            val_pred=val_pred,
            val_proba=val_proba,
            test_pred=test_pred,
            test_proba=test_proba,
            standardize_mean=mean,
            standardize_std=std,
            converged=converged,
            n_iter=n_iter,
        )


__all__ = ["FitResult", "ProbeFitter", "SklearnLogisticFitter"]
