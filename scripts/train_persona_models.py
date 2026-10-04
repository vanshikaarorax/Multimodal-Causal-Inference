from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from srcA.config import ARTIFACTS_DIR
from srcA.data import load_confirmed_personas, load_creatives
from srcA.embeddings import load_embeddings
from srcA.split import split_by_cluster, validate_no_cluster_leakage
from srcA.knn_model import train_knn_model, predict_personas as knn_predict
from srcA.classifier import train_classifier, predict_personas as classifier_predict
from srcA.evaluate import evaluate_per_persona

THRESHOLDS = np.arange(0.20, 0.71, 0.05)
K_VALUES = [3, 5, 7, 9, 11, 15]
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def parse_personas(value: str) -> list[str]:
    """Convert a persona string into a list of persona labels."""
    return [persona.strip() for persona in str(value).split("|") if persona.strip()]


def build_multilabel_matrix(labels: list[list[str]], persona_names: list[str]) -> np.ndarray:
    """Convert persona lists into a binary multi-label matrix."""
    persona_to_index = {persona: index for index, persona in enumerate(persona_names)}
    matrix = np.zeros((len(labels), len(persona_names)), dtype=int)
    for row_index, row in enumerate(labels):
        for persona in row:
            if persona in persona_to_index:
                matrix[row_index, persona_to_index[persona]] = 1
    return matrix


def evaluate_predictions(y_true: list[list[str]], y_pred: list[list[str]], persona_names: list[str], model_name: str, threshold: float, parameter: float) -> dict:
    """Evaluate multi-label persona predictions."""
    true_matrix = build_multilabel_matrix(y_true, persona_names)
    pred_matrix = build_multilabel_matrix(y_pred, persona_names)
    return {
        "model": model_name,
        "parameter": parameter,
        "threshold": threshold,
        "subset_accuracy": accuracy_score(true_matrix, pred_matrix),
        "macro_f1": f1_score(true_matrix, pred_matrix, average="macro", zero_division=0),
        "micro_f1": f1_score(true_matrix, pred_matrix, average="micro", zero_division=0),
        "weighted_f1": f1_score(true_matrix, pred_matrix, average="weighted", zero_division=0),
        "hamming_loss": hamming_loss(true_matrix, pred_matrix),
    }


def predictions_from_probabilities(probabilities: np.ndarray, persona_names: list[str], threshold: float) -> list[list[str]]:
    """Convert per-persona probabilities into multi-label predictions."""
    return [[persona_names[index] for index, score in enumerate(row) if score >= threshold] for row in probabilities]


def find_best_threshold(y_true: list[list[str]], probabilities: np.ndarray, persona_names: list[str], model_name: str, parameter: float) -> tuple[float, pd.DataFrame]:
    """Find the threshold with the highest development macro-F1."""
    results = []
    for threshold in THRESHOLDS:
        predictions = predictions_from_probabilities(probabilities, persona_names, float(threshold))
        results.append(evaluate_predictions(y_true, predictions, persona_names, model_name, float(threshold), parameter))
    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values(["macro_f1", "micro_f1"], ascending=False).iloc[0]
    return float(best_row["threshold"]), results_df


def main():
    creatives = load_creatives()
    personas = load_confirmed_personas()
    _, _, fused_embeddings, metadata = load_embeddings()
    clusters = pd.read_parquet(ARTIFACTS_DIR / "duplicate_clusters.parquet")

    labeled = personas[["creative_id", "personas"]].copy()
    labeled = labeled.merge(creatives[["creative_id"]], on="creative_id", how="inner", validate="one_to_one")
    labeled["persona_list"] = labeled["personas"].apply(parse_personas)

    split = split_by_cluster(labeled, clusters)
    validate_no_cluster_leakage(split.train, split.dev, split.test)

    embedding_index = {creative_id: index for index, creative_id in enumerate(metadata["creative_id"])}

    train_indices = [embedding_index[creative_id] for creative_id in split.train["creative_id"]]
    dev_indices = [embedding_index[creative_id] for creative_id in split.dev["creative_id"]]
    test_indices = [embedding_index[creative_id] for creative_id in split.test["creative_id"]]

    X_train = fused_embeddings[train_indices]
    X_dev = fused_embeddings[dev_indices]
    X_test = fused_embeddings[test_indices]

    y_train = split.train["persona_list"].tolist()
    y_dev = split.dev["persona_list"].tolist()
    y_test = split.test["persona_list"].tolist()

    persona_names = sorted({persona for row in y_train for persona in row})

    print(f"Train creatives: {len(X_train)}")
    print(f"Dev creatives: {len(X_dev)}")
    print(f"Test creatives: {len(X_test)}")
    print(f"Train clusters: {split.train['cluster_id'].nunique()}")
    print(f"Dev clusters: {split.dev['cluster_id'].nunique()}")
    print(f"Test clusters: {split.test['cluster_id'].nunique()}")
    print(f"Personas: {persona_names}")
    print(f"kNN k values: {K_VALUES}")
    print(f"LR C values: {C_VALUES}")
    print(f"Thresholds: {[round(float(t), 2) for t in THRESHOLDS]}")

    knn_search_results = []
    best_knn = None

    for k in K_VALUES:
        model = train_knn_model(X_train, y_train, n_neighbors=k)
        _, dev_probabilities = knn_predict(model, X_dev, threshold=0.0)

        best_threshold, threshold_results = find_best_threshold(
            y_dev, dev_probabilities, persona_names, "kNN", float(k)
        )

        best_row = threshold_results[threshold_results["threshold"] == best_threshold].iloc[0]
        knn_search_results.append(best_row.to_dict())

        if best_knn is None or (best_row["macro_f1"], best_row["micro_f1"]) > (best_knn["dev_macro_f1"], best_knn["dev_micro_f1"]):
            best_knn = {
                "k": k,
                "threshold": best_threshold,
                "dev_macro_f1": best_row["macro_f1"],
                "dev_micro_f1": best_row["micro_f1"],
            }

    classifier_search_results = []
    best_classifier = None

    for C in C_VALUES:
        model = train_classifier(X_train, y_train, C=C)
        _, dev_probabilities = classifier_predict(model, X_dev, threshold=0.0)

        best_threshold, threshold_results = find_best_threshold(
            y_dev, dev_probabilities, persona_names, "One-vs-Rest Logistic Regression", float(C)
        )

        best_row = threshold_results[threshold_results["threshold"] == best_threshold].iloc[0]
        classifier_search_results.append(best_row.to_dict())

        if best_classifier is None or (best_row["macro_f1"], best_row["micro_f1"]) > (best_classifier["dev_macro_f1"], best_classifier["dev_micro_f1"]):
            best_classifier = {
                "C": C,
                "threshold": best_threshold,
                "dev_macro_f1": best_row["macro_f1"],
                "dev_micro_f1": best_row["micro_f1"],
            }

    print("\nBest kNN configuration on DEV:")
    print(best_knn)

    print("\nBest Logistic Regression configuration on DEV:")
    print(best_classifier)

    best_knn_model = train_knn_model(X_train, y_train, n_neighbors=best_knn["k"])

    _, knn_test_probabilities = knn_predict(
        best_knn_model,
        X_test,
        threshold=0.0,
    )

    knn_test_predictions = predictions_from_probabilities(
        knn_test_probabilities,
        persona_names,
        best_knn["threshold"],
    )

    best_classifier_model = train_classifier(
        X_train,
        y_train,
        C=best_classifier["C"],
    )

    _, classifier_test_probabilities = classifier_predict(
        best_classifier_model,
        X_test,
        threshold=0.0,
    )

    classifier_test_predictions = predictions_from_probabilities(
        classifier_test_probabilities,
        persona_names,
        best_classifier["threshold"],
    )

    test_results = pd.DataFrame([
        evaluate_predictions(
            y_test,
            knn_test_predictions,
            persona_names,
            "kNN",
            best_knn["threshold"],
            best_knn["k"],
        ),
        evaluate_predictions(
            y_test,
            classifier_test_predictions,
            persona_names,
            "One-vs-Rest Logistic Regression",
            best_classifier["threshold"],
            best_classifier["C"],
        ),
    ])

    # NEW: per-persona metrics for the final LR model.
    per_persona_results = evaluate_per_persona(
        y_test,
        classifier_test_predictions,
        persona_names,
    )

    print("\nPer-persona TEST results:")
    print(per_persona_results.to_string(index=False))

    knn_search_df = pd.DataFrame(knn_search_results)
    classifier_search_df = pd.DataFrame(classifier_search_results)

    print("\nAll kNN configurations selected by DEV:")
    print(knn_search_df.to_string(index=False))

    print("\nAll Logistic Regression configurations selected by DEV:")
    print(classifier_search_df.to_string(index=False))

    print("\nFinal TEST results using DEV-selected configurations:")
    print(test_results.to_string(index=False))

    knn_output_path = ARTIFACTS_DIR / "knn_hyperparameter_search.csv"
    classifier_output_path = ARTIFACTS_DIR / "classifier_hyperparameter_search.csv"
    test_output_path = ARTIFACTS_DIR / "persona_model_comparison.csv"
    per_persona_output_path = ARTIFACTS_DIR / "per_persona_metrics.csv"

    knn_search_df.to_csv(knn_output_path, index=False)
    classifier_search_df.to_csv(classifier_output_path, index=False)
    test_results.to_csv(test_output_path, index=False)
    per_persona_results.to_csv(per_persona_output_path, index=False)

    print(f"\nSaved kNN search: {knn_output_path}")
    print(f"Saved Logistic Regression search: {classifier_output_path}")
    print(f"Saved final test results: {test_output_path}")
    print(f"Saved per-persona metrics: {per_persona_output_path}")


if __name__ == "__main__":
    main()