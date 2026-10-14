"""Simulate a large kidney-disease dataset and compare a baseline Random Forest
against a hyperparameter-tuned one.

The synthetic records are sampled from per-class statistics of the real
Kidney_data.csv (the same 8 features the Flask app uses), so they only reflect
what that small dataset shows. Results on simulated data say how tuning behaves
at scale, not how the model performs on real patients.

Usage:
    python simulate_and_tune.py
    python simulate_and_tune.py --rows 200000 --noise 0.05 --iters 20 --save
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

__author__ = "rkowdeed"

warnings.filterwarnings("ignore")

FEATURES = ['sg', 'htn', 'hemo', 'dm', 'al', 'appet', 'rc', 'pc']
CONTINUOUS = ['sg', 'hemo', 'rc']
DISCRETE = ['al']
BINARY = ['htn', 'dm', 'appet', 'pc']


def load_real_data():
    """Reuse the cleaning steps in model.py, stopping before the train/test split."""
    with open("model.py") as f:
        source = f.read().split("# Train Test Split:")[0]
    namespace = {}
    exec(source, namespace)
    data = namespace["dataset"][FEATURES + ["classification"]]
    return data.rename(columns={"classification": "target"})


def simulate(real, rows, overlap, noise, rng):
    """Sample `rows` records keeping the real class balance.

    overlap > 1 widens the continuous features so the classes overlap more;
    noise is the fraction of labels flipped at random.
    """
    priors = real["target"].value_counts(normalize=True)
    parts = []
    for label, prior in priors.items():
        cls = real[real["target"] == label]
        n = int(round(rows * prior))
        cov = np.cov(cls[CONTINUOUS].to_numpy().T) * overlap ** 2
        cont = rng.multivariate_normal(cls[CONTINUOUS].mean().to_numpy(), cov, size=n)
        part = pd.DataFrame(cont, columns=CONTINUOUS)
        for col in CONTINUOUS:
            part[col] = part[col].clip(real[col].min(), real[col].max())
        part["sg"] = (part["sg"] / 0.005).round() * 0.005
        part["rc"] = part["rc"].round(1)
        part["hemo"] = part["hemo"].round(1)
        for col in DISCRETE:
            part[col] = rng.choice(cls[col].to_numpy(), size=n)
        for col in BINARY:
            part[col] = (rng.random(n) < cls[col].mean()).astype(int)
        part["target"] = label
        parts.append(part)
    synthetic = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=0)
    flip = rng.random(len(synthetic)) < noise
    synthetic.loc[flip, "target"] = 1 - synthetic.loc[flip, "target"]
    return synthetic[FEATURES + ["target"]].reset_index(drop=True)


def report(name, model, X_test, y_test, real, seconds):
    synth_pred = model.predict(X_test)
    real_pred = model.predict(real[FEATURES])
    print(f"{name:<9} synthetic test acc={accuracy_score(y_test, synth_pred):.4f} "
          f"f1={f1_score(y_test, synth_pred):.4f} | "
          f"real-data acc={accuracy_score(real['target'], real_pred):.4f} | "
          f"fit time={seconds:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, default=100000, help="synthetic records to generate")
    parser.add_argument("--overlap", type=float, default=1.5, help="spread multiplier for continuous features")
    parser.add_argument("--noise", type=float, default=0.03, help="fraction of labels flipped")
    parser.add_argument("--iters", type=int, default=15, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv", type=int, default=3, help="cross-validation folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="write simulated_kidney_data.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    real = load_real_data()
    data = simulate(real, args.rows, args.overlap, args.noise, rng)
    print(f"Real rows: {len(real)} | simulated rows: {len(data)} | "
          f"overlap={args.overlap} noise={args.noise}")
    if args.save:
        data.to_csv("simulated_kidney_data.csv", index=False)
        print("Saved simulated_kidney_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data["target"], test_size=0.3, random_state=33,
        stratify=data["target"])

    start = time.time()
    baseline = RandomForestClassifier(random_state=args.seed, n_jobs=-1).fit(X_train, y_train)
    baseline_time = time.time() - start

    params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, 30, None],
        "min_samples_split": [2, 5, 10, 20, 50],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=args.seed, n_jobs=-1), params,
        n_iter=args.iters, cv=args.cv, random_state=args.seed)
    start = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start

    print("\nBest parameters:", search.best_params_)
    print(f"Best CV accuracy: {search.best_score_:.4f} (search took {search_time:.1f}s)\n")
    report("Baseline", baseline, X_test, y_test, real, baseline_time)
    report("Tuned", search.best_estimator_, X_test, y_test, real, search_time)


if __name__ == "__main__":
    main()
