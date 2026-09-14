"""Inference-only CLI: headline + price -> action + probs + disclaimer.

Temporal note: price is display context only, NOT a trained feature
(slice-1 model uses TF-IDF title+body). Loads frozen artifacts
read-only; never writes data/ or models/.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib

from src.train import combine_text

DISCLAIMER = "DISCLAIMER: Educational demo only — not investment advice. No live trading."
LABELS = ["buy", "hold", "sell"]


def resolve_run_dir(model_arg):
    """Resolve --model: run dir, or a `latest` pointer file holding run_id."""
    p = Path(model_arg)
    if p.is_file() and p.name == "latest":
        run_id = p.read_text(encoding="utf-8").strip().split()[0]
        return p.parent / run_id
    return p


def load_artifacts(run_dir):
    """Load (vectorizer, model) read-only; raise FileNotFoundError naming path."""
    run_dir = Path(run_dir)
    vec_path = run_dir / "vectorizer.pkl"
    model_path = run_dir / "model.pkl"
    for path in (vec_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"model artifact missing: {path}")
    try:
        return joblib.load(vec_path), joblib.load(model_path)
    except Exception as exc:
        raise IOError(f"model artifact corrupt in {run_dir}: {exc}") from exc


def predict_probs(vectorizer, model, title, body=""):
    """Vectorize title+body, return {label: prob} covering buy/hold/sell."""
    text = combine_text(title, body)
    proba = model.predict_proba(vectorizer.transform([text]))[0]
    probs = {label: 0.0 for label in LABELS}
    for label, p in zip(model.classes_, proba):
        if label in probs:
            probs[label] = float(p)
    total = sum(probs.values()) or 1.0
    return {k: v / total for k, v in probs.items()}


def build_parser():
    """Arg parser; --help documents price as display context only."""
    parser = argparse.ArgumentParser(
        prog="btc-signal",
        description="Map a BTC news headline to buy/sell/hold (educational, not advice).",
    )
    parser.add_argument("--title", required=True, help="News headline text.")
    parser.add_argument("--body", default="",
                        help="Optional article body (with title forms the text feature).")
    parser.add_argument("--price", required=True,
                        help="Current BTC/EUR price: display context only, "
                             "not a trained feature in slice 1.")
    parser.add_argument("--model", default="models/latest",
                        help="Run dir or `latest` pointer (default: models/latest).")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of text.")
    return parser


def main(argv=None):
    """Entry point; returns exit code (0/2/3/1), never raises on bad input."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        if not args.title or not args.title.strip():
            print("error: --title must be a non-empty string\n", file=sys.stderr)
            parser.print_usage(sys.stderr)
            return 2
        try:
            price = float(args.price)
        except (TypeError, ValueError):
            print(f"error: --price must be numeric, got {args.price!r}\n",
                  file=sys.stderr)
            parser.print_usage(sys.stderr)
            return 2
        if not price > 0:
            print(f"error: --price must be positive, got {args.price!r}\n",
                  file=sys.stderr)
            parser.print_usage(sys.stderr)
            return 2
        try:
            run_dir = resolve_run_dir(args.model)
            vectorizer, model = load_artifacts(run_dir)
        except (FileNotFoundError, IOError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        probs = predict_probs(vectorizer, model, args.title, args.body)
        action = max(LABELS, key=lambda label: probs[label])
        if args.json:
            print(json.dumps({"action": action, "probs": probs,
                              "price_ctx": price, "price_currency": "EUR",
                              "disclaimer": DISCLAIMER}))
        else:
            print(f"action: {action}")
            print("probs: " + " ".join(f"{label}={probs[label]:.2f}" for label in LABELS))
            print(f"price_ctx: {price:.2f} EUR")
            print(DISCLAIMER)
        return 0
    except Exception as exc:  # defensive: contract exit 1
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
