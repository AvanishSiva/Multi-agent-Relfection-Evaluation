import argparse
import json

from src.significance import run_all_comparisons


def format_comparison(c: dict) -> str:
    flag = "SIGNIFICANT (p < 0.05)" if c["significant"] else "not significant"
    if c["test"] == "fisher_exact":
        return (
            f"{c['metric']:<24} {c['group_a']:<10} vs {c['group_b']:<10} "
            f"rate_a={c['rate_a']:.2f} rate_b={c['rate_b']:.2f} "
            f"p={c['p_value']:.4f}  -> {flag}"
        )
    return (
        f"{c['metric']:<24} {c['group_a']:<10} vs {c['group_b']:<10} "
        f"n_a={c['n_a']:<3} n_b={c['n_b']:<3} "
        f"median_a={c['median_a']:.2f} median_b={c['median_b']:.2f} "
        f"p={c['p_value']:.4f}  -> {flag}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run significance tests over a saved experiment sweep.")
    parser.add_argument("--input", type=str, required=True, help="Path to a results/experiment_*.json file")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the comparisons as JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    comparisons = run_all_comparisons(data["results"])

    print(f"=== Significance tests over {len(data['results'])} games ({args.input}) ===\n")
    for c in comparisons:
        print(format_comparison(c))

    n_significant = sum(1 for c in comparisons if c["significant"])
    print(f"\n{n_significant} / {len(comparisons)} comparisons significant at p < 0.05")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(comparisons, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
