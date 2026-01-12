"""
Compute the overall proportion of budget-feasible committees covered by some voter,
aggregated across ALL elections (not per-election average).

A committee is "covered" if there exists a voter whose approval set is a superset of it.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pb_data_loader import parse_pb_file, enumerate_valid_committees


def main():
    data_dir = Path(__file__).parent.parent / "data"
    pb_files = list(data_dir.rglob("*.pb"))

    print(f"Found {len(pb_files)} .pb files")

    total_committees = 0
    total_covered = 0

    for pb_file in pb_files:
        try:
            data = parse_pb_file(str(pb_file))
            n_projects = len(data["projects"])
            if n_projects == 0 or n_projects > 13:
                continue

            project_ids = list(data["projects"].keys())
            project_costs = [data["projects"][pid]["cost"] for pid in project_ids]
            budget_raw = data["meta"].get("budget", 0)
            budget = int(float(budget_raw)) if isinstance(budget_raw, (int, float, str)) else 0

            project_to_idx = {pid: idx for idx, pid in enumerate(project_ids)}

            voter_sets = []
            for voter_id, approved_projects in data["votes"].items():
                approved_indices = frozenset(
                    project_to_idx[pid] for pid in approved_projects if pid in project_to_idx
                )
                voter_sets.append(approved_indices)

            committees = enumerate_valid_committees(project_costs, budget, show_progress=False)

            covered = 0
            for committee in committees:
                committee_set = frozenset(committee)
                if any(committee_set <= voter_set for voter_set in voter_sets):
                    covered += 1

            total_committees += len(committees)
            total_covered += covered
        except Exception as e:
            print(f"Warning: Failed to process {pb_file.name}: {e}")

    print()
    print("=" * 60)
    print("OVERALL COVERAGE (aggregated across all elections)")
    print("=" * 60)
    print(f"Total committees across all elections: {total_committees:,}")
    print(f"Total covered committees:              {total_covered:,}")
    print(f"Overall proportion covered:            {total_covered / total_committees:.4f} ({total_covered / total_committees * 100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
