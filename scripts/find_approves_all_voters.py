"""
Find the proportion of elections with at least one voter who approves all projects
that have non-zero approvals (i.e., all projects that anyone voted for).
"""

from pathlib import Path
from src.pb_data_loader import parse_pb_file


def main():
    # Get all .pb files
    data_dir = Path(__file__).parent.parent / "data"
    pb_files = list(data_dir.rglob("*.pb"))
    print(f"Found {len(pb_files)} .pb files")

    elections_with_approves_all = 0
    total_elections = 0
    elections_list = []

    for pb_file in pb_files:
        data = parse_pb_file(str(pb_file))
        n_projects = len(data["projects"])
        if n_projects == 0:
            continue

        total_elections += 1

        # Find projects with non-zero approvals
        project_approval_counts = {pid: 0 for pid in data["projects"]}
        for voter_id, approved in data["votes"].items():
            for pid in approved:
                if pid in project_approval_counts:
                    project_approval_counts[pid] += 1

        projects_with_approvals = {
            pid for pid, count in project_approval_counts.items() if count > 0
        }
        n_approved_projects = len(projects_with_approvals)

        if n_approved_projects == 0:
            continue

        # Check if any voter approves all projects with non-zero approvals
        has_approves_all = False
        for voter_id, approved in data["votes"].items():
            approved_set = set(approved)
            if projects_with_approvals.issubset(approved_set):
                has_approves_all = True
                break

        if has_approves_all:
            elections_with_approves_all += 1
            elections_list.append(pb_file.name)

    print(f"\nTotal elections: {total_elections}")
    print(
        f"Elections with a voter approving all non-zero-approval projects: {elections_with_approves_all}"
    )
    print(f"Proportion: {elections_with_approves_all / total_elections:.4f}")
    print(f"\nElections with approve-all voters:")
    for e in elections_list[:10]:
        print(f"  - {e}")
    if len(elections_list) > 10:
        print(f"  ... and {len(elections_list) - 10} more")


if __name__ == "__main__":
    main()
