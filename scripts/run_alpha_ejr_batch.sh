#!/bin/bash
# Run full alpha-EJR batch computation and update plots
# This script is meant to be run in tmux

set -e  # Exit on error
cd /home/ubuntu/multi-winner-bridging

echo "========================================"
echo "ALPHA-EJR BATCH PROCESSING"
echo "Started at: $(date)"
echo "========================================"

echo ""
echo "=== Step 1: Add alpha_EJR to all voting methods ==="
echo "This will process ~2028 committees across 338 elections"
echo "Estimated time: 60-90 minutes"
echo ""
uv run python scripts/add_alpha_ejr_to_results.py --workers 8

echo ""
echo "=== Step 2: Compute optimal alpha_EJR for best-possible ==="
echo "This will run full ILP optimization for 338 elections"
echo "Estimated time: 30-60 minutes"
echo ""
uv run python scripts/compute_optimal_alpha_ejr.py --workers 8

echo ""
echo "=== Step 3: Regenerate plots ==="
echo ""
uv run python scripts/plot_metrics_bar.py
uv run python scripts/plot_main_tradeoff_bar.py

echo ""
echo "========================================"
echo "ALL DONE!"
echo "Finished at: $(date)"
echo "========================================"
echo ""
echo "Updated files:"
echo "  - 338 voting_results.csv (added alpha_EJR column)"
echo "  - analysis/optimal_alpha_ejr.json"
echo "  - analysis/metrics_bar.png"
echo "  - analysis/main_tradeoff_bar.png"
