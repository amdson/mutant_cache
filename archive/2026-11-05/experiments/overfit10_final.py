import sys
import mutant_cache.harness as H
_orig = H._maybe_best
def latest_on_tie(model, best, best_state, row):
    # keep the LAST checkpoint among ties at the best Spearman (here: the final one)
    if row.get("spearman_depth_all") == best["spearman_depth_all"]:
        best = dict(best, spearman_depth_all=best["spearman_depth_all"] - 1e-12)
    return _orig(model, best, best_state, row)
H._maybe_best = latest_on_tie
sys.exit(H.main(sys.argv[1:]))
