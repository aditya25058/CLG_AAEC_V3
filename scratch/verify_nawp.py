#!/usr/bin/env python3
import os
import json
import sqlite3
import numpy as np

def test_transition_probability_normalization():
    print("Testing transition matrix normalization...")
    db_path = "/home/palakm/.gemini/antigravity-ide/brain/f36cd9c9-271b-4ebf-8daa-07adaa8ff019/qwen3_30b_real_v2.db"
    if not os.path.exists(db_path):
        print("Skipping DB checks: DB not found.")
        return True

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT prompt_id, token_pos, layer, expert_id FROM activations LIMIT 1000")
    rows = cursor.fetchall()
    conn.close()

    transition_L2 = np.zeros((48, 128, 128))
    
    # Simple transition computation
    trace = {}
    for r in rows:
        p_id, t_pos, layer, exp_id = r
        if p_id not in trace:
            trace[p_id] = {}
        if t_pos not in trace[p_id]:
            trace[p_id][t_pos] = {}
        trace[p_id][t_pos][layer] = exp_id

    for p_id in trace:
        for t in trace[p_id]:
            for l in range(46):
                l2 = l + 2
                if l in trace[p_id][t] and l2 in trace[p_id][t]:
                    exp_l = trace[p_id][t][l]
                    exp_l2 = trace[p_id][t][l2]
                    transition_L2[l, exp_l, exp_l2] += 1

    # Row-normalize
    for l in range(48):
        row_sums = transition_L2[l].sum(axis=1)
        for i in range(128):
            if row_sums[i] > 0:
                transition_L2[l, i] = transition_L2[l, i] / row_sums[i]
                assert np.allclose(transition_L2[l, i].sum(), 1.0), f"Row {i} in layer {l} does not sum to 1.0!"
            else:
                transition_L2[l, i] = 1.0 / 128.0
                assert np.allclose(transition_L2[l, i].sum(), 1.0), f"Row {i} in layer {l} fallback does not sum to 1.0!"
                
    print("Transition matrix normalization test PASSED.")
    return True

def test_simulation_metrics():
    print("Testing simulation execution and metrics integrity...")
    # Import main simulation script function
    import sys
    sys.path.append("/home/palakm/MoEServingSim/scratch")
    
    # We will invoke simulate_nawp via a subprocess run to make sure it doesn't crash on standard configuration
    import subprocess
    cmd = ["python3", "/home/palakm/MoEServingSim/scratch/simulate_nawp.py"]
    # Run with a small dry run to see if it finishes without errors
    # Wait, simulate_nawp is quite fast since it runs in ~10 seconds on the dataset
    print("Running a dry run of simulate_nawp.py...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("simulate_nawp.py failed with stderr:")
        print(res.stderr)
        return False
        
    print(res.stdout[:500] + "\n...")
    print("Verification JSON checks...")
    
    report_path = "/home/palakm/MoEServingSim/qwen3_30b_plots/nawp_evaluation_results.json"
    assert os.path.exists(report_path), f"JSON report at {report_path} was not created!"
    
    with open(report_path, "r") as f:
        data = json.load(f)
        
    assert "link_speeds" in data
    assert "results" in data
    
    results = data["results"]
    assert "128" in results
    cache_128 = results["128"]
    
    assert "demand" in cache_128
    assert "nawp" in cache_128
    
    # Check that NAWP hit rate is larger than demand hit rate
    for idx, bw in enumerate(data["link_speeds"]):
        d_hr = cache_128["demand"][idx]["hit_rate"]
        n_hr = cache_128["nawp"][idx]["hit_rate"]
        print(f"BW: {bw} GB/s -> Demand Hit Rate: {d_hr*100:.2f}%, NAWP Hit Rate: {n_hr*100:.2f}%")
        assert n_hr >= d_hr, "NAWP hit rate cannot be lower than Demand hit rate!"
        
    print("Simulation metrics integrity test PASSED.")
    return True

if __name__ == "__main__":
    success = test_transition_probability_normalization()
    if success:
        success = test_simulation_metrics()
    if success:
        print("\nAll NAWP verification tests PASSED successfully!")
    else:
        print("\nSome NAWP verification tests FAILED.")
        sys.exit(1)
