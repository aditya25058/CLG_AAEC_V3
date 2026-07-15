# evaluation/utils/trace_collector.py
import os
import sqlite3
import json
from typing import Dict, Tuple, Set

def load_real_traces_from_db(db_path: str) -> Tuple[Dict, Dict]:
    """
    Loads real hardware traces from the SQLite database.
    Splits prompts 50/50 into calibration and evaluation sets.
    Returns (calibration_db, evaluation_db) where each is structured as:
      { prompt_id: { token_pos: { layer: (expert_id, active_set, k50) } } }
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
        
    print(f"Loading traces from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prompt_id, token_pos, layer, expert_id, active_indices, energy_k_50 
        FROM activations 
        ORDER BY prompt_id, token_pos, layer
    """)
    rows = cursor.fetchall()
    conn.close()
    
    prompt_ids = sorted(list(set(row[0] for row in rows)))
    split_idx = len(prompt_ids) // 2
    calib_prompts = set(prompt_ids[:split_idx])
    eval_prompts = set(prompt_ids[split_idx:])
    
    calibration_db = {}
    evaluation_db = {}
    
    for row in rows:
        p_id, t_pos, layer, exp_id, indices_str, k50 = row
        indices = json.loads(indices_str)
        
        # Take the columns within the energy threshold for profiling/replay
        # Active indices in DB are ordered by descending energy magnitude
        active_set = set(indices)
        
        target_db = calibration_db if p_id in calib_prompts else evaluation_db
        
        if p_id not in target_db:
            target_db[p_id] = {}
        if t_pos not in target_db[p_id]:
            target_db[p_id][t_pos] = {}
            
        target_db[p_id][t_pos][layer] = (exp_id, active_set, k50)
            
    print(f"Successfully loaded traces. Calibration prompts: {len(calibration_db)}, Evaluation prompts: {len(evaluation_db)}")
    return calibration_db, evaluation_db
