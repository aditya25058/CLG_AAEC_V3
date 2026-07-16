#!/usr/bin/env python3
import os
import shutil
import glob

def main():
    base_dir = "outputs/phase3"
    artifact_base_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219"

    # Define target subfolders and patterns for files
    groups = {
        "bandwidth_sweep": [
            "epeg_results.json",
            "epeg_results_table.png",
            "epeg_comparison_plot.png",
            "epeg_disabled_bw_*.csv",
            "epeg_enabled_bw_*.csv"
        ],
        "threshold_ablation": [
            "epeg_pareto_frontier.png",
            "ablation_baseline.csv",
            "ablation_th_*.csv"
        ],
        "gain_ablation": [
            "epeg_gain_ablation.json",
            "epeg_gain_ablation.png",
            "epeg_ablation_baseline.csv",
            "epeg_ablation_comm_only.csv",
            "epeg_ablation_compute_only.csv",
            "epeg_ablation_full_epeg.csv"
        ],
        "topk_scaling": [
            "epeg_topk_scaling.json",
            "epeg_topk_scaling.png",
            "epeg_topk_*_disabled.csv",
            "epeg_topk_*_enabled.csv"
        ],
        "sota_comparison": [
            "epeg_sota_comparison.json",
            "epeg_sota_comparison.png",
            "compare_*.csv",
            "cross_*.csv"
        ],
        "multinode_baseline": [
            "test_multinode_baseline.csv",
            "test_multinode_epeg.csv"
        ]
    }

    for folder, patterns in groups.items():
        out_folder = os.path.join(base_dir, folder)
        art_folder = os.path.join(artifact_base_dir, folder)
        
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(art_folder, exist_ok=True)
        
        for pat in patterns:
            # Match files in the base outputs dir
            matched_files = glob.glob(os.path.join(base_dir, pat))
            for fpath in matched_files:
                fn = os.path.basename(fpath)
                
                # Destination paths
                dst_out = os.path.join(out_folder, fn)
                dst_art = os.path.join(art_folder, fn)
                
                # Copy outputs to subfolder
                shutil.copy(fpath, dst_out)
                print(f"Organized: {fn} -> {out_folder}")
                
                # Check if it exists in base artifacts dir, and copy to artifact subfolder
                src_art = os.path.join(artifact_base_dir, fn)
                if os.path.exists(src_art):
                    shutil.copy(src_art, dst_art)
                    print(f"Organized in artifacts: {fn} -> {art_folder}")

    print("Success: Organized all Phase 3 files into subfolders!")

if __name__ == "__main__":
    main()
