#!/usr/bin/env python3
import os
import shutil

def main():
    base_dir = "outputs/phase3"
    artifact_base_dir = "/home/palakm/.gemini/antigravity-ide/brain/ed3c2dc2-4e67-4959-b431-347772d4a219"

    # Define layout
    layout = {
        "ablation": [
            "epeg_ablation_breakdown.png"
        ],
        "ep_tp_scaling": [
            "epeg_ep_scaling_efficiency.png",
            "epeg_tp_vs_ep_scaling.png"
        ],
        "bandwidth_sensitivity": [
            "epeg_bandwidth_sensitivity.png"
        ],
        "topk": [
            "epeg_topk_scaling_study.png"
        ],
        "concurrency": [
            "epeg_concurrency_scaling.png"
        ]
    }

    # Include the consolidated JSON in all or specific folders, or keep it at base.
    # Let's copy to each folder for completeness.
    for folder, files in layout.items():
        # Paths
        out_folder = os.path.join(base_dir, folder)
        art_folder = os.path.join(artifact_base_dir, folder)
        
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(art_folder, exist_ok=True)
        
        for fn in files:
            # Source paths
            src_out = os.path.join(base_dir, fn)
            src_art = os.path.join(artifact_base_dir, fn)
            
            # Destination paths
            dst_out = os.path.join(out_folder, fn)
            dst_art = os.path.join(art_folder, fn)
            
            # Copy in output dir
            if os.path.exists(src_out):
                shutil.copy(src_out, dst_out)
                print(f"Copied {fn} to {out_folder}")
            else:
                print(f"Warning: {src_out} not found")
                
            # Copy in artifact dir
            if os.path.exists(src_art):
                shutil.copy(src_art, dst_art)
                print(f"Copied {fn} to {art_folder}")
            else:
                print(f"Warning: {src_art} not found")
                
        # Also copy JSON results to each folder for easy access
        json_fn = "epeg_tp_ep_sweep_results.json"
        src_json_out = os.path.join(base_dir, json_fn)
        src_json_art = os.path.join(artifact_base_dir, json_fn)
        
        if os.path.exists(src_json_out):
            shutil.copy(src_json_out, os.path.join(out_folder, json_fn))
        if os.path.exists(src_json_art):
            shutil.copy(src_json_art, os.path.join(art_folder, json_fn))

    print("Success: results categorized into subfolders!")

if __name__ == "__main__":
    main()
