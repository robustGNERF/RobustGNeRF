import json
import os

rootdir = '/disk1/chanho/3d/MetaFool/eval/llff_test'
# Define the file structure and models
models = ["eval_gnt_adv", "eval_gnt_adv_advAttack", "eval_gnt_plain", "eval_gnt_plain_advAttack", "eval_gnt_plain_near3", "eval_gnt_plain_near3_advAttack", "eval_gnt_adv_near3", "eval_gnt_adv_near3_advAttack"]
scenes = ["fern", "fortress", "flower", "orchids", "horns", "leaves", "room", "trex"]

def read_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    # Convert single quotes to double quotes and fix other JSON formatting issues
    data = data.replace("'", '"')
    try:
        return json.loads(data)
    except json.decoder.JSONDecodeError:
        # Handle any remaining JSON errors
        print(f"Error decoding JSON from file: {file_path}")
        return {}


# Main loop to go through each model and scene
results = {}
for model in models:
    results[model] = {}
    for scene in scenes:
        file_path = f"{rootdir}/{model}/psnr_{scene}_250000.txt"
        if not os.path.exists(file_path):
            file_path = f"{rootdir}/{model}/psnr_{scene}_240000.txt"
            if not os.path.exists(file_path):
                file_path = f"{rootdir}/{model}/psnr_{scene}_0.txt"
        data = read_file(file_path)

        # Extract metrics for the specific scene
        coarse_mean_psnr = data[scene]['coarse_mean_psnr']
        fine_mean_psnr = data[scene]['fine_mean_psnr']
        coarse_mean_ssim = data[scene]['coarse_mean_ssim']
        fine_mean_ssim = data[scene]['fine_mean_ssim']
        coarse_mean_lpips = data[scene]['coarse_mean_lpips']
        fine_mean_lpips = data[scene]['fine_mean_lpips']

        # Store results in our dictionary
        results[model][scene] = {
            'coarse_mean_psnr': coarse_mean_psnr,
            'fine_mean_psnr': fine_mean_psnr,
            'coarse_mean_ssim': coarse_mean_ssim,
            'fine_mean_ssim': fine_mean_ssim,
            'coarse_mean_lpips': coarse_mean_lpips,
            'fine_mean_lpips': fine_mean_lpips
        }

# Now, you have the summarized results stored in the results dictionary
# You can print them or process further as needed
for model, scenes_data in results.items():
    print(f"Model: {model}")
    for scene, metrics in scenes_data.items():
        print(f"  Scene: {scene}")
        for metric, value in metrics.items():
            if metric == 'fine_mean_psnr':
                print(f"    {metric}: {value}")

import csv 


# Now, save the summarized results to a CSV
with open("summarized_results.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Define CSV headers
    headers = ['Model', 'Scene', 'coarse_mean_psnr', 'fine_mean_psnr', 'coarse_mean_ssim', 'fine_mean_ssim', 'coarse_mean_lpips', 'fine_mean_lpips']
    csvwriter.writerow(headers)
    
    # Write data
    for model, scenes_data in results.items():
        for scene, metrics in scenes_data.items():
            row = [model, scene] + [round(metrics[metric],3) for metric in headers[2:]]
            csvwriter.writerow(row)

print("Summarized results saved to summarized_results.csv")
