#save this file where your model exists

#!/usr/bin/env python3
import os
import subprocess
import re
import sys
import importlib.util
import hashlib
import argparse
import logging
import shutil
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LLaMA model weights to Hugging Face format")
    parser.add_argument("--input_dir", default="/root/.llama/checkpoints", help="Directory containing LLaMA model folders")
    parser.add_argument("--convert_script", default="/usr/local/lib/python3.10/dist-packages/transformers/models/llama/convert_llama_weights_to_hf.py", help="Path to the conversion script")
    parser.add_argument("--dry_run", action="store_true", help="Perform a dry run without actual conversion")
    return parser.parse_args()

def check_accelerate():
    spec = importlib.util.find_spec("accelerate")
    if spec is None:
        logging.info("Accelerate library not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "accelerate>=0.26.0"], check=True)
    else:
        logging.info("Accelerate library found.")

def get_model_size(model_name: str) -> Optional[str]:
    match = re.search(r'(\d+)B', model_name)
    return match.group(1) if match else None

def check_file_integrity(file_path: str) -> bool:
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            for chunk in iter(lambda: f.read(8192), b""):
                file_hash.update(chunk)
        logging.info(f"MD5 hash for {file_path}: {file_hash.hexdigest()}")
        return True
    except Exception as e:
        logging.error(f"Error checking file integrity for {file_path}: {e}")
        return False

def convert_model(model_folder: str, args: argparse.Namespace) -> None:
    output_folder = f"{model_folder}-hf"
    output_path = os.path.join(args.input_dir, output_folder)
    
    # Check if output directory exists and remove it if it does
    if os.path.exists(output_path):
        logging.info(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    model_path = os.path.join(args.input_dir, model_folder)
    model_size = get_model_size(model_folder)
    if not model_size:
        logging.warning(f"Could not determine model size for {model_folder}. Skipping.")
        return

    checkpoint_files = [f for f in os.listdir(model_path) if f.startswith("consolidated") and f.endswith(".pth")]
    if not all(check_file_integrity(os.path.join(model_path, f)) for f in checkpoint_files):
        logging.error(f"File integrity check failed for {model_folder}. Skipping conversion.")
        return

    cmd = [
        sys.executable,
        args.convert_script,
        "--input_dir", model_path,
        "--model_size", f"{model_size}B",
        "--output_dir", output_path
    ]

    if any(version in model_folder for version in ["3.1", "3.2"]):
        cmd.extend(["--llama_version", "3.1"])

    logging.info(f"Converting {model_folder}...")
    logging.info(f"Conversion parameters:")
    logging.info(f"  Input directory: {model_path}")
    logging.info(f"  Model size: {model_size}B")
    logging.info(f"  Output directory: {output_path}")
    logging.info(f"  LLaMA version: {'3.1' if '3.1' in model_folder or '3.2' in model_folder else '2'}")
    logging.info(f"Running command: {' '.join(cmd)}")
    
    if args.dry_run:
        logging.info(f"Dry run: would execute {' '.join(cmd)}")
        return

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(result.stdout)
        logging.info(f"Successfully converted {model_folder} to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting {model_folder}:")
        logging.error(e.stdout)
        logging.error(e.stderr)
    except Exception as e:
        logging.error(f"Unexpected error converting {model_folder}: {e}")

def main():
    args = setup_argparse()

    if not os.path.exists(args.input_dir):
        logging.error(f"Error: Input directory {args.input_dir} does not exist.")
        return

    if not os.path.exists(args.convert_script):
        logging.error(f"Error: Conversion script not found at {args.convert_script}")
        return

    check_accelerate()

    llama_folders = [folder for folder in os.listdir(args.input_dir) 
                     if os.path.isdir(os.path.join(args.input_dir, folder)) and folder.startswith("Llama")]
    
    if not llama_folders:
        logging.warning(f"No Llama model folders found in {args.input_dir}")
        return

    print("Available models for conversion:")
    for i, folder in enumerate(llama_folders, 1):
        print(f"{i}. {folder}")

    while True:
        try:
            choice = int(input("Enter the number of the model you want to convert: "))
            if 1 <= choice <= len(llama_folders):
                selected_folder = llama_folders[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    convert_model(selected_folder, args)

if __name__ == "__main__":
    main()
