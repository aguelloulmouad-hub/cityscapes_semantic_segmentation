"""
Setup helper: installs all required Python packages for segment_video.py.
Run this once before running the main script.

    python setup_and_run.py
"""
import subprocess
import sys

PACKAGES = [
    "torch",
    "torchvision",
    "segmentation-models-pytorch",
    "opencv-python",
    "numpy",
]

def main():
    print("[*] Installing required packages ...")
    for pkg in PACKAGES:
        print(f"  -> {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("[+] All packages installed.")
    print()
    print("[*] You can now run:")
    print("    python segment_video.py")

if __name__ == "__main__":
    main()
