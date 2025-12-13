import os
import sys
import subprocess

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output file path
output_file = os.path.join(script_dir, "output.txt")

# Get Python version
python_version = sys.version.split()[0]

# Get pip version
try:
    pip_version = subprocess.check_output(
        [sys.executable, "-m", "pip", "--version"],
        text=True
    ).strip()
except Exception as e:
    pip_version = f"Error checking pip version: {e}"

# Write results to file
with open(output_file, "w") as f:
    f.write(f"Python version: {python_version}\n")
    f.write(f"Pip version: {pip_version}\n")

print(f"Versions written to {output_file}")
