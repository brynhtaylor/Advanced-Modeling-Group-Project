import os
import subprocess
import webbrowser
import time
import sys

# Try to import pkg_resources, which comes with setuptools
try:
    import pkg_resources
except ImportError:
    # If pkg_resources is not available, try installing setuptools
    print("Setuptools is missing. Installing setuptools...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    # Try importing pkg_resources again
    import pkg_resources

import pkg_resources

print("Running setup script...")
# Rest of the setup script...

# Get the current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Set up the virtual environment directory
venv_dir = os.path.join(current_directory, 'venv')

print("Setting up virtual environment...")
# Create a virtual environment if it doesn't exist
if not os.path.exists(venv_dir):
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
print("Virtual environment set up.")

# Activate the virtual environment
activate_script = os.path.join(venv_dir, 'Scripts', 'activate')
pip_exec = os.path.join(venv_dir, 'Scripts', 'pip')
print("Virtual environment activated.")

# Read requirements.txt
with open(os.path.join(current_directory, 'requirements.txt')) as f:
    required = f.read().splitlines()

# Get installed packages in the virtual environment
installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# Install missing dependencies with prefer binary to avoid long build times
for requirement in required:
    package, version = requirement.split("==")
    if package in installed and installed[package] == version:
        print(f"{package} is already installed.")
    else:
        print(f"Installing {requirement}...")
        subprocess.check_call([pip_exec, "install", "--prefer-binary", requirement])

# Start Flask server in a subprocess
print("\nStarting Flask server...")
flask_process = subprocess.Popen(
    ["python", os.path.join(current_directory, "Frontier_Formulator_v11", "app.py")],
    shell=True
)

# Allow server some time to start
time.sleep(2)

# Open the web browser
print("Opening browser...")
webbrowser.open("http://127.0.0.1:5000")

# Wait for Flask process to end
flask_process.wait()

# Keep the terminal open if something goes wrong
input("\nPress Enter to close the window (if setup fails or needs manual intervention)...")