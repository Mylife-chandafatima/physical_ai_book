import os
import subprocess
import sys

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies for data loading...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_data_loader():
    """Run the data loader script"""
    print("Loading book data into Qdrant...")
    os.system("python load_data.py")

if __name__ == "__main__":
    # Change to backend directory
    os.chdir("backend")
    
    # Install dependencies
    install_dependencies()
    
    # Run the data loader
    run_data_loader()