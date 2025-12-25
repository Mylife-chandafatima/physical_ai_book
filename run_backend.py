import os
import subprocess
import sys

def install_dependencies():
    """Install required Python packages"""
    print("Installing backend dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_server():
    """Run the FastAPI server"""
    print("Starting the RAG chatbot server...")
    os.system("uvicorn main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    # Change to backend directory
    os.chdir("backend")
    
    # Install dependencies
    install_dependencies()
    
    # Run the server
    run_server()