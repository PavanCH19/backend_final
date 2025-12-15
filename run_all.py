import os
import sys
import subprocess
import platform

# -------------------------------------------------
# PATHS
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

VENV_DIR = os.path.join(PROJECT_ROOT, "venv")
REQ_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")

PYTHON_MODELS_DIR = os.path.join(PROJECT_ROOT, "python_models")
FINE_TUNE_DIR = os.path.join(PYTHON_MODELS_DIR, "fine_tune")
SETUP_DIR = os.path.join(PYTHON_MODELS_DIR, "setup")

IS_WINDOWS = platform.system() == "Windows"


def run(cmd, cwd=None):
    print(f"\n▶ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)


# -------------------------------------------------
# 1️⃣ CREATE VENV
# -------------------------------------------------
def create_venv():
    if not os.path.exists(VENV_DIR):
        print("🐍 Creating virtual environment...")
        run(f"{sys.executable} -m venv venv", cwd=PROJECT_ROOT)
    else:
        print("✅ Virtual environment already exists")


# -------------------------------------------------
# 2️⃣ GET VENV PYTHON
# -------------------------------------------------
def venv_python():
    if IS_WINDOWS:
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")


# -------------------------------------------------
# 3️⃣ INSTALL PYTHON DEPS
# -------------------------------------------------
def install_python_deps():
    print("📦 Installing Python dependencies...")
    python = venv_python()
    run(f"{python} -m pip install --upgrade pip")
    run(f"{python} -m pip install -r {REQ_FILE}")


# -------------------------------------------------
# 4️⃣ INSTALL NODE DEPS
# -------------------------------------------------
def install_node_deps():
    print("📦 Installing npm dependencies...")
    run("npm install", cwd=PROJECT_ROOT)


# -------------------------------------------------
# 5️⃣ RUN BOTH MODELS
# -------------------------------------------------
def run_models():
    python = venv_python()

    print("🧠 Running fine-tune model...")
    run(f"{python} fine_tune.py", cwd=FINE_TUNE_DIR)

    print("🧠 Running setup/index model...")
    run(f"{python} index.py", cwd=SETUP_DIR)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print("\n================ BACKEND PIPELINE STARTED ================\n")

    create_venv()
    install_python_deps()
    install_node_deps()
    run_models()

    print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
