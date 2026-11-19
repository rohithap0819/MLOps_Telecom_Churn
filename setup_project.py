"""
Project Setup Script
--------------------
Prepares the local Git/DVC environment for both:

1. GitHub Clone
2. GitHub Download ZIP

For a Git clone:
    - Existing Git repository is preserved.
    - Existing DVC configuration is preserved.

For a ZIP download:
    - A local Git repository is created automatically.
    - The existing baseline dataset is added to the local DVC cache.
    - No GitHub repository or remote is created.

Run:
    python setup_project.py
"""

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_DATA = PROJECT_ROOT / "data" / "data_sources" / "train.csv"
BASELINE_DVC = PROJECT_ROOT / "data" / "data_sources" / "train.csv.dvc"
GIT_DIR = PROJECT_ROOT / ".git"
DVC_DIR = PROJECT_ROOT / ".dvc"


def run_command(command, description):
    """Run a command and stop if it fails."""

    print()
    print(f"--> {description}")
    print("    " + " ".join(command))

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True
    )

    if result.returncode != 0:
        print()
        print(f"[ERROR] {description} failed.")
        sys.exit(result.returncode)


def check_command(command_name, install_message):
    """Check whether a required command is available."""

    if shutil.which(command_name) is None:
        print()
        print(f"[ERROR] '{command_name}' was not found.")
        print()
        print(install_message)
        sys.exit(1)


def main():
    print("=" * 70)
    print("MLOps Telecom Churn - Project Setup")
    print("=" * 70)

    print()
    print(f"Project directory: {PROJECT_ROOT}")

    # ------------------------------------------------------------
    # 1. Check required tools
    # ------------------------------------------------------------
    print()
    print("Checking required tools...")

    check_command(
        "git",
        "Please install Git and make sure 'git' is available from the terminal."
    )

    check_command(
        "dvc",
        "Please install the project requirements first using:\n"
        "    python -m pip install -r requirements.txt"
    )

    print("[OK] Git is available.")
    print("[OK] DVC is available.")

    # ------------------------------------------------------------
    # 2. Check baseline files
    # ------------------------------------------------------------
    print()
    print("Checking project files...")

    if not BASELINE_DATA.exists():
        print()
        print(f"[ERROR] Baseline dataset not found:")
        print(f"        {BASELINE_DATA}")
        sys.exit(1)

    if not BASELINE_DVC.exists():
        print()
        print(f"[ERROR] DVC metadata not found:")
        print(f"        {BASELINE_DVC}")
        sys.exit(1)

    if not DVC_DIR.exists():
        print()
        print("[ERROR] .dvc directory is missing.")
        print("        The project DVC configuration is incomplete.")
        sys.exit(1)

    print("[OK] Baseline dataset found.")
    print("[OK] Baseline DVC metadata found.")
    print("[OK] .dvc directory found.")

    # ------------------------------------------------------------
    # 3. Detect GitHub clone vs ZIP
    # ------------------------------------------------------------
    print()
    print("Checking Git repository...")

    if GIT_DIR.exists():
        print("[OK] Existing Git repository detected.")
        print("     Project appears to be a GitHub clone.")

        clone_mode = True

    else:
        print("[INFO] No Git repository detected.")
        print("       Project appears to have been extracted from a ZIP.")

        clone_mode = False

        # --------------------------------------------------------
        # 4. ZIP workflow
        # --------------------------------------------------------
        print()
        print("Preparing local Git environment for ZIP workflow...")

        run_command(
            ["git", "init"],
            "Creating local Git repository"
        )

        print()
        print("[OK] Local Git repository created.")

    # ------------------------------------------------------------
    # 5. Prepare DVC
    # ------------------------------------------------------------
    print()
    print("Checking DVC state...")

    if clone_mode:
        # A GitHub clone already contains the Git-tracked baseline
        # and its DVC metadata. Do not run `dvc add` against train.csv
        # because the baseline dataset is intentionally also kept in Git.
        print()
        print("[INFO] GitHub clone detected.")
        print("[INFO] Existing DVC configuration will be preserved.")
        print("[INFO] Baseline train.csv will not be re-added to DVC.")

    else:
        # In a ZIP extraction there is no Git history yet.
        # Therefore train.csv is not Git-tracked locally, so DVC
        # can safely create/populate its local cache.
        print()
        print("[INFO] ZIP workflow detected.")
        print("[INFO] Preparing baseline dataset for local DVC usage...")

        run_command(
            [
                sys.executable,
                "-m",
                "dvc",
                "add",
                str(BASELINE_DATA.relative_to(PROJECT_ROOT))
            ],
            "Adding baseline dataset to local DVC cache"
        )

        print()
        print("[OK] Baseline dataset is available in the local DVC cache.")

    # ------------------------------------------------------------
    # 6. Verify DVC
    # ------------------------------------------------------------
    print()
    print("Verifying DVC setup...")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dvc",
            "status"
        ],
        cwd=PROJECT_ROOT,
        text=True
    )

    if result.returncode == 0:
        print()
        print("[OK] DVC setup check completed.")
    else:
        print()
        print("[WARNING] DVC status reported a problem.")
        print("          Review the message above before continuing.")

    # ------------------------------------------------------------
    # 7. Final message
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("PROJECT SETUP COMPLETE")
    print("=" * 70)

    print()
    print("You can now continue with the normal project workflow.")

    print()
    print("Next recommended steps:")
    print("  1. python -m pytest")
    print("  2. python modelling/train.py")
    print("  3. python modelling/model_selection.py")
    print("  4. Continue with the remaining project workflows.")

    print()
    print("Important:")
    print("- No GitHub repository was created.")
    print("- No GitHub remote was configured.")
    print("- Existing GitHub clone configuration was not replaced.")
    print("- ZIP users do not need to manually run git init.")
    print("- ZIP users do not need to manually run dvc add.")


if __name__ == "__main__":
    main()