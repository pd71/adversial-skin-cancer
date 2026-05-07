"""Environment validation script for local development setup."""

import platform
import sys
from typing import List


def _check_import(module_name: str) -> bool:
    """Try importing a module and report status."""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {module_name} imported (version: {version})")
        return True
    except ImportError:
        print(f"[MISSING] {module_name} is not installed.")
        return False
    except Exception as exc:  # pragma: no cover - defensive for local envs
        print(f"[ERROR] {module_name} import failed: {exc}")
        return False


def main() -> None:
    """Run environment checks for Python and required packages."""
    print("=== Environment Check ===")
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print()

    missing_or_failed: List[str] = []

    tf_ok = _check_import("tensorflow")
    if tf_ok:
        import tensorflow as tf  # Imported only after success for explicit reporting.

        print(f"[INFO] TensorFlow available: {tf.__version__}")
    else:
        missing_or_failed.append("tensorflow")

    for package in ["numpy", "pandas", "sklearn"]:
        if not _check_import(package):
            missing_or_failed.append(package)

    print()
    if missing_or_failed:
        unique_missing = sorted(set(missing_or_failed))
        print("[ACTION REQUIRED] Some dependencies are missing or failed to import:")
        print(", ".join(unique_missing))
        print("Run this command from project root:")
        print("pip install -r requirements.txt")
        sys.exit(1)

    print("[SUCCESS] Environment looks good. All required packages are available.")


if __name__ == "__main__":
    main()
