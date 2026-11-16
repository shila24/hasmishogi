"""Configure, build, and test the portable C++ backend."""
import argparse
from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--build-dir", default="build")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not shutil.which("cmake"):
        raise SystemExit("CMake was not found. Install CMake and a C++20 compiler first.")
    build = root / args.build_dir
    configure = ["cmake", "-S", str(root), "-B", str(build),
                 "-DCMAKE_BUILD_TYPE=" + ("Debug" if args.sanitize else "Release"),
                 "-DHASAMI_SANITIZE=" + ("ON" if args.sanitize else "OFF")]
    if sys.platform == "darwin" and not os.environ.get("CMAKE_OSX_ARCHITECTURES"):
        # An x86_64 CMake under Rosetta would otherwise emit an Intel library that arm64 Python cannot load.
        configure.append("-DCMAKE_OSX_ARCHITECTURES=" + platform.machine())
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--parallel", "2"], check=True)
    subprocess.run(["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"], check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
