import os
import site

search_dirs = site.getsitepackages()
for d in search_dirs:
    for root, dirs, files in os.walk(d):
        if "lib" in dirs:
            lib_path = os.path.join(root, "lib")
            if any(f.startswith("libcudnn.so") for f in os.listdir(lib_path)):
                print(f"CuDNN found at: {lib_path}")
            if any(f.startswith("libcusolver.so") for f in os.listdir(lib_path)):
                print(f"CuSolver found at: {lib_path}")
