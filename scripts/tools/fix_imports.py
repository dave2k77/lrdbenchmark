
import os

target_dir = r"c:\Users\davia\git_clone_projects\lrdbenchmark\lrdbenchmark\analysis\high_performance"
old_import = "from models.estimators.base_estimator import BaseEstimator"
new_import = "from lrdbenchmark.analysis.base_estimator import BaseEstimator"

print(f"Scanning {target_dir}...")

count = 0
for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_import in content:
                print(f"Fixing {file}...")
                new_content = content.replace(old_import, new_import)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                count += 1

print(f"Fixed {count} files.")
