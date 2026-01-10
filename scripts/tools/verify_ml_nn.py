import sys
import os
import traceback

# Ensure we can import from the project root
sys.path.append(os.getcwd())

def debug_imports():
    print("Debugging Imports...")
    
    print("\n1. Trying to import lrdbenchmark.models.pretrained_models.ml_pretrained")
    try:
        from lrdbenchmark.models.pretrained_models.ml_pretrained import (
            RandomForestPretrainedModel,
            SVREstimatorPretrainedModel,
            GradientBoostingPretrainedModel,
        )
        print("   SUCCESS")
    except Exception as e:
        print("   FAILED")
        traceback.print_exc()

    print("\n2. Trying to import lrdbenchmark.models.pretrained_models.cnn_pretrained")
    try:
        from lrdbenchmark.models.pretrained_models.cnn_pretrained import CNNPretrainedModel
        print("   SUCCESS")
    except Exception as e:
        print("   FAILED")
        traceback.print_exc()

if __name__ == "__main__":
    debug_imports()
