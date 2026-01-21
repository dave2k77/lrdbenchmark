import sys
import os
import traceback

sys.path.append(os.getcwd())

with open("import_debug_log.txt", "w") as f:
    f.write("Debugging Imports...\n")
    
    f.write("\n1. Trying to import ml_pretrained\n")
    try:
        from lrdbenchmark.models.pretrained_models.ml_pretrained import (
            RandomForestPretrainedModel,
            SVREstimatorPretrainedModel,
            GradientBoostingPretrainedModel,
        )
        f.write("   SUCCESS\n")
    except Exception as e:
        f.write(f"   FAILED: {e}\n")
        traceback.print_exc(file=f)

    f.write("\n2. Trying to import cnn_pretrained\n")
    try:
        from lrdbenchmark.models.pretrained_models.cnn_pretrained import CNNPretrainedModel
        f.write("   SUCCESS\n")
    except Exception as e:
        f.write(f"   FAILED: {e}\n")
        traceback.print_exc(file=f)
