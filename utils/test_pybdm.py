
import numpy as np
import sys

try:
    from pybdm import BDM
    print("pybdm imported successfully.")
except ImportError:
    print("pybdm not installed.")
    sys.exit(1)

try:
    bdm = BDM(ndim=2, alphabet=4)
    print("BDM(ndim=2, alphabet=4) initialized.")
except TypeError:
    print("BDM(ndim=2, alphabet=4) failed. Trying BDM(ndim=2).")
    try:
        bdm = BDM(ndim=2)
        print("BDM(ndim=2) initialized.")
    except Exception as e:
        print(f"BDM init failed: {e}")
        sys.exit(1)

# Test with small matrix
mat = np.random.randint(0, 4, (10, 10), dtype=int)
try:
    val = bdm.bdm(mat)
    print(f"BDM value for 10x10 matrix: {val}")
except Exception as e:
    print(f"BDM calculation failed: {e}")

# Test with larger matrix
mat_large = np.random.randint(0, 4, (128, 128), dtype=int)
try:
    val = bdm.bdm(mat_large)
    print(f"BDM value for 128x128 matrix: {val}")
except Exception as e:
    print(f"BDM calculation for large matrix failed: {e}")
