# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Setup check
#
# Run this notebook from the top before Day 1. It installs nothing on your laptop beyond what the repository's
# lock file pins (locally) or `requirements-colab.txt` (in a hosted notebook), and it should print the library
# versions, **4,776 assessments** and two check marks.

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/fnd-aic-101.git"
    if not Path("/content/fnd-aic-101").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/fnd-aic-101"], check=True)
    os.chdir("/content/fnd-aic-101/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("hurbuth.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())
import platform
import matplotlib, numpy as np, pandas as pd, sklearn
import hurbuth as U

print(f"Python {platform.python_version()}; numpy {np.__version__}; pandas {pd.__version__}; "
      f"scikit-learn {sklearn.__version__}; matplotlib {matplotlib.__version__}")
df = U.assessments()
train, val, test = U.split(df)
print(f"{len(df)} assessments of {df.client_id.nunique()} clients; training {len(train)}, validation {len(val)}, test {len(test)}")
U.check(len(df) == 4776, "the case data are the course's version")
U.check(len(train) + len(val) + len(test) == len(df), "the split covers every assessment once")
