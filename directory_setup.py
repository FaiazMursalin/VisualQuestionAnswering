# run after unzipping the data folder
# creates required output folders

from pathlib import Path

folders = ["Checkpoints", "H5 models", "Keras models", "Output", "Pickle files", "PNG files"]
for p in folders:
    Path(f"/{p}").mkdir(parents=True, exist_ok=True)

