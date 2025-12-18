import os
from pathlib import Path


ckpt_dir = Path("/mnt/public/algm/yzy/train_repos/verl/logs")


for item in ckpt_dir.iterdir():
    if item.is_dir() is False:
        print(f"{item} is not a directory")
        continue

    for subitem in item.iterdir():
        if subitem.is_dir() is False:
            os.system(f"rm -rf {subitem}")
            continue

        files = [subsubitem.name for subsubitem in subitem.iterdir()]
        if "hf" not in files:
            os.system(f"rm -rf {subitem}")


# from IPython import embed; embed()