import os
import glob
import json


def get_target_directories(base_path: str):
    pattern = os.path.join(base_path, "*", "*", "*")

    paths = glob.glob(pattern)

    valid_paths = [p.replace('\\', '/') for p in paths if os.path.isdir(p)]

    return valid_paths


if __name__ == "__main__":
    FOLDER_BAZOWY = "wysylka"

    sciezki = get_target_directories(FOLDER_BAZOWY)

    print(json.dumps(sciezki, ensure_ascii=False))