import os
import shutil
import stat


def on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def prepare_and_copy_folders(input_folder: str, output_folder: str):
    abs_input = os.path.abspath(input_folder)
    abs_output = os.path.abspath(output_folder)

    if os.name == 'nt':
        abs_input = "\\\\?\\" + abs_input
        abs_output = "\\\\?\\" + abs_output

    if os.path.exists(abs_output):
        print(f"Próbuję usunąć stary folder docelowy: {abs_output}")
        shutil.rmtree(abs_output, onerror=on_rm_error)

    print(f"Kopiuję {abs_input} -> {abs_output}")
    shutil.copytree(abs_input, abs_output, dirs_exist_ok=True)


def cp_dir(input_folder: str, output_folder: str):
    abs_input = os.path.abspath(input_folder)
    if os.name == 'nt':
        abs_input = "\\\\?\\" + abs_input

    if not os.path.exists(abs_input):
        print(f"BŁĄD: Zdefiniowany folder źródłowy '{abs_input}' nie istnieje.")
        return

    prepare_and_copy_folders(input_folder, output_folder)


if __name__ == "__main__":
    FOLDER_ZRODLOWY = "plot"
    FOLDER_DOCELOWY = "wysylka"
    cp_dir(FOLDER_ZRODLOWY, FOLDER_DOCELOWY)