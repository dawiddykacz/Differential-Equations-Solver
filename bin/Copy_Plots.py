import os
import shutil

def safe_copytree(src, dst):
    os.makedirs(dst, exist_ok=True)

    with os.scandir(src) as it:
        for entry in it:
            s = entry.path
            d = os.path.join(dst, entry.name)

            if entry.is_dir():
                safe_copytree(s, d)
            else:
                try:
                    try:
                        os.remove(d)
                    except FileNotFoundError:
                        pass
                except Exception:
                    pass

                try:
                    shutil.copyfile(s, d)
                except Exception as e:
                    print(f"Nie udało się skopiować pliku {s}: {e}")


def prepare_and_copy_folders(input_folder: str, output_folder: str):
    abs_input = os.path.abspath(input_folder)
    abs_output = os.path.abspath(output_folder)

    if os.name == 'nt':
        abs_input = "\\\\?\\" + abs_input
        abs_output = "\\\\?\\" + abs_output

    if os.path.exists(abs_output):
        print(f"Próbuję usunąć stary folder docelowy: {abs_output}")
        shutil.rmtree(abs_output, ignore_errors=True)

    print(f"Kopiuję {abs_input} -> {abs_output}")
    safe_copytree(abs_input, abs_output)


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