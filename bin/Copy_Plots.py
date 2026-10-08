import os
import shutil


def safe_copytree(src, dst):
    """
    Autorskie kopiowanie, które zmusza Pythona do ignorowania metadanych
    katalogów i plików. Rozwiązuje problem 'Operation not permitted' w Dockerze.
    """
    # Tworzymy folder docelowy (ignorujemy błąd, jeśli już istnieje)
    os.makedirs(dst, exist_ok=True)

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)

        if os.path.isdir(s):
            # Rekurencyjne kopiowanie dla podfolderów
            safe_copytree(s, d)
        else:
            # Usuwamy stary plik, jeśli istnieje, aby go swobodnie nadpisać
            try:
                if os.path.exists(d):
                    os.remove(d)
            except Exception:
                pass

            # copyfile kopiuje TYLKO bitową zawartość pliku (żadnych uprawnień/metadanych)
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
        # Próbujemy usunąć, ale jak się nie uda przez blokady Dockera, idziemy dalej
        shutil.rmtree(abs_output, ignore_errors=True)

    print(f"Kopiuję {abs_input} -> {abs_output}")
    # Używamy naszej kuloodpornej funkcji zamiast shutil.copytree
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