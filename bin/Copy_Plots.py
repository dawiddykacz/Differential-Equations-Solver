import os
import shutil


def prepare_and_copy_folders(input_folder: str, output_folder: str):
    # Konwersja na ścieżki absolutne
    abs_input = os.path.abspath(input_folder)
    abs_output = os.path.abspath(output_folder)

    # Dodanie prefixu omijającego limit długości w Windows
    if os.name == 'nt':
        abs_input = "\\\\?\\" + abs_input
        abs_output = "\\\\?\\" + abs_output

    """Krok 1: Próba usunięcia z wymuszeniem ignorowania zablokowanych plików"""
    if os.path.exists(abs_output):
        print(f"Próbuję usunąć stary folder docelowy: {abs_output}")
        # ignore_errors=True sprawia, że skrypt nie przerwie się, jeśli Docker przytrzyma jakiś plik
        shutil.rmtree(abs_output, ignore_errors=True)

    """Krok 2: Kopiowanie z nadpisywaniem BEZ metadanych"""
    print(f"Kopiuję {abs_input} -> {abs_output}")

    # KLUCZOWA ZMIANA: copy_function=shutil.copy
    # Zmusza Pythona do kopiowania samej zawartości plików z pominięciem metadanych
    # (których modyfikacja wywoływała błąd "Operation not permitted" w Dockerze).
    shutil.copytree(
        abs_input,
        abs_output,
        dirs_exist_ok=True,
        copy_function=shutil.copy
    )


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