import os
import shutil
import re
import argparse

import yaml
import math
import matplotlib.pyplot as plt

# Słowniki używane do mapowania
MAPOWANIE = {
    "Convergence": "Funkcja straty vs l epok",
    "Loss_PDE": "Reziduum funkcji straty vs l epok",
    "Loss PDE": "Reziduum funkcji straty vs l epok",
    "Loss_Conditions": "Czesc warunkow funkcji straty vs l epok",
    "Loss Conditions": "Czesc warunkow funkcji straty vs l epok",
    "Loss_Conditions_Data": "Czesc pomiarow funkcji straty vs l epok",
    "Loss Conditions Data": "Czesc pomiarow funkcji straty vs l epok",
    "Grad_PDE_Max": "Maksymalny gradiend reziduum funkcji straty vs l epok",
    "Grad PDE Max": "Maksymalny gradiend reziduum funkcji straty vs l epok",
    "Grad_BC_Max": "Maksymalny gradiend czesci warunkow funkcji straty vs l epok",
    "Grad BC Max": "Maksymalny gradiend czesci warunkow funkcji straty vs l epok",
    "Grad_Data_Max": "Maksymalny gradiend czesci pomiarow funkcji straty vs l epok",
    "Grad Data Max": "Maksymalny gradiend czesci pomiarow funkcji straty vs l epok",
    "Network_Stiffness": "Sztywnosc vs l epok",
    "Network Stiffness": "Sztywnosc vs l epok",
    "Mean square error": "Blad sredniokwadratowy vs l epok",
    "loss_bc": "Czesc warunkow funkcji straty vs l epok",
    "loss_data": "Czesc pomiarow funkcji straty vs l epok",
    "Trainable variable_variable_0_value": "Wartosc parametru dyfuzji vs l epok",
    "Trainable variable_abs_error_variable_0_value": "Wartosc bledu absolutnego parametru dyfuzji vs l epok",
}

PLOT_GROUPS = {
    "1 przyklad Podejscie_1": ["1.1.a", "1.1.b1", "1.1.b2", "1.1.c", "1.1.d1", "1.1.d2"],
    "1 przyklad Podejscie_2": ["1.2.a", "1.2.b1", "1.2.b2", "1.2.c", "1.2.d1", "1.2.d2"],
    "2 przyklad Podejscie_1": ["2.1.a", "2.1.b1", "2.1.b2", "2.1.c", "2.1.d1", "2.1.d2"],
    "2 przyklad Podejscie_2": ["2.2.a", "2.2.b1", "2.2.b2", "2.2.c", "2.2.d1", "2.2.d2"],
}


def get_name_mapped(name: str) -> str | None:
    first_part = name.split(" ")[0]
    second_part = ""
    third_part = ""

    if "simple" in name:
        second_part = "1"
    else:
        second_part = "2"

    has_pde_1 = bool(re.search(r"weight_pde 1(?!\d)", name))
    has_data_1 = bool(re.search(r"weight_data 1(?!\d)", name))

    if "wang" in name:
        if has_pde_1 and has_data_1:
            third_part = "c"
        elif has_pde_1:
            third_part = "d2"
        else:
            third_part = "d1"
    else:
        if has_pde_1 and has_data_1:
            third_part = "a"
        elif has_pde_1:
            third_part = "b2"
        else:
            third_part = "b1"

    try:
        int(first_part)
        int(second_part)
    except ValueError:
        return None

    return f"{first_part}.{second_part}.{third_part}"


def map_and_clean_folders(output_folder: str):
    """Krok 3: Zmienia nazwy folderów i usuwa duplikaty/niepasujące"""
    used_names = set()
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        if not os.path.isdir(item_path):
            continue

        mapped_name = get_name_mapped(item)

        # Jeśli nazwa pomyślnie zmapowana i nie ma jeszcze takiego folderu
        if mapped_name and mapped_name is not None and mapped_name not in used_names:
            new_path = os.path.join(output_folder, mapped_name)
            if not os.path.exists(new_path):
                os.rename(item_path, new_path)
                used_names.add(mapped_name)
                print(f"Zmapowano folder: '{item}' -> '{mapped_name}'")
            else:
                shutil.rmtree(item_path)
                print(f"Usunięto (kolizja z innym): {item}")
        else:
            # Duplikaty (mapped_name in used_names) lub błędne mapowanie
            shutil.rmtree(item_path)
            print(f"Usunięto duplikat / śmieciowy folder: {item}")


def rename_files_in_folder(folder_path: str):
    """Krok 4: Mapuje nazwy plików wykresów, usuwa pliki których nie da się zmapować (zostawia yaml)"""
    if not os.path.exists(folder_path):
        return

    pliki = os.listdir(folder_path)
    grupy = {}

    for plik in pliki:
        sciezka = os.path.join(folder_path, plik)
        if not os.path.isfile(sciezka):
            continue

        nazwa, rozszerzenie = os.path.splitext(plik)
        nazwa = nazwa.strip()
        nazwa_lower = nazwa.lower()
        dopasowano = False

        # 1. Szukamy dopasowania
        for klucz in MAPOWANIE.keys():
            klucz_lower = klucz.lower()
            if nazwa_lower == klucz_lower:
                grupy.setdefault((klucz, rozszerzenie), []).append((plik, None))
                dopasowano = True
                break
            elif nazwa_lower.startswith(klucz_lower + "-"):
                sufiks = nazwa_lower[len(klucz_lower) + 1:]
                if sufiks.isdigit():
                    grupy.setdefault((klucz, rozszerzenie), []).append((plik, int(sufiks)))
                    dopasowano = True
                    break

        # 2. Sprawdzamy czy plik nie został wcześniej zmapowany
        if not dopasowano:
            for wartosc in MAPOWANIE.values():
                wartosc_lower = wartosc.lower()
                if nazwa_lower == wartosc_lower or nazwa_lower.startswith(f"{wartosc_lower} - zoom "):
                    dopasowano = True
                    break

        # 3. Usuwamy pliki których nie udało się zmapować (nie dotyczy yaml/yml)
        if not dopasowano:
            if plik.lower().endswith(('.yaml', '.yml')):
                continue
            os.remove(sciezka)

    # 4. Nadawanie nowych nazw i suffiksów zoom
    for (klucz, rozszerzenie), lista_plikow in grupy.items():
        lista_plikow.sort(key=lambda x: (x[1] is not None, x[1] if x[1] is not None else 0))
        nowa_nazwa_bazowa = MAPOWANIE[klucz]
        licznik_zoom = 1

        for oryginalny_plik, numer_sufiksu in lista_plikow:
            if numer_sufiksu is None:
                nowa_nazwa = f"{nowa_nazwa_bazowa}{rozszerzenie}"
            else:
                nowa_nazwa = f"{nowa_nazwa_bazowa} - zoom {licznik_zoom}{rozszerzenie}"
                licznik_zoom += 1

            if oryginalny_plik != nowa_nazwa:
                stara_sciezka = os.path.join(folder_path, oryginalny_plik)
                nowa_sciezka = os.path.join(folder_path, nowa_nazwa)
                try:
                    os.replace(stara_sciezka, nowa_sciezka)
                except Exception as e:
                    print(f"BŁĄD przy zmianie {oryginalny_plik}: {e}")


def load_save_data(folder_path: str):
    file_path = os.path.join(folder_path, 'save_data.yml')
    print(f"loading {file_path}")
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Błąd pliku YAML ({file_path}): {e}")
        return None


def dir_data(base_folder: str):
    items = os.listdir(base_folder)
    data = dict()
    for item in items:
        path = os.path.join(base_folder, item)
        if os.path.isdir(path):
            d = load_save_data(path)
            if d is not None:
                data[item] = d
                print(f"Loaded {path}")
    return data


def find_elbow_index(y_values):
    if len(y_values) < 10: return 0
    min_y, max_y = min(y_values), max(y_values)
    val_range = max_y - min_y
    if val_range == 0: return 0

    y_norm = [(val - min_y) / val_range for val in y_values]
    x_norm = [i / (len(y_values) - 1) for i in range(len(y_values))]
    dy = y_norm[-1] - y_norm[0]
    denominator = math.sqrt(dy ** 2 + 1)
    max_dist = -1
    elbow_idx = 0

    for i in range(len(y_values)):
        dist = abs(dy * x_norm[i] - y_norm[i] + y_norm[0]) / denominator
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i
    return elbow_idx


def get_mapped_stat_name(stat_name: str):
    mapping_lower = {k.lower(): v for k, v in MAPOWANIE.items()}
    return mapping_lower.get(stat_name.lower())


def plot_merged_statistics(data, output_folder, max_zooms=3):
    if not data:
        return

    os.makedirs(output_folder, exist_ok=True)
    all_statistics = set()
    for stats in data.values():
        all_statistics.update(stats.keys())

    for stat_name in all_statistics:
        plot_lines = []
        for folder_id, stats in data.items():
            if stat_name in stats:
                x = stats[stat_name].get("x", [])
                y = stats[stat_name].get("y", [])
                if x and y:
                    plot_lines.append((folder_id, x, y))

        if not plot_lines:
            continue

        display_name = get_mapped_stat_name(stat_name)
        if display_name is None:
            continue

        safe_filename = display_name.replace("/", "_").replace("\\", "_").replace(" ", "_")

        # Rysowanie pełnego wykresu
        plt.figure(figsize=(12, 7))
        for folder_id, x, y in plot_lines:
            plt.plot(x, y, label=folder_id, marker='.', markersize=4)

        plt.title(f"{display_name} (Pełny układ)", fontsize=14, pad=15)
        plt.xlabel("Oś X", fontsize=12)
        plt.ylabel("Wartość", fontsize=12)
        plt.legend(title="ID Folderu", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{safe_filename}.png"), dpi=150)
        plt.close()

        # Rysowanie przybliżeń (Zoom)
        current_start_idx = 0
        total_points = len(plot_lines[0][1])

        for zoom_level in range(1, max_zooms + 1):
            elbow_indices = []
            for folder_id, x, y in plot_lines:
                current_y = y[current_start_idx:]
                elbow_indices.append(find_elbow_index(current_y))

            if not elbow_indices: break
            avg_elbow_step = int(sum(elbow_indices) / len(elbow_indices))
            if avg_elbow_step <= 0: break

            current_start_idx += avg_elbow_step
            max_allowed_idx = int(total_points * 0.95)
            current_start_idx = min(current_start_idx, max_allowed_idx)

            plt.figure(figsize=(12, 7))
            for folder_id, x, y in plot_lines:
                plt.plot(x[current_start_idx:], y[current_start_idx:], label=folder_id, marker='.', markersize=4)

            plt.title(f"{display_name} (Zoom {zoom_level})", fontsize=14, pad=15)
            plt.xlabel("Oś X", fontsize=12)
            plt.ylabel("Wartość", fontsize=12)
            plt.legend(title="ID Folderu", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{safe_filename}_zoom_{zoom_level}.png"), dpi=150)
            plt.close()

            if current_start_idx >= max_allowed_idx:
                break


def process_all_data(output_folder: str):
    # Krok 3
    print("\n--- Mapowanie i czyszczenie podfolderów ---")
    map_and_clean_folders(output_folder)

    # Krok 4
    print("\n--- Zmiana nazw plików i usuwanie niezmapowanych obrazków ---")
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        if os.path.isdir(item_path) and item != "wykresy_wynikowe":
            rename_files_in_folder(item_path)

    # Krok 5
    print("\n--- Tworzenie folderów zbiorczych z wykresami z .yaml ---")
    collected_data = dir_data(output_folder)
    print("\nLoaded data")
    base_output_folder = os.path.join(output_folder, "wykresy_wynikowe")  # Folder wewnątrz "wysylki"

    # Tworzymy folder docelowy dla wykresów w środku `wysylka`
    if os.path.exists(base_output_folder):
        shutil.rmtree(base_output_folder)
    os.makedirs(base_output_folder, exist_ok=True)

    for group_name, allowed_ids in PLOT_GROUPS.items():
        grouped_data = {folder_id: stats for folder_id, stats in collected_data.items() if folder_id in allowed_ids}

        if grouped_data:
            print(f"Generuję wykresy dla grupy: {group_name} ...")
            group_output_folder = os.path.join(base_output_folder, group_name)
            plot_merged_statistics(grouped_data, group_output_folder, max_zooms=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przetwarzanie danych z podanej ścieżki.")
    parser.add_argument("data_path", type=str, help="Ścieżka do katalogu z danymi")

    args = parser.parse_args()

    process_all_data(args.data_path)
    print("\nGotowe!")
