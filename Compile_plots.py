import os
import yaml
import shutil
import math
import matplotlib.pyplot as plt

NAME_MAPPED = {
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
    "Mean square error": "Blad sredniokwadratowy vs l epok"
}

PLOT_GROUPS = {
    "Podejscie_1": ["1a", "1b", "1c", "1d"],
    "Podejscie_2": ["2a", "2b", "2c", "2d"]
}


def clear_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def load_save_data(folder_path):
    file_path = os.path.join(folder_path, 'save_data.yml')
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as yaml_error:
        print(f"Błąd podczas parsowania pliku YAML ({file_path}): {yaml_error}")
        return None
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd ({file_path}): {e}")
        return None


def dir_data(base_folder):
    items = os.listdir(base_folder)
    data = dict()
    for item in items:
        path = os.path.join(base_folder, item)
        if os.path.isdir(path):
            d = load_save_data(path)
            if d is not None:
                data[item] = d
    return data


def find_elbow_index(y_values):
    if len(y_values) < 10:
        return 0

    min_y, max_y = min(y_values), max(y_values)
    val_range = max_y - min_y
    if val_range == 0:
        return 0

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


def get_mapped_name(stat_name):
    mapping_lower = {k.lower(): v for k, v in NAME_MAPPED.items()}
    return mapping_lower.get(stat_name.lower(), stat_name)


def plot_merged_statistics(data, output_folder="wykresy", max_zooms=3):
    if not data:
        print(f"Brak danych do wygenerowania wykresów w folderze: {output_folder}")
        return

    clear_and_create_folder(output_folder)

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

        display_name = get_mapped_name(stat_name)
        safe_filename = display_name.replace("/", "_").replace("\\", "_").replace(" ", "_")

        plt.figure(figsize=(12, 7))
        for folder_id, x, y in plot_lines:
            plt.plot(x, y, label=folder_id, marker='.', markersize=4)

        plt.title(f"{display_name} (Pełny układ)", fontsize=14, pad=15)
        plt.xlabel("Oś X", fontsize=12)
        plt.ylabel("Wartość", fontsize=12)
        plt.legend(title="ID Folderu", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        full_save_path = os.path.join(output_folder, f"{safe_filename}.png")
        plt.savefig(full_save_path, dpi=150)
        plt.close()

        current_start_idx = 0
        total_points = len(plot_lines[0][1])

        for zoom_level in range(1, max_zooms + 1):
            elbow_indices = []

            for folder_id, x, y in plot_lines:
                current_y = y[current_start_idx:]
                elbow_indices.append(find_elbow_index(current_y))

            if not elbow_indices:
                break

            avg_elbow_step = int(sum(elbow_indices) / len(elbow_indices))

            if avg_elbow_step <= 0:
                break

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

            zoom_save_path = os.path.join(output_folder, f"{safe_filename}_zoom_{zoom_level}.png")
            plt.savefig(zoom_save_path, dpi=150)
            plt.close()

            if current_start_idx >= max_allowed_idx:
                break

        print(f"Wygenerowano pliki dla: {stat_name} -> {display_name}")


if __name__ == "__main__":
    target_folder = "wysyl/bez szumów"
    base_output_folder = "wykresy_wynikowe"

    collected_data = dir_data(target_folder)

    for group_name, allowed_ids in PLOT_GROUPS.items():
        print(f"\n--- Przetwarzanie grupy: {group_name} ---")

        grouped_data = {folder_id: stats for folder_id, stats in collected_data.items() if folder_id in allowed_ids}

        if grouped_data:
            group_output_folder = os.path.join(base_output_folder, group_name)
            plot_merged_statistics(grouped_data, group_output_folder, max_zooms=4)
        else:
            print(f"Brak pasujących katalogów dla ID zdefiniowanych w grupie {group_name}")