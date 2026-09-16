import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

metric_keys = [
    "variable_0_last_value",
    "variable_0_last_abs_error",
    "last_mean_square_error",
    "last_loss",
    "last_network_stiffness",
    "last_loss_pde",
    "last_loss_conditions",
    "last_loss_conditions_data",
    "last_grad_pde_max",
    "last_grad_bc_max",
    "last_grad_data_max",
    "last_grad_pde_mean",
    "last_grad_data_mean",
    "last_grad_bc_mean",
    "last_max_abs_error"
]


def parse(text: str, folder_path: Path):
    name_match = re.search(r"^(.*?)(?:\s*\(|\s+with noise\b|\s+n\s*=|\s+weight\s*=|\s+alpha\s*=|$)",
                           text, flags=re.IGNORECASE)

    alpha_match = re.search(r"\balpha\s*=\s*([0-9.]+)", text)
    alpha_lower_match = re.search(r"\balpha_lower\s*=\s*([0-9.]+)", text)
    weight_match = re.search(r"\bweight\s*=\s*([0-9.]+)", text)
    n_match = re.search(r"\bn\s*=\s*([0-9.]+)", text)
    wang_flag = bool(re.search(r"\bwang\b", text, flags=re.IGNORECASE))

    name = name_match.group(1).strip() if name_match else text.strip()
    alpha = float(alpha_match.group(1)) if alpha_match else None
    alpha_lower = float(alpha_lower_match.group(1)) if alpha_lower_match else None

    weight = round(float(weight_match.group(1)), 1) if weight_match else None
    n_param = round(float(n_match.group(1)), 1) if n_match else None
    if weight is None:
        weight = n_param
    if weight is None:
        return None
    print(name)

    d = {
        "name": name,
        "alpha": alpha,
        "betta": alpha_lower,
        "weight": weight,
        "is_wang_dynamic_weight": wang_flag,
    }

    data_file = folder_path / "data.yml"
    if data_file.is_file():
        with open(data_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            if isinstance(data, dict):
                d.update(data)

    return d


def plot_heatmaps(df: pd.DataFrame, path: Path, value_key: str = "last_mean_square_error"):
    if value_key not in df.columns:
        return

    unique_names = df["name"].dropna().unique()

    for name in unique_names:
        safe_name = re.sub(r'[\\/*?:"<>| ]', "_", name)
        file = path / f"{safe_name}_{value_key}.png"
        print(f"Zapisywanie heatmapy: {file}")

        subset = df[df["name"] == name].copy()

        if subset.empty:
            continue

        heatmap_matrix = subset.pivot_table(
            index="is_wang_dynamic_weight",
            columns="weight",
            values=value_key,
            aggfunc="mean",
        )

        heatmap_matrix = heatmap_matrix.sort_index(ascending=False)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_matrix,
            annot=True,
            fmt=".4g",
            cmap="viridis",
            cbar_kws={"label": value_key},
        )

        plt.title(f"{name}\nMetric: {value_key}")

        plt.xlabel("static weight value")
        plt.ylabel("dynamic weight")
        plt.tight_layout()

        plt.savefig(file, dpi=300)
        plt.close()


def list_path(path: str):
    folder = Path(path)

    all_raw_data = []

    for test_dir in [p for p in folder.iterdir() if p.is_dir()]:
        for architecture_dir in [p for p in test_dir.iterdir() if p.is_dir()]:
            for noise_dir in [p for p in architecture_dir.iterdir() if p.is_dir()]:

                local_data = []

                for example_dir in [p for p in noise_dir.iterdir() if p.is_dir()]:
                    example_data = parse(example_dir.name, folder_path=example_dir)
                    if example_data is None:
                        continue

                    local_data.append(example_data.copy())

                    example_data["Test"] = test_dir.name
                    example_data["Architecture"] = architecture_dir.name
                    example_data["Noise"] = noise_dir.name

                    all_raw_data.append(example_data)

                if not local_data:
                    print(f"Pominięto pusty katalog: {noise_dir}")
                    continue

                df_local = pd.DataFrame(local_data)
                for metric_key in metric_keys:
                    plot_heatmaps(df_local, path=noise_dir, value_key=metric_key)

    if not all_raw_data:
        print("Nie znaleziono żadnych danych.")
        return

    print("\n--- Generowanie zbiorczych wykresów ---")
    df_all = pd.DataFrame(all_raw_data)

    # 3. Usuwanie pustych wymiarów z siatki, jeśli jest tylko 1 test lub 1 problem (name)
    has_multiple_tests = df_all["Test"].nunique() > 1
    has_multiple_names = df_all["name"].nunique() > 1

    for metric in metric_keys:
        if metric not in df_all.columns:
            print(f"Brak danych dla metryki: {metric}")
            continue

        boxplot_folder = folder / "boxplots"
        boxplot_folder.mkdir(parents=True, exist_ok=True)
        print(f"Generowanie Box Plot dla: {metric}...")

        g = sns.catplot(
            data=df_all,
            x="Architecture",
            y=metric,
            hue="Noise",
            col="name" if has_multiple_names else None,
            row="Test" if has_multiple_tests else None,
            kind=current_kind,
            sharey=False,
            palette="Set2",
            height=5,
            aspect=1.2,
            **plot_kwargs
        )

        g.fig.suptitle(f"Zbiorczy Box Plot: {metric}", y=1.03, fontsize=16)

        out_file = boxplot_folder / f"global_boxplot_{metric}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Zapisano wykres -> {out_file}")


if __name__ == "__main__":
    list_path(path="plot")