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
    "last_max_abs_error",
]


def parse(text: str, folder_path: Path):
    name_match = re.search(r"^(.*?)\s*\(", text)
    alpha_match = re.search(r"\balpha\s*=\s*([0-9.]+)", text)
    alpha_lower_match = re.search(r"\balpha_lower\s*=\s*([0-9.]+)", text)

    name = name_match.group(1).strip() if name_match else None
    alpha = float(alpha_match.group(1)) if alpha_match else None
    alpha_lower = float(alpha_lower_match.group(1)) if alpha_lower_match else None

    d = {
        "name": name,
        "alpha": alpha,
        "betta": alpha_lower,
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

        subset = df[df["name"] == name]

        heatmap_matrix = subset.pivot_table(
            index="betta",
            columns="alpha",
            values=value_key,
            aggfunc="mean",
        )

        heatmap_matrix = heatmap_matrix.sort_index(ascending=False)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_matrix,
            annot=True,
            fmt=".4f",
            cmap="viridis",
            cbar_kws={"label": value_key},
        )

        plt.title(f"{name}\nMetric: {value_key}")
        plt.xlabel("alpha")
        plt.ylabel("betta")
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

        print(f"Generowanie wykresu dla: {metric}...")

        # 1. Zliczanie max elementów w grupie.
        # Jeśli = 1, rysujemy barplot. Jeśli > 1 rysujemy prawdziwy boxplot.
        max_samples = df_all.groupby(["Test", "Architecture", "Noise", "name"])[metric].count().max()
        current_kind = "box" if max_samples > 1 else "bar"

        # Delikatna przezroczystość boxplotów (aby było widać kropki pod spodem)
        plot_kwargs = {"boxprops": {'alpha': 0.6}} if current_kind == "box" else {}

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

        # 2. Nakładanie kropek (stripplot) z wynikami
        if current_kind == "box":
            g.map_dataframe(
                sns.stripplot,
                x="Architecture",
                y=metric,
                hue="Noise",
                dodge=True,
                palette="dark:black",  # Ustawia kolor kropek na czarny
                alpha=0.6,
                size=5,
                legend=False
            )

        # Dopracowanie estetyczne tytułu
        title_parts = [f"Metryka: {metric}"]
        if not has_multiple_tests:
            title_parts.append(f"Test: {df_all['Test'].iloc[0]}")
        if not has_multiple_names:
            title_parts.append(f"Problem: {df_all['name'].iloc[0]}")

        g.fig.suptitle(" | ".join(title_parts), y=1.05, fontsize=14)

        # Zapis pod nazwą uwzględniającą typ wykresu (bar albo box)
        out_file = folder / f"global_{current_kind}plot_{metric}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Zapisano wykres -> {out_file}")


if __name__ == "__main__":
    list_path(path="plot")