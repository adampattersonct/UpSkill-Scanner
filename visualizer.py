"""Publication-ready chart generation for UpskillScan."""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import pandas as pd

from aggregator import CATEGORY_LABELS


# Category color mapping (colorblind-friendly)
CATEGORY_COLORS = {
    "technical_skills": "#0077BB",   # Blue
    "soft_skills": "#009988",        # Teal/Green
    "certifications": "#EE7733",     # Orange
    "domain_knowledge": "#AA3377",   # Purple
}
CATEGORY_COLORS_BY_LABEL = {
    CATEGORY_LABELS[k]: v for k, v in CATEGORY_COLORS.items()
}


class SkillVisualizer:
    """Generates publication-ready visualizations."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    def bar_chart_top_skills(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
        filename: str = "top_skills_bar.png",
    ) -> str:
        """Horizontal bar chart of top skills, colored by category."""
        if df.empty:
            print("  No data for bar chart")
            return ""

        top = df.head(top_n).copy()
        top = top.iloc[::-1]  # Reverse for horizontal bar (highest at top)

        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        colors = [
            CATEGORY_COLORS.get(cat, "#999999") for cat in top["category"]
        ]
        bars = ax.barh(top["skill"], top["count"], color=colors)

        # Add count labels
        for bar, count in zip(bars, top["count"]):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=10,
            )

        ax.set_xlabel("Number of Postings", fontsize=12)
        ax.set_ylabel("")
        ax.set_title(f"Top {top_n} In-Demand Skills", fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=label)
            for label, color in CATEGORY_COLORS_BY_LABEL.items()
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def heatmap_skills_by_region(
        self,
        df: pd.DataFrame,
        top_n: int = 15,
        filename: str = "skills_region_heatmap.png",
    ) -> str:
        """Heatmap of skills x regions with percentages."""
        if df.empty:
            print("  No data for heatmap")
            return ""

        # df should be the pivot matrix from aggregate_by_region
        # Drop the total_count column for display
        display_cols = [c for c in df.columns if c != "total_count"]
        plot_df = df[display_cols].head(top_n)

        fig, ax = plt.subplots(figsize=(14, max(10, top_n * 0.6)))

        sns.heatmap(
            plot_df,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "% of Postings"},
        )

        ax.set_title(
            f"Top {top_n} Skills by Region (% of Postings)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=11, rotation=0)
        ax.tick_params(axis="x", labelsize=11, rotation=45)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def treemap_by_category(
        self,
        national_df: pd.DataFrame,
        filename: str = "skills_treemap.png",
    ) -> str:
        """Treemap of skills grouped by category."""
        if national_df.empty:
            print("  No data for treemap")
            return ""

        # Get top skills per category
        parts = []
        for cat, group in national_df.groupby("category"):
            parts.append(group.nlargest(10, "count"))
        top_per_cat = pd.concat(parts, ignore_index=True)

        if top_per_cat.empty:
            print("  No skills to plot in treemap")
            return ""

        labels = []
        sizes = []
        colors = []

        for _, row in top_per_cat.iterrows():
            labels.append(f"{row['skill']}\n({row['count']})")
            sizes.append(row["count"])
            colors.append(CATEGORY_COLORS.get(row["category"], "#999999"))

        fig, ax = plt.subplots(figsize=(14, 10))

        squarify.plot(
            sizes=sizes,
            label=labels,
            color=colors,
            alpha=0.85,
            text_kwargs={"fontsize": 9, "wrap": True},
            ax=ax,
            edgecolor="black",
            linewidth=2,
        )

        ax.set_title(
            "Skills by Category (size = frequency)",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=label)
            for label, color in CATEGORY_COLORS_BY_LABEL.items()
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            fontsize=10,
            framealpha=0.9,
        )

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def bar_chart_by_region(
        self,
        region_name: str,
        national_df: pd.DataFrame,
        region_matrix: pd.DataFrame,
        top_n: int = 15,
        filename: str = None,
    ) -> str:
        """Bar chart for a single region."""
        if region_matrix.empty or region_name not in region_matrix.columns:
            print(f"  No data for region {region_name}")
            return ""

        region_col = region_matrix[region_name].sort_values(ascending=False).head(top_n)
        region_col = region_col.iloc[::-1]  # Reverse for horizontal bar

        # Map skills to categories from national_df
        skill_to_cat = dict(zip(national_df["skill"], national_df["category"]))

        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        colors = [
            CATEGORY_COLORS.get(skill_to_cat.get(skill, ""), "#999999")
            for skill in region_col.index
        ]
        bars = ax.barh(region_col.index, region_col.values, color=colors)

        for bar, pct in zip(bars, region_col.values):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=10,
            )

        ax.set_xlabel("% of Postings", fontsize=12)
        ax.set_title(
            f"Top {top_n} Skills — {region_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=11)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=label)
            for label, color in CATEGORY_COLORS_BY_LABEL.items()
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        plt.tight_layout()
        if filename is None:
            filename = f"top_skills_{region_name.lower()}.png"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path
