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

        # Exclude certifications from treemap
        working_df = national_df[national_df["category"] != "certifications"]

        # Group by category so each category forms a solid color block
        parts = []
        for cat in ["technical_skills", "soft_skills", "domain_knowledge"]:
            group = working_df[working_df["category"] == cat]
            if not group.empty:
                parts.append(group.nlargest(10, "count"))
        top_per_cat = pd.concat(parts, ignore_index=True)

        if top_per_cat.empty:
            print("  No skills to plot in treemap")
            return ""

        labels = []
        sizes = []
        colors = []

        for _, row in top_per_cat.iterrows():
            # Single-line labels fit better in thin rectangular boxes
            labels.append(f"{row['skill']} ({row['count']})")
            sizes.append(row["count"])
            colors.append(CATEGORY_COLORS.get(row["category"], "#999999"))

        fig, ax = plt.subplots(figsize=(16, 10))

        # Compute rects first, then draw with per-box font sizing
        normed = squarify.normalize_sizes(sizes, 16, 10)
        rects = squarify.squarify(normed, 0, 0, 16, 10)

        for rect, label, color in zip(rects, labels, colors):
            ax.add_patch(plt.Rectangle(
                (rect["x"], rect["y"]), rect["dx"], rect["dy"],
                facecolor=color, alpha=0.85, edgecolor="black", linewidth=2,
            ))
            # Scale font size based on box area
            area_pct = (rect["dx"] * rect["dy"]) / (16 * 10)
            fontsize = max(7, min(13, 7 + area_pct * 60))
            ax.text(
                rect["x"] + rect["dx"] / 2,
                rect["y"] + rect["dy"] / 2,
                label,
                ha="center", va="center",
                fontsize=fontsize,
                fontweight="bold" if fontsize >= 9 else "normal",
            )

        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)

        ax.set_title(
            "Skills by Category (size = frequency)",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")

        # Legend
        from matplotlib.patches import Patch
        shown_cats = set(top_per_cat["category"])
        legend_elements = [
            Patch(facecolor=color, label=label)
            for label, color in CATEGORY_COLORS_BY_LABEL.items()
            if any(cat for cat, c in CATEGORY_COLORS.items() if c == color and cat in shown_cats)
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_elements),
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
