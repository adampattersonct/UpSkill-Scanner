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
        job_titles: list = None,
        regions: list = None,
        total_postings: int = None,
    ) -> str:
        """Treemap of skills grouped by category."""
        if national_df.empty:
            print("  No data for treemap")
            return ""

        # Exclude certifications from treemap
        working_df = national_df[national_df["category"] != "certifications"]

        # Group by category so each category forms a solid color block
        cat_order = ["technical_skills", "soft_skills", "domain_knowledge"]
        cat_groups = {}
        for cat in cat_order:
            group = working_df[working_df["category"] == cat]
            if not group.empty:
                cat_groups[cat] = group.nlargest(10, "count")

        if not cat_groups:
            print("  No skills to plot in treemap")
            return ""

        # Collect all shown skills for legend
        top_per_cat = pd.concat(cat_groups.values(), ignore_index=True)

        fig_w, fig_h = 16, 10
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Two-level layout: allocate a vertical strip per category
        # with a minimum width so smaller categories stay readable
        min_share = 0.22
        cat_totals = {cat: g["count"].sum() for cat, g in cat_groups.items()}
        grand_total = sum(cat_totals.values())

        raw_shares = {cat: t / grand_total for cat, t in cat_totals.items()}
        shares = {cat: max(min_share, s) for cat, s in raw_shares.items()}
        share_sum = sum(shares.values())
        shares = {cat: s / share_sum for cat, s in shares.items()}

        # Draw each category strip, then squarify skills within it
        x_offset = 0
        for cat in cat_order:
            if cat not in cat_groups:
                continue
            group = cat_groups[cat]
            strip_w = shares[cat] * fig_w
            color = CATEGORY_COLORS.get(cat, "#999999")

            sizes = group["count"].tolist()
            skill_names = [r["skill"] for _, r in group.iterrows()]
            counts = [r["count"] for _, r in group.iterrows()]

            normed = squarify.normalize_sizes(sizes, strip_w, fig_h)
            rects = squarify.squarify(normed, x_offset, 0, strip_w, fig_h)

            for rect, skill_name, count in zip(rects, skill_names, counts):
                ax.add_patch(plt.Rectangle(
                    (rect["x"], rect["y"]), rect["dx"], rect["dy"],
                    facecolor=color, alpha=0.85, edgecolor="black", linewidth=2,
                ))
                area_pct = (rect["dx"] * rect["dy"]) / (fig_w * fig_h)
                fontsize = max(7, min(13, 7 + area_pct * 60))

                # Estimate if single-line label fits the box width
                # ~0.11 units per character at fontsize 10 (heuristic)
                label_single = f"{skill_name} ({count})"
                char_w = 0.011 * fontsize
                pad = 0.4  # padding inside box edges
                avail_w = rect["dx"] - pad

                if len(label_single) * char_w <= avail_w:
                    # Fits on one line
                    label = label_single
                else:
                    # Try smaller font on one line
                    smaller = max(6, fontsize - 1.5)
                    char_w_s = 0.011 * smaller
                    if len(label_single) * char_w_s <= avail_w:
                        fontsize = smaller
                        label = label_single
                    else:
                        # Wrap: skill name on top, count below
                        fontsize = max(6, fontsize - 1)
                        label = f"{skill_name}\n({count})"

                ax.text(
                    rect["x"] + rect["dx"] / 2,
                    rect["y"] + rect["dy"] / 2,
                    label,
                    ha="center", va="center",
                    fontsize=fontsize,
                    fontweight="bold" if fontsize >= 9 else "normal",
                    linespacing=1.2,
                )

            x_offset += strip_w

        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)

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

        # Add job titles and region info below legend
        info_lines = []
        if job_titles:
            info_lines.append(f"Job titles: {', '.join(job_titles)}")
        if regions:
            info_lines.append(f"Regions: {', '.join(regions)}")
        if total_postings is not None:
            info_lines.append(f"n = {total_postings}")
        if info_lines:
            info_text = "  |  ".join(info_lines)
            ax.text(
                0.5, -0.08,
                info_text,
                ha="center", va="top",
                fontsize=9, style="italic", color="#555555",
                transform=ax.transAxes,
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
