# The Data Diamond — Visual Style Guide

Reference for maintaining consistent branding across all Data Diamond projects and content.

---

## Color Palette

| Name | Hex | Usage |
|------|-----|-------|
| **Gold** | `#C8A96E` | Primary accent. Titles, projected values, CI bars, KDE fills, delta labels, footer brand name. |
| **Ember** | `#D4562A` | Power/elite. Barrels, hard hits, 95+ mph velo, dominant K counts, HR events, hit results. |
| **Sage** | `#6BA38E` | Positive/growth. Improvement trends, positive deltas, mid-tier highlights, strikes, walks. |
| **Slate** | `#7B8FA6` | Neutral. Secondary text, observed values, axes, reference lines, footer subtitle, watermark, decline bars. |
| **Cream** | `#F5F2EE` | Background. Figure, axes, and save facecolor. |
| **Dark** | `#0F1117` | Primary text on cream background. Player names, bold labels, axis ticks. |

### When to use what
- **Gold** = the thing you want people to look at (projections, headlines, brand name)
- **Ember** = power, danger, elite performance (barrels, 95+ velo, HRs, dominant Ks)
- **Sage** = good/positive direction (improvement trends, positive deltas, plate discipline)
- **Slate** = context/secondary info (observed values, baselines, supporting text)
- **Dark** = readable body text and bold labels

### Three-tier EV bars
| Tier | Threshold | Color |
|------|-----------|-------|
| Elite | 105+ mph | Gold |
| Hard | 95–104 mph | Ember |
| Below | <95 mph | Slate |

---

## Typography

| Element | Font | Size | Weight | Color |
|---------|------|------|--------|-------|
| Chart title (16:9) | sans-serif | 22pt | bold | Gold |
| Chart title (5:7, 1:1) | sans-serif | 18pt | bold | Gold |
| Chart subtitle (16:9) | sans-serif | 13pt | normal | Slate |
| Chart subtitle (5:7, 1:1) | sans-serif | 12pt | normal | Slate |
| Player name (card) | sans-serif | 26pt | bold | Dark |
| Team/hand label | sans-serif | 16pt | normal | Slate |
| Big stat number | sans-serif | 32pt | bold | Dark (observed) / Gold (projected) |
| Stat label | sans-serif | 13pt | normal | Slate |
| Bar chart name | sans-serif | 11pt | bold | Dark |
| Bar chart value | sans-serif | 11pt | bold | Dark |
| Delta label (pp) | sans-serif | 11pt | bold | Gold |
| Observed reference | sans-serif | 9pt | normal | Slate |
| Footer brand | sans-serif | 12pt | bold | Gold |
| Footer subtitle | sans-serif | 10pt | normal | Slate |
| Base rcParam | sans-serif | 14pt | normal | Dark |

**Font family**: Always `sans-serif` (system default). No custom fonts required.

---

## Figure Formats

### 16:9 Landscape (Twitter/X feed cards)
- **Size**: 16 x 9 inches
- **Use for**: Mover charts, comparison tables, leaderboards, movement profiles
- **Layout**: Typically 1x2 subplot grid (left/right panels)

### 5:7 Portrait (detail cards)
- **Size**: 10 x 14 inches
- **Use for**: Pitching line, at-bat breakdown, batter game cards
- **Layout**: Vertical stack via GridSpec or figure text

### 1:1 Square (single-stat features)
- **Size**: 10 x 10 inches
- **Use for**: Individual HR cards, player projection cards
- **Layout**: Vertical stack via GridSpec

### Output
- **Format**: PNG
- **DPI**: 300
- **Padding**: 0.3 inches (`savefig.pad_inches`)
- **Save directory**: `outputs/content/` (project-specific subdirectories)

---

## Chart Styling Rules

### Background & Frame
- Background: Cream (`#F5F2EE`) on figure, axes, and saved file
- **No spines** (top, right, left, bottom all hidden by default)
- **No grid lines**
- Exception: KDE density plots show bottom spine in Slate

### Bars
- **Height**: 0.55 units (horizontal bars)
- **Alpha**: 0.85
- **Color**: Sage for positive direction, Slate for negative direction, Ember for elite
- **X-axis range**: 0.05 to 0.45 for K% charts

### Reference Lines
- **Style**: Dashed (`--`)
- **Color**: Slate
- **Width**: 1.5pt
- Marks observed/baseline values

### KDE Density Plots
- **Fill**: Gold at alpha 0.3
- **Outline**: Gold at linewidth 2pt
- **Observed line**: Slate dashed, 2pt
- **Projected line**: Gold solid, 2.5pt
- **Bandwidth**: 0.25 (Scott's Rule variant)

### Credible Interval Bars
- **Line**: Gold, 8pt width, round cap
- **End caps**: Gold `|` marker, size 20
- **Mean marker**: Dark diamond (`D`), size 8
- **Endpoint labels**: 13pt Slate, placed below bar

### Legends
- Background: Cream
- Border: Slate
- Text: Dark
- Font size: 10pt
- Position: upper right (default)

---

## Watermark

- **Text**: "TheDataDiamond"
- **Size**: 60pt
- **Color**: Slate at **alpha 0.03** (barely visible)
- **Rotation**: 30 degrees
- **Position**: Dead center of figure (0.5, 0.5)
- **Z-order**: 0 (behind everything)

The watermark should be invisible at a glance but detectable on close inspection or when image contrast is adjusted.

---

## Brand Footer

Every chart gets a footer with three elements:

```
[Logo]  TheDataDiamond                          Live Game Content
^left                                                      right^
```

### Footer positions by aspect ratio

| Aspect | Logo pos | Logo zoom | Brand x | Subtitle x | Text y | Brand size | Sub size |
|--------|----------|-----------|---------|------------|--------|------------|----------|
| 16:9 | (0.03, 0.02) | 0.025 | 0.08 | 0.96 | 0.02 | 12pt | 10pt |
| 5:7 | (0.03, 0.01) | 0.018 | 0.08 | 0.96 | 0.012 | 11pt | 9pt |
| 1:1 | (0.03, 0.02) | 0.022 | 0.08 | 0.96 | 0.02 | 12pt | 10pt |

### Footer subtitle presets

| Key | Text |
|-----|------|
| `"live"` | Live Game Content |
| `"postgame"` | Post-Game Recap |
| `"highlight"` | Game Highlight |
| `"projection"` | Bayesian Projection Model |

Custom strings also accepted — presets are just shortcuts.

---

## Header System

Uniform structure across all aspect ratios:

```
         CARD TITLE              <- Gold, bold, centered
    Subtitle / context line      <- Slate, normal, centered
```

Font sizes scale by ratio (see Typography table above). The `add_header()` function handles positioning automatically.

---

## Name Formatting

Player names follow this pattern:

```
First Last (TEAM) XHP
```

- **First Last**: Full name (not abbreviated on mover charts)
- **TEAM**: 2-3 letter abbreviation in parentheses. Omit parentheses if no team data.
- **XHP**: `RHP` or `LHP` for pitchers (throwing hand), `RHP` or `LHP` for hitters (batting side)

On individual cards, the name is split:
- Line 1: Full name in all caps (26pt bold Dark)
- Line 2: `TEAM  |  XHP` (16pt Slate)

---

## Implementation

The canonical theme lives in the shared **`tdd_theme`** package (`E:\data_analytics\tdd_theme\`):

- **`tdd_theme/palette.py`** — GOLD, EMBER, SAGE, SLATE, CREAM, DARK
- **`tdd_theme/theme.py`** — `apply_theme()` rcParams
- **`tdd_theme/branding.py`** — `add_watermark()`, `add_header()`, `add_brand_footer()`
- **`tdd_theme/layout.py`** — aspect sizes, footer/header configs per ratio
- **`tdd_theme/save.py`** — `save_card()`, `format_pct()`

Each project has a thin wrapper that re-exports from `tdd_theme` and adds project-specific paths:
- **`gamefeed/scripts/theme.py`**
- **`player_profiles/src/viz/theme.py`**

To use in a new project:
```bash
pip install -e E:\data_analytics\tdd_theme
```
```python
from tdd_theme import apply_theme, add_watermark, add_brand_footer, save_card
from tdd_theme import GOLD, EMBER, SAGE, SLATE, CREAM, DARK

apply_theme()
fig, ax = plt.subplots()
# ... build your chart ...
add_watermark(fig)
add_brand_footer(fig, subtitle="projection")
save_card(fig, "my_chart", aspect="16:9")
```
