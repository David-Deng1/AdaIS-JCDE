"""
hitlo.config.color_scheme
============================
Nordic Precision Scientific Console -- color tokens.

Design direction: cool, authoritative, premium.
Signature colour: #00D4AA (emerald cyan).
"""

# ---------- Primary palette ----------
PRIMARY = "#0F2B47"
PRIMARY_LIGHT = "#1A4A7A"

# ---------- Signature & accent colors ----------
ACCENT_CYAN = "#00D4AA"       # signature emerald cyan
ACCENT_BLUE = "#3B82F6"
ACCENT_AMBER = "#F59E0B"
ACCENT_ROSE = "#EF4444"
ACCENT_EMERALD = "#10B981"
ACCENT_PURPLE = "#8B5CF6"

# ---------- Backward-compatible aliases ----------
ACCENT_TEAL = ACCENT_CYAN
ACCENT_ORANGE = ACCENT_AMBER
ACCENT_RED = ACCENT_ROSE
ACCENT_GREEN = ACCENT_EMERALD

# ---------- Background / surface ----------
BG_PAGE = "#F8FAFB"
BG_CARD = "#FFFFFF"
BG_INSET = "#F1F5F9"

# ---------- Border ----------
BORDER_DEFAULT = "#E2E8F0"

# ---------- Text ----------
TEXT_PRIMARY = "#1A1A2E"
TEXT_SECONDARY = "#64748B"
TEXT_TERTIARY = "#94A3B8"

# ---------- Shadows ----------
SHADOW_SM = "0 1px 3px rgba(15,43,71,0.06), 0 1px 2px rgba(15,43,71,0.04)"
SHADOW_MD = "0 4px 12px rgba(15,43,71,0.08), 0 2px 4px rgba(15,43,71,0.04)"
SHADOW_LG = "0 8px 24px rgba(15,43,71,0.10), 0 4px 8px rgba(15,43,71,0.05)"
SHADOW_GLOW_CYAN = "0 0 20px rgba(0,212,170,0.15)"

# ---------- Ordered color cycle for Plotly traces (10 colours) ----------
# ---------- Attribute color scales (G10: cross-view attribute mapping) ----------
def get_attribute_colorscale(mode: str = "default"):
    """Return a Plotly-compatible colorscale for cross-view attribute mapping.

    Parameters
    ----------
    mode : str
        One of: "default", "cv" (constraint violation), "feasibility",
        "cluster", "diversity", "satisfaction".
    """
    scales = {
        "default": "Viridis",
        "cv": [[0, ACCENT_EMERALD], [0.5, ACCENT_AMBER], [1, ACCENT_ROSE]],
        "feasibility": [[0, ACCENT_ROSE], [1, ACCENT_EMERALD]],
        "cluster": "Set2",
        "diversity": [[0, "#F8FAFB"], [1, ACCENT_PURPLE]],
        "satisfaction": [[0, ACCENT_ROSE], [0.5, ACCENT_AMBER], [1, ACCENT_EMERALD]],
    }
    return scales.get(mode, "Viridis")


PLOTLY_COLORS = [
    ACCENT_CYAN,      # #00D4AA  emerald cyan (signature first)
    ACCENT_BLUE,      # #3B82F6  blue
    ACCENT_AMBER,     # #F59E0B  amber
    ACCENT_PURPLE,    # #8B5CF6  purple
    ACCENT_ROSE,      # #EF4444  rose
    ACCENT_EMERALD,   # #10B981  emerald
    PRIMARY_LIGHT,    # #1A4A7A  blue light
    "#06B6D4",        # cyan-500
    "#F97316",        # orange-500
    "#A855F7",        # violet-500
]
