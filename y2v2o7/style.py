"""Publication figure style (Nature Physics conventions)."""

import matplotlib as mpl

RC = {
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':          7,
    'axes.labelsize':     7,
    'axes.titlesize':     7,
    'xtick.labelsize':    6,
    'ytick.labelsize':    6,
    'legend.fontsize':    6,
    'axes.linewidth':     0.5,
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
    'xtick.major.size':   2.5,
    'ytick.major.size':   2.5,
    'xtick.minor.size':   1.5,
    'ytick.minor.size':   1.5,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'lines.linewidth':    0.8,
    'figure.dpi':         150,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
}

COL1 = 3.386   # 8.6 cm — single column
COL2 = 7.008   # 17.8 cm — double column

C_BLUE   = '#0073BD'
C_RED    = '#D92B2B'
C_ORANGE = '#ED8C00'
C_GREEN  = '#38A12B'
C_PURPLE = '#9533BF'
C_GREY   = '#808080'
C_BLACK  = '#1A1A1A'


def use():
    mpl.rcParams.update(RC)


def label(ax, letter, dark_bg=False, x=0.025, y=0.97):
    """Add Nature Physics panel label: bold 'a.' at top-left."""
    color = 'white' if dark_bg else 'black'
    bbox = dict(facecolor='k', alpha=0.55, pad=1.5,
                boxstyle='round,pad=0.2', edgecolor='none') if dark_bg else None
    ax.text(x, y, f'{letter}.', transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left', color=color,
            bbox=bbox)


def save(fig, name):
    for ext in ('pdf', 'png'):
        path = f'Figures/{name}.{ext}'
        fig.savefig(path, dpi=600 if ext == 'pdf' else 300)
        print(f'  saved {path}')
