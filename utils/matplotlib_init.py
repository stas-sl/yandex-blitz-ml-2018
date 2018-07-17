import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython import get_ipython
from cycler import cycler

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

colors = ['#388EF3', '#FF9649']

params = {
    'figure.facecolor': '#D3C8FF',
    'axes.facecolor': '#D3C8FF',
    'savefig.facecolor': '#D3C8FF',
    'savefig.pad_inches': 0.2,
    'savefig.bbox': 'standard',
    'savefig.frameon': False,
    'figure.figsize': (620 / 72, 380 / 72),
    'font.family': 'Kalam',
    'legend.fontsize': 'xx-large',
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 24,
    'xtick.labelsize': 'xx-large',
    'ytick.labelsize': 'xx-large',
    'figure.titleweight': 'normal',
    'axes.titleweight': 'normal',
    'axes.xmargin': 0.05,
    'axes.ymargin': 0.05,
    'axes.titlepad': 16.0,
    'legend.frameon': True,
    'legend.edgecolor': '#AC8AF5',
    'legend.framealpha': 0.8,
    'legend.fancybox': False,
    'axes.prop_cycle': cycler('color', colors),
    'axes.edgecolor': '#360C90',
    'xtick.color': '#360C90',
    'axes.labelcolor': '#360C90',
    'ytick.color': '#360C90',
    'text.color': '#360C90',
    'grid.linestyle': '--',
    'grid.linewidth': 1,
    'grid.color': '#360C90',
    'grid.alpha': 0.1
}

plt.rcParams.update(params)

font_files = font_manager.findSystemFonts(fontpaths=['assets/'])
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
