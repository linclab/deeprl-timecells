"""
linclab_plot_utils.py

Authors: Colleen Gillon

Updated: March 2022

Requirements: 
- python (tested on 3.7)
- matplotlib (>= 3.3.1 for adding fonts from directory)
- numpy

Contents:
- LINCLAB_COLS          : module-wide dictionary of LiNCLab colors
- linclab_plt_defaults(): sets plotting parameters to use LiNCLab colors and 
			              to improve figure better readability.
                          Call it before you start plotting. 
                          NOTE: This just updates the defaults. You can still use 
                          custom plotting settings in your scripts. 
- linclab_colormap()    : returns a 2 or 3-color pyplot colormap using LiNCLab colors.
- update_font_manager() : updates mpl font manager with fonts from a directory.
- set_font()            : sets matplotlib font to preferred font/font family.
- help_logging()        : prints information on using the python `logging` module.
"""

import logging
from pathlib import Path
import warnings

import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

LINCLAB_COLS = {"blue"  : "#50a2d5", # Linclab blue
                "red"   : "#eb3920", # Linclab red
                "grey"  : "#969696", # Linclab grey
                "green" : "#76bb4b", # Linclab green
                "purple": "#9370db",
                "orange": "#ff8c00",
                "pink"  : "#bb4b76",
                "yellow": "#e0b424",
                "brown" : "#b04900",
                }


#############################################
def linclab_plt_defaults(font="Liberation Sans", fontdir=None, 
                         log_fonts=False, example=False, dirname=".", 
                         **cyc_args):
    """
    linclab_plt_defaults()

    Sets pyplot defaults to Linclab style.

    Optional args:
        - font (str or list): font (or font family) to use, or list in order of 
                              preference
                              default: "Liberation Sans"
        - fontdir (str)     : directory to where extra fonts (.ttf) are stored
                              default: None
        - log_fonts (bool)  : if True, an alphabetical list of available fonts 
                              is logged (at logging.INFO level).
                              NOTE: For tips on using logging, run help_logging().
                              default: False
        - example (bool)    : if True, an example plot is created and saved
                              default: False
        - dirname (str)     : directory in which to save example if example is 
                              True 
                              default: "."

    Kewyord args:
        - cyc_args (dict): keyword arguments for plt.cycler()
    """

    col_order = ["blue", "red", "grey", "green", "purple", "orange", "pink", 
                 "yellow", "brown"]
    colors = [LINCLAB_COLS[key] for key in col_order] 
    col_cyc = plt.cycler(color=colors, **cyc_args)

    # set pyplot params
    params = {"axes.labelsize"       : "x-large",  # x-large axis labels
              "axes.linewidth"       : 1.5,        # thicker axis lines
              "axes.prop_cycle"      : col_cyc,    # line color cycle
              "axes.spines.right"    : False,      # no axis spine on right
              "axes.spines.top"      : False,      # no axis spine at top
              "axes.titlesize"       : "x-large",  # x-large axis title
              "errorbar.capsize"     : 4,          # errorbar cap length
              "figure.titlesize"     : "xx-large", # xx-large figure title
              "figure.autolayout"    : True,       # adjusts layout
              "font.size"            : 12,         # basic font size value
              "legend.fontsize"      : "x-large",  # x-large legend text
              "lines.dashed_pattern" : [8.0, 4.0], # longer dashes
              "lines.linewidth"      : 2.5,        # thicker lines
              "lines.markeredgewidth": 2.5,        # thick marker edge widths 
                                                   # (e.g., cap thickness) 
              "lines.markersize"     : 10,         # bigger markers
              "patch.linewidth"      : 2.5,        # thicker lines for patches
              "savefig.format"       : "svg",      # figure save format
              "savefig.bbox"         : "tight",    # tight cropping of figure
              "xtick.labelsize"      : "x-large",  # x-large x-tick labels
              "xtick.major.size"     : 8.0,        # longer x-ticks
              "xtick.major.width"    : 2.0,        # thicker x-ticks
              "ytick.labelsize"      : "x-large",  # x-large y-tick labels
              "ytick.major.size"     : 8.0,        # longer y-ticks
              "ytick.major.width"    : 2.0,        # thicker y-ticks
              }

    set_font(font, fontdir, log_fonts)

    # update pyplot parameters
    plt.rcParams.update(params)

    # create and save an example plot, if requested
    if example:
        fig, ax = plt.subplots(figsize=[8, 8])
        
        n_col = len(colors)
        x = np.arange(10)[:, np.newaxis]
        y = np.repeat(x / 2., n_col, axis=1) - np.arange(-n_col, 0)
        ax.plot(x, y)

        # label plot
        legend_labels = [
            f"{name}: {code}" for name, code in zip(col_order, colors)
            ]
        ax.legend(legend_labels)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_title("Example plot", y=1.02)
        ax.axvline(x=1, ls="dashed", c="k")
    
        dirname = Path(dirname)
        dirname.mkdir(parents=True, exist_ok=True)
        
        ext = plt.rcParams["savefig.format"]
        savepath = dirname.joinpath(f"example_plot").with_suffix(f".{ext}")
        fig.savefig(savepath)

        logger.info(f"Example saved under {savepath}")



#############################################
def linclab_colormap(nbins=100, gamma=1.0, no_white=False):
    """
    linclab_colormap()

    Returns a matplotlib colorplot using the linclab blue, white and linclab 
    red.

    Optional args:
        - nbins (int)    : number of bins to use to create colormap
                           default: 100
        - gamma (num)    : non-linearity
                           default: 1.0
        - no_white (bool): if True, white as the intermediate color is omitted 
                           from the colormap.
                           default: False

    Returns:
        - cmap (colormap): a matplotlib colormap
    """

    colors = [LINCLAB_COLS["blue"], "#ffffff", LINCLAB_COLS["red"]]
    name = "linclab_bwr"
    if no_white:
        colors = [colors[0], colors[-1]]
        name = "linclab_br"

    # convert to RGB
    rgb_col = [[] for _ in range(len(colors))]
    for c, col in enumerate(colors):
        ch_vals = mpl.colors.to_rgb(col)
        for ch_val in ch_vals:
            rgb_col[c].append(ch_val)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        name, rgb_col, N=nbins, gamma=gamma)

    return cmap


#############################################
def update_font_manager(fontdir):
    """
    update_font_manager(fontdir)

    Adds fonts from a font directory to the font manager.
    
    Required args:
        - fontdir (Path): directory to where extra fonts (.ttf) are stored
    """

    fontdir = Path(fontdir)
    if not fontdir.exists():
        raise OSError(f"{fontdir} font directory does not exist.")

    # add new fonts to list of available fonts if a font directory is provided
    fontdirs = [fontdir, ]
    # prevent a long stream of debug messages
    logging.getLogger("matplotlib.font_manager").disabled = True
    font_files = fm.findSystemFonts(fontpaths=fontdirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)

    return


#############################################
def set_font(font="Liberation Sans", fontdir=None, log_fonts=False):
    """
    set_font()

    Sets pyplot font to preferred values.
    
    NOTE: This function is particular convoluted to enable to clearest warnings 
    when preferred fonts/font families are not found.

    Optional args:
        - font (str or list): font or font family to use, or list in order of 
                              preference
                              default: "Liberation Sans"
        - fontdir (str)     : directory to where extra fonts (.ttf) are stored
                              default: None
        - log_fonts (bool)  : if True, an alphabetical list of available fonts 
                              is logged (at logging.INFO level). 
                              NOTE: For tips on using logging, run help_logging().
                              default: False
    """

    # keep in lower case
    font_families = ["cursive", "family", "fantasy", "monospace", 
        "sans-serif", "serif"]

    if fontdir is not None:
        update_font_manager(fontdir)
    
    # compile list of available fonts/font families 
    # (includes checking each font family to see if any of its fonts are available)
    all_fonts = list(set([f.name for f in fm.fontManager.ttflist]))
    all_fonts_lower = [font.lower() for font in all_fonts]
    font_families_found = []
    none_found_str = " (no family fonts found)"
    for f, font_family in enumerate(font_families):
        available_fonts = list(filter(lambda x: x.lower() in all_fonts_lower, 
            plt.rcParams[f"font.{font_family}"]))
        if len(available_fonts) == 0:
            font_families_found.append(
                f"{font_family}{none_found_str}")
        else:
            font_families_found.append(font_family)

    # log list of font families and fonts, if requested
    if log_fonts:
        font_log = ""
        for i, (font_type, font_list) in enumerate(zip(
            ["Font families", "Available fonts"], [font_families, all_fonts])):
            sep = "" if i == 0 else "\n\n"
            sorted_fonts_str = f"\n{TAB}".join(sorted(font_list))
            font_log = (f"{font_log}{sep}{font_type}:"
                f"\n{TAB}{sorted_fonts_str}")
        
        logger.info(font_log)
    
    # compile ordered list of available fonts/font families, in the preferred 
    # order to use in setting the mpl font choice parameter.
    fonts = font
    if not isinstance(fonts, list):
        fonts = [fonts]

    params = {
        "font.family": plt.rcParams["font.family"]
    }
    fonts_idx_added = []
    for f, font in enumerate(fonts):
        if font.lower() in all_fonts_lower:
            font_name = all_fonts[all_fonts_lower.index(font.lower())]
        elif (font.lower() in font_families_found and 
            none_found_str not in font.lower()):
            font_name = font.lower()
        else:
            if font.lower() in font_families:
                fonts[f] = f"{font} family fonts"
            continue

        # if found, add/move to correct position in list
        if font_name in params["font.family"]:
            params["font.family"].remove(font_name)

        params["font.family"].insert(len(fonts_idx_added), font_name)
        fonts_idx_added.append(f)

    #  warn if the first (set of) requested fonts/font families were not found.
    if len(fonts_idx_added) == 0:
        first_font_added = None
    else: 
        first_font_added = min(fonts_idx_added)
    if first_font_added != 0:
        omitted_str = ", ".join(fonts[: first_font_added])
        selected_str = ""
        if len(plt.rcParams["font.family"]) != 0:
            selected = plt.rcParams["font.family"][0]
            selected_str = f"\nFont set to {selected}."
            if selected in font_families:
                selected_str = selected_str.replace(".", " family.")
        warnings.warn(f"Requested font(s) not found: {omitted_str}."
            f"{selected_str}", category=UserWarning, stacklevel=1)
    
    plt.rcParams.update(params)

    return


#############################################
def help_logging():
    """
    help_logging()
    
    Prints instructions for basic use of logging python module.
    """

    logging_tips = ("For basic logging to the console, run:\n\n"
        "import logging\n"
        "logging.basicConfig(level=logging.INFO)"
        "\n\nThis should allow subsequent logs at the logging.INFO level and "
        "above to be printed to the console.")
    
    print(logging_tips)
    
    
