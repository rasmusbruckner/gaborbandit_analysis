def latex_plt(matplotlib):
    """ This function updates the matplotlib library to use Latex and changes some default plot parameters

    :param matplotlib: Matplotlib instance
    :return: Updated matplotlib instance
    """

    # Use Latex for matplotlib
    pgf_with_latex = {
        "font.family": "serif",
        "font.sans-serif": [],
        "axes.labelsize": 6,
        "font.size": 6,
        "legend.fontsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.titlesize": 6,
        "pgf.rcfonts": False,
        "figure.dpi": 100,
        "text.latex.unicode": True,
        "pgf.preamble": [
             r"\usepackage[utf8x]{inputenc}",
             r"\usepackage[T1]{fontenc}",
             r"\usepackage{cmbright}",
             ]
    }

    # Update parameters
    matplotlib.rcParams.update(pgf_with_latex)

    return matplotlib
