# This script contains plot utilities

from PIL import Image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from GbAgent import Agent
from GbAgentVars import AgentVars

def cm2inch(*tupl):
    """ This function convertes cm to inches

    Obtained from: https://stackoverflow.com/questions/14708695/
    specify-figure-size-in-centimeter-in-matplotlib/22787457
    :param tupl: Size of plot in cm
    :return: Converted image size in inches
    """

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def center_x(cell_lower_left_x, cell_width, word_length):
    """ This function centers text along the x-axis

    :param cell_lower_left_x: Lower left x-coordinate
    :param cell_width: Width of cell in which text appears
    :param word_length: Length of plotted word
    :return: Centered x-position
    """

    return cell_lower_left_x + (cell_width / 2.0) - (word_length / 2.0)


def center_y(cell_lower_left_y, cell_height, y0, word_height):
    """ This function centers text along the y-axis

    :param cell_lower_left_y: Lower left y-coordinate
    :param cell_height: Height of cell in which text appears
    :param y0: Lower bound of text (sometimes can be lower than cell_lower_left-y (i.e. letter y))
    :param word_height: Height of plotted word
    :return: Centered y-position
    """

    return cell_lower_left_y + ((cell_height / 2.0) - y0) - (word_height / 2.0)


def get_text_coords(f, ax, cell_lower_left_x, cell_lower_left_y, printed_word, fontsize):
    """ This function computes the length and height of a text und consideration of the font size

    :param f: Figure object
    :param ax: Axis object
    :param cell_lower_left_x: Lower left x-coordinate
    :param cell_lower_left_y: Lower left y-coordinate
    :param printed_word: Text of which length is computed
    :param fontsize: Specified font size
    :return: word_length, word_height, bbox: Computed word length and height and text coordinates
    """

    # Print text to lower left cell corner
    t = ax.text(cell_lower_left_x, cell_lower_left_y, printed_word, fontsize=fontsize)

    # Get text coordinates
    f.canvas.draw()
    bbox = t.get_window_extent().inverse_transformed(ax.transData)
    word_length = bbox.x1 - bbox.x0
    word_height = bbox.y1 - bbox.y0

    # Remove printed word
    t.set_visible(False)

    return word_length, word_height, bbox


def plot_centered_text(f, ax, cell_x0, cell_y0, cell_x1, cell_y1,
                       text, fontsize, fontweight='normal', c_type='both'):
    """ This function plots centered text

    :param f: Figure object
    :param ax: Axis object
    :param cell_x0: Lower left x-coordinate
    :param cell_y0: Lower left y-coordinate
    :param cell_x1: Lower right x-coordinate
    :param cell_y1: Lower upper left y-coordinate
    :param text: Printed text
    :param fontsize: Current font size
    :param fontweight: Current font size
    :param c_type: Centering type (y: only y axis; both: both axes)
    :return: ax, word_length, word_height, bbox: Axis object, length and height of printed text, text coordinates
    """

    # Get text coordinates
    word_length, word_height, bbox = get_text_coords(f, ax, cell_x0, cell_y0,
                                                     text, fontsize)

    # Compute cell width and height
    cell_width = (cell_x1 - cell_x0)
    cell_height = (cell_y1 + cell_y0)

    # Compute centered x position: lower left + half of cell width, then subtract half of word length
    # x = cell_lower_left_x + (cell_width / 2.0) - (word_length / 2.0)
    x = center_x(cell_x0, cell_width, word_length)

    # Compute centered y position: same as above but additionally correct for word height
    # (because some letters such as y start below y coordinate)
    # y = cell_lower_left_y + ((cell_height / 2.0) - bbox.y0) - (word_height / 2.0)
    y = center_y(cell_y0, cell_height, bbox.y0, word_height)

    # Print centered text
    if c_type == 'both':
        ax.text(x, y, text, fontsize=fontsize, fontweight=fontweight)
    else:
        ax.text(cell_x0, y, text, fontsize=fontsize, fontweight=fontweight)

    return ax, word_length, word_height, bbox


def plot_image(f, img_path, cell_x0, cell_x1, cell_y0, ax, text_y_dist, text, text_pos, fontsize,
               zoom=0.2, cell_y1=np.nan):
    """ This function plots images and corresponding text for the task schematic

    :param f: Figure object
    :param img_path: Path of image
    :param cell_x0: Left x-position of area in which it is plotted centrally
    :param cell_x1: Rigth x-position of area in which it is plotted centrally
    :param cell_y0: Lower y-position of image -- if cell_y1 = nan
    :param ax: Plot axis
    :param text_y_dist: y-position distance to image
    :param text: Displayed text
    :param text_pos: Position of printed text (below vs. above)
    :param fontsize: Text font size
    :param zoom: Scale of image
    :param cell_y1: Upper x-position of area in which image is plotted (lower corresponds to cell_y0)
    :return ax, bbox: Axis object, image coordinates
    """

    # Open image
    img = Image.open(img_path)

    # Image zoom factor and axis and coordinates
    imagebox = OffsetImage(img, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, (cell_x0, cell_y0), xybox=None,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0, frameon=False)
    ax.add_artist(ab)

    # Get cell width
    cell_width = cell_x1 - cell_x0
    image_x = cell_x0 + (cell_width/2)

    if not np.isnan(cell_y1):
        cell_height = cell_y1 - cell_y0
        image_y = cell_y0 + (cell_height / 2)
    else:
        image_y = cell_y0

    # Remove image and re-plot at correct coordinates
    ab.remove()
    ab = AnnotationBbox(imagebox, (image_x, image_y), xybox=None,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0, frameon=False)
    ax.add_artist(ab)

    # Get image coordinates
    f.canvas.draw()
    renderer = f.canvas.renderer
    # bbox = imagebox.get_window_extent(renderer).inverse_transformed(ax.transAxes)
    bbox = imagebox.get_window_extent(renderer).inverse_transformed(ax.transData)

    if text_pos == 'left_below':
        # Plot text below image
        x = bbox.x0
        y = bbox.y0 - text_y_dist
    elif text_pos == 'centered_below':
        # Plot text centrally above image
        word_length, _, _ = get_text_coords(f, ax, bbox.x0, bbox.y0, text, 6)
        cell_width = bbox.x1 - bbox.x0
        x = center_x(bbox.x0, cell_width, word_length)
        y = bbox.y0 - text_y_dist
    else:
        # Plot text centrally above image
        word_length, _, _ = get_text_coords(f, ax, bbox.x0, bbox.y0, text, 6)
        cell_width = bbox.x1 - bbox.x0
        x = center_x(bbox.x0, cell_width, word_length)
        y = bbox.y1 + text_y_dist

    ax.text(x, y, text, fontsize=fontsize, color='k')

    return ax, bbox


def plot_arrow(ax, x1, y1, x2, y2, shrink_a=1, shrink_b=1, connectionstyle="arc3,rad=0", arrow_style="<-"):
    """ This function plot arrows for the task schematic

    :param ax: Axis object
    :param x1: x-position of starting point
    :param y1: y-position of starting point
    :param x2: x-position of end point
    :param y2: y-position of end point
    :param shrink_a: Degree with which arrow is decrasing at starting point
    :param shrink_b: Degree with which arrow is decrasing at end point
    :param connectionstyle: Style of connection line
    :param arrow_style: Style of arrow
    :return ax: Axis object
    """

    ax.annotate("", xy=(x1, y1), xycoords='data', xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle=arrow_style, color="0.5", shrinkA=shrink_a, shrinkB=shrink_b,
                                patchA=None, patchB=None, connectionstyle=connectionstyle))

    return ax


def plot_rec(ax, patches, cell_lower_left_x, width, cell_lower_left_y, height):
    """ This function plots a rectangle

    :param ax: Axis object
    :param patches: Patches object
    :param cell_lower_left_x: Lower left corner x coordinate of rectangle
    :param width: Width of rectangle
    :param cell_lower_left_y: Lower left corner y coordinate of rectangle
    :param height: Height of rectangle
    :return: Axis object
    """

    p = patches.Rectangle(
        (cell_lower_left_x, cell_lower_left_y), width, height,
        fill=False, transform=ax.transAxes, clip_on=False, linewidth=0.5)

    ax.add_patch(p)

    return ax


def plot_table(ax, n_rows=8, n_cols=4, col_header_line=0.1):
    """

    :param ax: Axis object
    :param n_rows: Number of rows
    :param n_cols: Number of columns
    :param col_header_line: Size of column header cells
    :return: row_lines, col_lines: Table lines
    """

    # Create grid lines
    row_lines = np.linspace(0, 1, n_rows + 1)
    col_lines = np.linspace(col_header_line, 1, n_cols)

    # Cycle over columns lines
    for i in range(0, len(col_lines)-1):
        ax.axvline(ymin=0, ymax=1, x=col_lines[i], color='k', linewidth=0.5, alpha=1)

    # Cycle over row lines
    for i in range(1, len(row_lines)-1):
        ax.axhline(xmin=0, xmax=1, y=row_lines[i], color='k', linewidth=0.5, alpha=1)

    # Add columns header line
    col_lines = np.concatenate((0, col_lines), axis=None)

    return row_lines, col_lines


def label_subplots(f, texts, x_offset=0.07, y_offset=0.015):
    """ This function labels the subplots

     Obtained from: https://stackoverflow.com/questions/52286497/
     matplotlib-label-subplots-of-different-sizes-the-exact-same-distance-from-corner

    :param f: Figure handle
    :param x_offset: Shifts labels on x-axis
    :param y_offset: Shifts labels on y-axis
    :param texts: Subplot labels
    """

    # Get axes
    axes = f.get_axes()

    # Cycle over subplots and place labels
    for a, l in zip(axes, texts):
        x = a.get_position().x0
        y = a.get_position().y1
        f.text(x - x_offset, y + y_offset, l, size=12)


def bs_plot(ax, agent, c_set, color_ind, sigma, scalar_map, which_state=0):
    """ This function plots pi_1 as a function of c_t

    :param ax: Axis object
    :param agent: Agent object
    :param c_set: Set of contrasts
    :param color_ind: Color indices
    :param sigma: Perceptual sensitivity parameter
    :param scalar_map: Color map
    :param which_state: Belief state that is plotted
    :return: Axis object
    """

    # Initialize counter for colors
    counter = 0

    # Cycle over observations
    for i in range(0, len(c_set)):

        # Current color
        if i < 12:
            color = scalar_map.to_rgba(color_ind[0])
        elif i >= 36:
            color = scalar_map.to_rgba(color_ind[-1])
        else:
            color = scalar_map.to_rgba(color_ind[counter])
            counter += 1

        # Compute and plot belief state for high sensitivity condition
        agent.sigma = sigma
        pi_0, pi_1 = agent.p_s_giv_o(c_set[i])

        if which_state == 0:
            ax.plot(c_set[i], pi_0, '.', color=color, markersize=2)
        else:
            ax.plot(c_set[i], pi_1, '.', color=color, alpha=0.3, markersize=2)

    return ax


def plot_observations(ax_0, ax_1, scalar_map, c_t, color_ind, sigma, x, x_lim, labels="full"):
    """ This function plots the observation illustration

    :param ax_0: First axis object
    :param ax_1: Second axis object
    :param scalar_map: Color map
    :param c_t: Contrast differences
    :param color_ind: Color indices
    :param sigma: Perceptual sensitivity parameter
    :param x: x-axis
    :param x_lim: x limits
    :param labels: Label style (full vs. reduced)
    :return: ax_0, ax_1: Axis objects
    """

    # Cycle over contrast differences
    for i in range(0, len(c_t)):

        # Plot contrast differences
        # ---------------------------

        # Current color
        cval = scalar_map.to_rgba(color_ind[i])

        # Low noise
        _, stemlines, _ = ax_0.stem(c_t[i], [1], markerfmt=" ", use_line_collection=True)
        plt.setp(stemlines, linewidth=0.5, color=cval)  # set stem width and color

        # Plot distributions over the observations
        # ----------------------------------------

        # High sensitivity
        fit = stats.norm.pdf(x, c_t[i], sigma)
        ax_1.plot(x, fit, '-', linewidth=0.5, color=cval)

    if labels == "full":

        # Adjust contrast difference plot
        ax_0.set_ylim(0, 1)
        ax_0.tick_params(labelsize=6)
        ax_0.tick_params(axis='both', which='major', labelsize=6)
        ax_0.set_xlim(-x_lim, x_lim)
        ax_0.axes.get_yaxis().set_ticks([])
        ax_0.text(-0.1, -0.8, '"left"', size=8, rotation=0, color='grey', ha="center", va="center")
        ax_0.text(0.1, -0.8, '"right"', size=8, rotation=0, color='grey', ha="center", va="center")

        # Adjust observation plot
        ax_1.set_ylim(0, 25)
        ax_1.set_xlim(-x_lim, x_lim)
        ax_1.axes.get_yaxis().set_ticks([])
        ax_1.set_ylabel('Observation\nprobability', fontsize=6)

    else:

        # Adjust contrast difference plot
        ax_0.set_ylim(0, 1)
        ax_0.tick_params(labelsize=6)
        ax_0.set_xlim(-x_lim, x_lim)
        ax_0.axes.get_yaxis().set_ticks([])

        # Adjust observation plot
        ax_1.set_ylim(0, 25)
        ax_1.tick_params(labelsize=6)
        ax_1.set_xlim(-x_lim, x_lim)
        ax_1.axes.get_yaxis().set_ticks([])

    return ax_0, ax_1


def plot_agent_demo(ax_0, ax_1, ax_2, ax_3, ax_4, x, df_sim, header, agent_color, markersize=1, fontsize=8,
                    labels='full', bs='normal'):
    """ This function plots the agent demonstration

    :param ax_0: Zeroth axis object
    :param ax_1: Second axis object
    :param ax_2: Third axis object
    :param ax_3: Fourth axis object
    :param ax_4: Fifth axis object
    :param x: x-axis
    :param df_sim: data frame with simulated data
    :param header: Plot headers
    :param agent_color: Agent color code
    :param markersize: Marker size
    :param fontsize: Font size
    :param labels: Axis label style (full vs. reduced)
    :param bs: belief state style (normal vs. categorical)
    """

    plot_range = np.arange(25)

    # State
    # -----
    ax_0.plot(x, df_sim['s_t'][plot_range], 'o', color='k', markersize=markersize)
    ax_0.set_title(header, fontsize=fontsize)
    ax_0.set_ylim([-0.2, 1.2])
    ax_0.tick_params(labelsize=fontsize)
    ax_0.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    # Presented contrast difference
    # -----------------------------
    ax_1.bar(x, df_sim['u_t'][plot_range], color='k', width=0.2)
    ax_1.set_ylim([-0.1, 0.1])
    ax_1.tick_params(labelsize=fontsize)
    ax_1.axhline(0, color='black', lw=0.5, linestyle='--')
    ax_1.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    # Belief state
    # ------------
    if bs == 'normal':
        ax_2.bar(x, df_sim['pi_1'][plot_range], color=agent_color, width=0.2)
    else:
        ax_2.bar(x, df_sim['d_t'][plot_range], color=agent_color, width=0.2)
    ax_2.tick_params(labelsize=fontsize)
    ax_2.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    ax_2.set_ylim(0, 1)

    # Expected values and economic choice
    # -----------------------------------

    # Select expected value for current economic choice
    which_choice = df_sim['a_t'][plot_range]
    sel_v_a_0 = df_sim['v_a_0'][plot_range]
    sel_v_a_1 = df_sim['v_a_1'][plot_range]
    selected_ev = pd.concat([sel_v_a_0[which_choice == 0], sel_v_a_1[which_choice == 1]]).sort_index()

    ax_3.plot(x, selected_ev, markersize=markersize, color=agent_color, linestyle='-', label='HHZ 1')
    ax_3.axhline(0.5, color='black', lw=0.5, linestyle='--')
    ax_3.tick_params(labelsize=fontsize)

    # Economic decision
    #ax_3.plot(x[df_sim['a_t'] == 0], df_sim['a_t'][df_sim['a_t'] == 0], 'o', color='k', markersize=markersize)
    #ax_3.plot(x[df_sim['a_t'] == 1], df_sim['a_t'][df_sim['a_t'] == 1], 'o', color='k', markersize=markersize)
    ax_3.plot(x[which_choice == 0], which_choice[which_choice == 0], 'o', color='k', markersize=markersize)
    ax_3.plot(x[which_choice == 1], which_choice[which_choice == 1], 'o', color='k', markersize=markersize)
    ax_3.set_ylim([-0.2, 1.2])
    ax_3.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    # Reward
    # ------
    ax_4.plot(x, df_sim['r_t'][plot_range], 'ko', markersize=markersize)
    ax_4.set_ylim([-0.2, 1.2])
    ax_4.tick_params(labelsize=fontsize)

    # Contingency parameter
    # ----------------------------

    # Compute mean expected values
    mean_corr_a1 = df_sim.groupby(df_sim['t'])['e_mu_t'].mean()

    # Plot all expected values
    # ax_0.axhline(0.5, color='black', lw=0.5, linestyle='--')
    # ax_0.axhline(0.8, color='black', lw=0.5, linestyle='--')
    for _, group in df_sim.groupby('block'):
        group.plot(x='t', y='e_mu_t', ax=ax_4, legend=False, color='gray', linewidth=1, alpha=0.2)

    # ax_0.set_ylim([0.2, 1])
    # ax_0.tick_params(labelsize=fontsize)
    # ax_0.set_xlabel(r'Trial $(t)$', fontsize=fontsize)
    # ax_0.set_ylabel(r'Expected Value ($E_{\mu}$)', fontsize=fontsize)
    ax_4.plot(x, mean_corr_a1, linewidth=1, color=agent_color, linestyle='--')
    # ax_0.set_title('Belief-State weighted Bayesian learning (A1)', fontsize=8)
    ax_4.plot(x, df_sim['e_mu_t'][plot_range], markersize=markersize, color=agent_color)
    ax_4.axhline(0.8, color='black', lw=0.5, linestyle='--')
    ax_4.set_xlabel('Trial', fontsize=fontsize)
    ax_4.set_ylim([0.5, 1.1])

    # Adjust labels
    # -------------
    if labels == 'reduced':
        ax_0.set_yticklabels('')
        ax_1.set_yticklabels('')
        ax_2.set_yticklabels('')
        ax_3.set_yticklabels('')
        ax_4.set_yticklabels('')

    else:
        ax_0.set_ylabel('State', fontsize=fontsize)
        ax_1.set_ylabel('Contrast\ndifference', fontsize=fontsize)
        ax_2.set_ylabel('Belief\nstate')
        ax_3.set_ylabel('Ev and \nchoice')
        ax_4.set_ylabel('Reward and\nlearning')


def get_bic(df_bic, div):
    """ This function computes the summed BIC's for the model recovery plot

    :param df_bic: Data frame containing BIC
    :param div: Divisor to adjust y-axis
    :return:
    """

    # Extract BIC for perceptual and economic choices
    bic_d = df_bic.groupby(['agent'])['d_BIC'].sum()
    bic_a = df_bic.groupby(['agent'])['a_BIC'].sum()

    # Sum BIC's and divide
    bic_ag_0 = (bic_d[0] + bic_a[0]) / div
    bic_ag_1 = (bic_d[1] + bic_a[1]) / div
    bic_ag_2 = (bic_d[2] + bic_a[2]) / div
    bic_ag_3 = (bic_d[3] + bic_a[3]) / div
    bic_ag_4 = (bic_d[4] + bic_a[4]) / div
    bic_ag_5 = (bic_d[5] + bic_a[5]) / div
    bic_ag_6 = (bic_d[6] + bic_a[6]) / div

    return bic_ag_0, bic_ag_1, bic_ag_2, bic_ag_3, bic_ag_4, bic_ag_5, bic_ag_6

def plot_pmu(agent, bs0, bs1, color_ind, s_map, current_ax, plot_leg=True):
    """ This function illustrates evolution of p(mu) for given constant belief states

    :param agent_obj: Agent object instance
    :param bs0: p(s_t=0|c_t)
    :param bs1: p(s_t=1|c_t)
    :param color_ind: cval_ind
    :param s_map: scalar_map
    :param current_ax: Current axis
    :return: agent_obj, current_ax
    """

    #agent_vars = AgentVars()
    #agent = Agent(agent_vars)

    n = 10

    coeff = np.array([1])
    mu = np.linspace(0, 1, 101)
    p_mu = np.full([101, 10], np.nan)
    v_a_t = np.full([n, 2], np.nan)
    p_mu[:, 0] = np.polyval(np.array([1]), mu)

    # Plot analytical solution
    current_ax.plot(mu, p_mu, color='k')
    agent.c_t = coeff
    v_a_t[0, :] = agent.compute_valence(1, 0)

    update = np.full(n, np.nan)
    pe = np.full(n, np.nan)
    pi_rel = np.full(n, np.nan)

    pe[0] = 0.5

    for i in range(1, n):

        if bs0 >= bs1:
            pe[i] = 1 - agent.E_mu_t
        else:
            pe[i] = 0 - agent.E_mu_t

        prev_ev = agent.E_mu_t

        agent.a_t = np.float(0)

        agent.q_0, agent.q_1 = agent.compute_q(np.float(1), np.float(bs0), np.float(bs1))

        #agent.t = agent.c_t.size + 1
        #coeff = agent.update_coefficients()
        agent.update_coefficients()
        #agent.c_t = coeff
        p_mu[:, i] = np.polyval(agent.c_t, mu)

        current_ax.plot(mu, p_mu[:, i], color=s_map.to_rgba(color_ind[i]))
        #current_ax.plot(mu, p_mu[:, i])

        v_a_t[i, :] = agent.compute_valence(1, 0)

        update[i] = agent.E_mu_t - prev_ev

        pi_rel[i] = -1**2 * (bs0 - bs1) * (bs1 * bs0)

    if plot_leg:
        legend_elements = [Line2D([0], [0], color=s_map.to_rgba(color_ind[0]), lw=1, label=r'$t=0$'),
                           Line2D([0], [0], color=s_map.to_rgba(color_ind[1]), lw=1, label=r'$t=1$'),
                           Line2D([0], [0], color=s_map.to_rgba(color_ind[2]), lw=1, label=r'$t=2$'),
                           Line2D([0], [0], color=s_map.to_rgba(color_ind[3]), lw=1, label=r'$t=3$'),
                           Line2D([0], [0], color=s_map.to_rgba(color_ind[4]), lw=1, label=r'$t=4$'),
                           Line2D([0], [0], color=s_map.to_rgba(color_ind[5]), lw=1, label=r'$t=5$'),
                           Line2D([0], [0], color=s_map.to_rgba(color_ind[6]), lw=1, label=r'$t=6$'),
                           Line2D([0], [0], color=s_map.to_rgba(color_ind[7]), lw=1, label=r'$t=7$'),
                           Line2D([0], [0], color=s_map.to_rgba(color_ind[8]), lw=1, label=r'$t=8$'),
                           Line2D([0], [0], color=s_map.to_rgba(color_ind[9]), lw=1, label=r'$t=9$')]

        current_ax.legend(handles=legend_elements)

    return agent, current_ax, v_a_t, update, pe, pi_rel