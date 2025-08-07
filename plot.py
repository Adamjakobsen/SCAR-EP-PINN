import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import *
import numpy as np
import matplotlib.animation as animation
import matplotlib.tri as tri
from matplotlib.patches import Rectangle
import numpy.ma as ma
import seaborn as sns
import os
import pandas as pd

EXPERIMENT_GILS = np.linspace(0.01, 0.1, 19)
DATA_SPLIT = {"train": [0.01, 0.015, 0.02, 0.03, 0.035, 0.04, 0.05, 0.055, 0.06, 0.07, 0.075, 0.08],
              "interpolation": [0.025, 0.045, 0.065],
              "extrapolation": [0.085, 0.09, 0.095, 0.1]}

def expernums_from_gils(gil_vals):
    return [np.where(np.isclose(EXPERIMENT_GILS, gil))[0][0] for gil in gil_vals]

def get_dataset_color(gil):
    if np.isclose(gil, DATA_SPLIT["train"]).any():
        return 'g'
    elif np.isclose(gil, DATA_SPLIT["interpolation"]).any():
        return 'b'
    elif np.isclose(gil, DATA_SPLIT["extrapolation"]).any():
        return 'r'
    else:
        raise Exception(f"GIL {gil} not found in data split.")

def get_dataset_label(gil):
    if np.isclose(gil, DATA_SPLIT["train"]).any():
        return "Training"
    elif np.isclose(gil, DATA_SPLIT["interpolation"]).any():
        return "Interpolation"
    elif np.isclose(gil, DATA_SPLIT["extrapolation"]).any():
        return "Extrapolation"
    else:
        raise Exception(f"GIL {gil} not found in data split.")

def get_gstr(gil):
    return rf"$\mathbf{{g_{{il}}^s}}={gil:.3f}\,\mathbf{{S/m}}$"

def set_plot_env():
    """
    Function for setting the plotting environment equal for all plots
    """
    context = {
        'font.size': 25.0,
        'axes.labelsize': 25.0,
        'axes.titlesize': 25.0,
        'xtick.labelsize': 25.0,
        'ytick.labelsize': 25.0,
        'legend.fontsize': 25.0,
        'legend.title_fontsize': None,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 6.0,
        'patch.linewidth': 1.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0
    }

    plt.rcParams['text.usetex'] = False
    sns.set_theme(context=context, style="ticks",
                  palette="colorblind", font="sans-serif", font_scale=1)

def add_cbar_to_plotgrids(fig, axs, pcm, label):
    fig.subplots_adjust(
        left=0.20,
        right=0.88,
        top=0.9,    # ⬅️ push the subplots down to 75% of the figure
        bottom=0.1,
        hspace=0.001,
        wspace=0.15
    )
    # now add the horizontal colorbar *over* the space you freed up:
    cb = fig.colorbar(
        pcm,
        ax=axs,
        orientation='horizontal',
        location='bottom',    # requires mpl ≥3.2
        fraction=0.05,     # bar thickness
        pad=0.07,          # small pad from the top of the subplot area
        label=label, #r"$AT$ [ms]"
    )

    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.xaxis.set_label_position('bottom')

class MeshPlotData(object):
    def __init__(self, cmap, vmin, vmax, cbar_label):

        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.cbar_label = cbar_label

        self.vertices, self.triangles, _ = get_data()
        self.vertices *= 1e-3

        self.triang = tri.Triangulation(self.vertices[:,0], 
                                        self.vertices[:,1],
                                        self.triangles)

class VMMeshPlotData(MeshPlotData):
    def __init__(self, t_plot):
        super().__init__('cividis', -80, 20, r"$V_m$ [mV]")

        self.vm_FEM   = np.load("./data/vm_variation_scar_full.npy")            # (T, V, E)
        self.vm_PIMM = np.load("./data/vm_PINN.npy")            # (T, V, E)
        #cond_all = np.load("./data/conductivities_variation_scar_full.npy")
        #cond_all *= 12.9/1.4

        dt = 5
        T  = self.vm_FEM.shape[0]
        t  = np.linspace(0, (T-1)*dt, T)/12.9

        self.frame = int(t_plot/dt)
        if self.frame < 0 or self.frame >= T:
            raise ValueError("t_plot out of range")

        self.t_plot = t_plot

    def get_data(self, exp_idx):
        # FEM solution at t_plot
        v_fem = self.vm_FEM[self.frame, :, exp_idx]

        # PINN prediction
        v_pinn = self.vm_PIMM[self.frame, :, exp_idx]
        return v_fem, self.triang, v_pinn, self.triang

class ActMeshPlotData(MeshPlotData):
    def __init__(self):
        self.activations_all_FEM = np.load("./data/activations_all_FEM.npy")
        self.activations_all_PINN = np.load("./data/activations_all_PINN.npy")

        max_activation_time = np.nanmax(self.activations_all_FEM)
        
        super().__init__('plasma', 
                          0,
                          max_activation_time,
                          r"$AT$ [ms]")

    def get_data(self, exp_idx):

        pinn_activation_time = self.activations_all_PINN[:, exp_idx]
        fem_activation_time = self.activations_all_FEM[:, exp_idx]

        fem_activation_time_masked = ma.masked_invalid(fem_activation_time)
        pinn_activation_time_masked = ma.masked_invalid(pinn_activation_time)
        
        triang_pinn = tri.Triangulation(self.vertices[:, 0], self.vertices[:, 1], self.triangles)
        triang_fem = tri.Triangulation(self.vertices[:, 0], self.vertices[:, 1], self.triangles)
        
        tri_mask_fem = np.any(np.isnan(fem_activation_time[self.triang.triangles]), axis=1)
        tri_mask_pinn = np.any(np.isnan(pinn_activation_time[self.triang.triangles]), axis=1)
        
        triang_pinn.set_mask(tri_mask_pinn)
        triang_fem.set_mask(tri_mask_fem)

        return fem_activation_time_masked, triang_fem, pinn_activation_time_masked, triang_pinn

def plot_mesh_boundary(ax):
    boundary, lines = get_boundary()
    boundary = boundary * 1e-3
    x_boundary = boundary[:, 0]
    y_boundary = boundary[:, 1]

    lines_array = np.array(boundary[lines])
    x_coords = lines_array[:, :, 0] 
    y_coords = lines_array[:, :, 1]  

    ax.plot(x_coords.T, y_coords.T, linewidth=3, color="k")

def plot_meshdata_vertical(plotdata, 
                          gils_plot,
                          filename,
                          zoom = None,
                          figsize = (15, 24),
                          trace_meshboundary = False) -> None:
    set_plot_env()

    g_il_values = EXPERIMENT_GILS
    expernums = expernums_from_gils(gils_plot)

    n_exp = len(expernums)

    # create n_exp × 2 grid, share axes
    fig, axs = plt.subplots(nrows=n_exp,
                            ncols=2, 
                            figsize = figsize,
                            sharex=True, 
                            sharey=True,
                            gridspec_kw={'hspace': 0.001})
    
    # reduce vertical whitespace between rows
    set_plot_env()
    
    indicies = range(n_exp)
    
    for i, exp_idx, gil in zip(indicies, expernums, gils_plot):
        
        data_fem, triang_fem, data_pinn, triang_pinn = plotdata.get_data(exp_idx)
        
        # left: PINN
        ax = axs[i,0]
        pcm = ax.tripcolor(triang_pinn,
                           data_pinn,
                           shading='gouraud', 
                           cmap = plotdata.cmap,
                           vmin = plotdata.vmin, 
                           vmax = plotdata.vmax,
                           rasterized = True)
        
        ax.set_aspect('equal')
        if i == 0:
            ax.set_title("PINN",
                         fontsize = 1.5 * plt.rcParams['axes.labelsize'],
                         fontweight ='bold')
 
        # right: FEM
        ax2 = axs[i,1]
        ax2.tripcolor(triang_fem, 
                      data_fem,
                      shading='gouraud',
                      cmap = plotdata.cmap,
                      vmin = plotdata.vmin, 
                      vmax= plotdata.vmax,
                      rasterized = True)

        ax2.set_aspect('equal')
        if i == 0:
            ax2.set_title("FEM",
                          fontsize=1.5 * plt.rcParams['axes.labelsize'],
                          fontweight='bold')

        if trace_meshboundary:
            plot_mesh_boundary(ax)
            plot_mesh_boundary(ax2)

        # hide spines
        for aa in (ax, ax2):
            for sp in aa.spines.values():
                sp.set_visible(False)
                
                if i == n_exp-1:
                    axs[i,0].spines['bottom'].set_visible(True)
                    axs[i,1].spines['bottom'].set_visible(True)
                    
        axs[i,0].spines['left'].set_visible(True)
        # vertical row-label: bold g_il including units
        # 1) draw the y-axis label in default color
        ax.set_ylabel("y [mm]",
                      fontsize=20,
                      rotation=90,
                      labelpad=30,
                      color="black",
                      va="center")

        # decide color based on category
        txt_color = get_dataset_color(gil)

        # bold, colored g_il text
        g_str = get_gstr(gil)
        ax.text(x =-0.40,
                y = 0.5,
                s = g_str,
                transform = ax.transAxes,
                fontsize = 1.2*plt.rcParams['axes.labelsize'],
                va="center", 
                ha = "center",
                rotation = 90,
                color=txt_color)

        # ticks: y only on left column; x only on bottom row
        ax.tick_params(left=True,
                       labelleft=True,
                       bottom=(i==n_exp-1),
                       labelbottom=(i==n_exp-1)
        )
        ax2.tick_params(left=False,
                        labelleft=False,
                        bottom=(i==n_exp-1),
                        labelbottom=(i==n_exp-1)
        )

        # x-axis labels on bottom row
        if i == n_exp-1:
            ax.set_xlabel("x [mm]", fontsize=20)
            ax2.set_xlabel("x [mm]", fontsize=20)

        if zoom:
            plt.ylim(zoom[0],zoom[1])
            plt.xlim(zoom[2],zoom[3])
        else:
            pad = 1
            ymin = plotdata.vertices[:,1].min()
            ymax = plotdata.vertices[:,1].max()
            if trace_meshboundary:
                ymin -= pad
                ymax += pad

            plt.ylim(ymin, ymax)

    add_cbar_to_plotgrids(fig, axs, pcm, plotdata.cbar_label)

    plt.savefig(f"./results/{filename}",
                dpi=300)
    plt.close(fig)

def get_vm_errordata():
    g_il_values = EXPERIMENT_GILS
    v_FEM_all = np.load("./data/vm_variation_scar_full.npy")
    v_PINN_all = np.load("./data/vm_PINN.npy")
    errors = np.zeros(len(g_il_values))
    for g_il_index, g_il in enumerate(g_il_values):
        v_PINN = v_PINN_all[:, :, g_il_index]
        v_FEM = v_FEM_all[:, :, g_il_index]
        # compute absolute error
        abs_error = np.abs(v_PINN - v_FEM)
        # compute relative error
        rel_error = np.sum(abs_error) / np.sum(np.abs(v_FEM))
        errors[g_il_index] = rel_error
    return errors

def calculate_inactivation_dice_coefficient(isnanFEM, isnanPINN):
    intersection = np.sum(np.logical_and(isnanFEM, isnanPINN), axis=0)
    sum_isnan = np.sum(isnanFEM, axis=0) + np.sum(isnanPINN, axis=0)
    dice_coefficient = 2 * intersection / sum_isnan
    return dice_coefficient

def get_act_errordata():
    activations_all_FEM = np.load("./data/activations_all_FEM.npy")
    activations_all_PINN = np.load("./data/activations_all_PINN.npy")
    
    isnanFEM = np.isnan(activations_all_FEM)
    isnanPINN = np.isnan(activations_all_PINN)
    
    inact_size_FEM =  isnanFEM.sum(axis= 0)
    inact_size_PINN = isnanPINN.sum(axis = 0)

    inact_dice = calculate_inactivation_dice_coefficient(isnanFEM, isnanPINN)

    inactivation_df = pd.DataFrame(np.array([EXPERIMENT_GILS,
                                             inact_size_FEM,
                                             inact_size_PINN,
                                             inact_dice]).T,
                                             columns = ["$g_{il}^s$", "FEM", "PINN", "Dice"])

    notnull_FEMPINN = np.logical_not(isnanPINN | isnanFEM)
    
    act_FEM = np.zeros(activations_all_FEM.shape)
    act_FEM[notnull_FEMPINN] = activations_all_FEM[notnull_FEMPINN]

    act_PINN = np.zeros(activations_all_PINN.shape)
    act_PINN[notnull_FEMPINN] = activations_all_PINN[notnull_FEMPINN]

    # Calculate relative absolute errors for each row
    relabs_errors = np.sum(np.abs(act_PINN - act_FEM), axis=0)
    relabs_errors /= np.sum(np.abs(act_FEM), axis=0)
    
    return relabs_errors, inactivation_df

def plot_errorbars_horizontal(xlabel, filename):
    set_plot_env()
    g_il_values = EXPERIMENT_GILS
    
    bar_height = 0.005

    fig, ax = plt.subplots(figsize=(12, 24))

    count_train = count_interp = count_extra = 0
    
    # Collect all data for a single bar chart
    bar_colors = [get_dataset_color(g_il) for g_il in g_il_values]
    bar_labels = [get_dataset_label(g_il) for g_il in g_il_values]

    # Remove the right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Align the x-axis with the bottom bar
    ax.set_ylim(g_il_values[0] - bar_height / 2, g_il_values[-1] + bar_height / 2)

    # Plot all bars in a single call
    ax.barh(
        g_il_values,  # y positions
        errors,       # bar widths
        color=bar_colors,
        height=bar_height)

    ax.set_yticks(g_il_values[::2])
    ax.tick_params(axis='y', rotation=0)

    ax.set_xlabel(xlabel, fontsize=1.5 * plt.rcParams['axes.labelsize'])
    ax.set_ylabel(r'$g_{il}^s$ [S/m]')

    #Unique legend
    # Manually create legend with unique labels
    unique_labels_colors = dict(zip(bar_labels, bar_colors))  # Ensure unique labels
    legend_handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in unique_labels_colors.items()]
    ax.legend(handles=legend_handles, loc='center right')

    path = os.path.join("./results/", filename)
    plt.savefig(path, bbox_inches='tight')

def plot_fem_data(t_plot, gil_vals, filename, zoom) -> None:
    expernums = expernums_from_gils(gil_vals)
    vertices, triangles, _ = get_data()
    # scar_vertices = np.load("./data/gregion3_coordinates.npy")
    vertices = vertices * 1e-3
    # scar_vertices = scar_vertices * 1e-3

    vm_all = np.load("./data/vm_variation_scar_full.npy")
    activation_FEM_all = np.load("./data/activations_all_FEM.npy")
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    n_timepoints = vm_all.shape[0]
    dt = 5
    t_start = 0
    t_end = (n_timepoints - 1) * dt
    t = np.linspace(t_start, t_end, n_timepoints) / \
        12.9

    boundary, _ = get_boundary()
    boundary = boundary * 1e-3
    x_boundary = boundary[:, 0]
    y_boundary = boundary[:, 1]

    # Create the figure and subplots
    fig, axs_grid = plt.subplots(
        2, 4, figsize=(26, 12), sharex=True, sharey=True)
    axs = axs_grid.flatten()
    max_activation_time = np.nanmax(activation_FEM_all)
    
    positions = range(len(gil_vals))

    for i, expernum, gil in zip(positions, expernums, gil_vals):
        fem_activation_time = activation_FEM_all[:, expernum]

        # Triangulate and mask invalid data for FEM and PINN
        triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
        triang_v = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

        fem_activation_time_masked = ma.masked_invalid(fem_activation_time)

        tri_mask_fem = np.any(
            np.isnan(fem_activation_time[triang.triangles]), axis=1)

        triang.set_mask(tri_mask_fem)

        ax_AT = axs_grid[0, :].flatten()[i]
        tc_fem = ax_AT.tricontourf(triang, fem_activation_time_masked, levels=15,
                                   cmap='plasma', vmin=0, vmax=max_activation_time)
        if i == 0:
            tc_fem0 = tc_fem

        ax_AT.set_aspect('equal')
        ax_AT.scatter(x_boundary, y_boundary, color='black', s=1)
        # aspect ratio
        ax_AT.set_aspect('equal')

        # Plot
        ax_V = axs_grid[1, :].flatten()[i]
        # Plot Vm
        v = vm_all[:, :, expernum]
        v = v[int(t_plot/5), :]

        V_plot = ax_V.tripcolor(
            triang_v, v, shading='gouraud', vmin=-80, vmax=20, cmap=plt.cm.cividis, rasterized = True)

        ax_AT.set_title(
            r'$\bf{g_{{il}}^s}$ = '+f'{gil}' + r' S/m', fontsize=24, fontweight='bold')
        # aspect ratio
        ax_V.set_aspect('equal')
        ax_V.set_ylim(0, 76)

    # Define custom axes for the colorbars
    # [left, bottom, width, height]
    cbar_ax_AT = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    # adjust positions as needed
    cbar_ax_V = fig.add_axes([0.92, 0.1, 0.015, 0.35])

    # Create the colorbars in the specified axes
    Cbar_AT = fig.colorbar(tc_fem0, cax=cbar_ax_AT)
    Cbar_AT.ax.tick_params(labelsize=22)
    Cbar_AT.set_label('Activation time [ms]', fontsize=24)

    Cbar_V = fig.colorbar(V_plot, cax=cbar_ax_V)
    Cbar_V.ax.tick_params(labelsize=22)
    Cbar_V.set_label(r'$V_m$ [mV]', fontsize=24)
    # Add a single colorbar for all subplots
    # cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    # cbar = fig.colorbar(tc_fem0, cax=cbar_ax,
    #                    label='Activation time [ms]')
    # cbar_ax.tick_params(labelsize=22)
    # cbar.set_label('Activation time [ms]', fontsize=30)

    # hide all spines and ticks if not necessary (only keep left y axis (axs_grid[:,0]) and bottom x axis (axs_grid[-1, :]))

    for ax in axs_grid.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, labelleft=False,
                       bottom=False, labelbottom=False)

    # Add shared axis labels for rows and columns
    for ax in axs_grid[:, 0]:

        ax.tick_params(left=True, labelleft=True, labelsize=22)
        ax.spines['left'].set_visible(True)

    axs_grid[0, 0].set_ylabel(r'$\bf{AT}$'+'\n\ny [mm]', fontsize=24)
    axs_grid[1, 0].set_ylabel(
        r'$\bf{V_m}$  $\bf{t=430~ms}$'+'\n\ny [mm]', fontsize=24)
    for ax in axs_grid[-1, :]:
        ax.set_xlabel('x [mm]', fontsize=24)
        ax.tick_params(bottom=True, labelbottom=True, labelsize=22)
        ax.spines['bottom'].set_visible(True)

    # plt.tight_layout()  # Adjust for colorbar

    # add red striped rectangler for zoom
    if zoom is not None:
        x_min, x_max = zoom[2], zoom[3]
        y_min, y_max = zoom[0], np.max(vertices[:, 1]+1)
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=5, edgecolor='black', facecolor='none', linestyle='--')
        axs_grid[1, 0].add_patch(rect)
    # Extend
    # plt.ylim(np.min(vertices[:, 1]), np.max(vertices[:, 1])+1)
    plt.savefig(f"./results/{filename}", bbox_inches='tight')
    plt.close(fig)

def plot_fem_zoom(t_plot, gil_vals, zoom, filename) -> None:
    """
    Plot 4x4 grid of 4 experiments with zoomed in region of FEM data
    """
    set_plot_env()
    
    expernums = expernums_from_gils(gil_vals)

    vertices, triangles, _ = get_data()
    # scar_vertices = np.load("./data/gregion3_coordinates.npy")
    vertices = vertices * 1e-3
    vm_all = np.load("./data/vm_variation_scar_full.npy")
    triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    # plot 4 experiments at t_plot
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)

    positions = range(len(gil_vals))

    for i, expernum, gil_val in zip(positions, expernums, gil_vals):
        v = vm_all[:, :, expernum]
        v = v[int(t_plot/5), :]
        if i == 0:
            ax_index_x = 0
            ax_index_y = 0
        elif i == 1:
            ax_index_x = 1
            ax_index_y = 0
        elif i == 2:
            ax_index_x = 0
            ax_index_y = 1
        else:
            ax_index_x = 1
            ax_index_y = 1

        V_plot = axs[ax_index_y, ax_index_x].tripcolor(triang, v, shading='gouraud', vmin=-80, vmax=20, cmap=plt.cm.cividis, rasterized = True)
        # g_il value as title
        axs[ax_index_y, ax_index_x].set_title(
            r'$\bf{g_{il}^s}$'+f'={gil_val:.3f}', fontsize=24, fontweight='bold', pad=10)
        # axs[ax_index_y, ax_index_x].text(0.5, 0.5, f'g_il={g_il_values[experiment]:.3f}', transform=axs[ax_index_y, ax_index_x].transAxes, fontsize=20,
        #                                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
        # aspect
        axs[ax_index_y, ax_index_x].set_aspect('equal')
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, labelleft=False,
                       bottom=False, labelbottom=False)
    for ax in axs[:, 0]:

        ax.tick_params(left=True, labelleft=True, labelsize=22)
        ax.spines['left'].set_visible(True)
        ax.set_ylabel('y [mm]', fontsize=24)
    for ax in axs[-1, :]:
        ax.set_xlabel('x [mm]', fontsize=24)
        ax.tick_params(bottom=True, labelbottom=True, labelsize=22)
        ax.spines['bottom'].set_visible(True)

    if zoom is not None:
        axs[1, 0].set_xlim(zoom[2], zoom[3])
        axs[1, 0].set_ylim(zoom[0], zoom[1])

        axs[1, 1].set_xlim(zoom[2], zoom[3])
        axs[1, 1].set_ylim(zoom[0], zoom[1])
    # colorbar right of the figure

    cbar_ax = fig.add_axes([0.95, 0.15, 0.04, 0.7])
    fig.colorbar(V_plot, cax=cbar_ax, label=r"$V_{m}$ [mV]")
    
    fig.savefig(f"./results/{filename}",
                bbox_inches='tight')

def plot_combined_errors(act_errors, vm_errors, filename):

    set_plot_env()

    fig, axs = plt.subplots(2, 1, figsize=(15, 6), sharex = True, sharey = True)
    
    # Collect all data for a single bar chart
    bar_colors = list(map(get_dataset_color, EXPERIMENT_GILS))
    bar_labels = list(map(get_dataset_label, EXPERIMENT_GILS))
    
    ticklabels = ["{:.3f}".format(gil) for gil in EXPERIMENT_GILS]

    axs[0].bar(x = range(len(act_errors)),
               height = act_errors,
               color = bar_colors,
               tick_label = ticklabels)
    
    axs[0].set_ylabel(r'$ \mathrm{\Delta}_{AT}$')

    axs[1].bar(x = range(len(vm_errors)),
               height = vm_errors,
               color = bar_colors,
               tick_label = ticklabels)
    
    axs[1].set_ylabel(r'$ \mathrm{\Delta}_{VM}$')

    axs[1].tick_params(axis='x', rotation=90)
    axs[1].set_xlabel("$g_{il}^s$ [S/m]")

    # Add a legend to the right of the plots
    unique_colors_labels = list(set(zip(bar_colors, bar_labels)))  # Get unique color-label pairs
    unique_colors_labels.sort(key=lambda x: bar_colors.index(x[0]))  # Preserve original order

    legend_handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) 
                      for color, label in unique_colors_labels]
    axs[0].legend(handles=legend_handles, 
                  loc='center left',
                  bbox_to_anchor=(1.05, 0.5),
                  title="Dataset")

    plt.savefig(f"./results/{filename}",
                bbox_inches='tight')

def make_fig2(t_plot, gils_plot):
    zoom = [50, 75, 15, 50]
    
    plot_fem_data(t_plot,
                  gils_plot,
                  "Fig2_a.svg",
                  zoom)

    plot_fem_zoom(t_plot,
                   gils_plot,
                   zoom,
                   "Fig2_c.svg")

def make_fig3(vm_errors, act_errors):
    plot_combined_errors(act_errors, 
                         vm_errors,
                         "Fig3.svg")

def make_fig4(vm_errors, act_errors, t_plot, gils_plot):
    plot_meshdata_vertical(ActMeshPlotData(),
                           gils_plot[::-1],
                           "Fig4_a.svg",
                           trace_meshboundary = True)
    
    plot_meshdata_vertical(VMMeshPlotData(t_plot),
                           gils_plot[::-1],
                           "Fig4_b.svg")

def make_fig5(t_plot, gils_plot):
    zoom = [55, 75, 15, 45]
    figsize = (15, 20)

    plot_meshdata_vertical(ActMeshPlotData(),
                           gils_plot[::-1],
                           "Fig5_a.svg",
                           zoom = zoom,
                           figsize = figsize,
                           trace_meshboundary = True)

    plot_meshdata_vertical(VMMeshPlotData(t_plot),
                           gils_plot[::-1],
                           "Fig5_b.svg",
                           zoom = zoom,
                           figsize = figsize)

if __name__ == "__main__":
    t_plot = 430
    gils_plot = [0.01, 0.02, 0.025, 0.1]
    
    vm_errors = get_vm_errordata()
    act_errors, inact_df = get_act_errordata()

    with open("./results/inactivation_data.txt", "w") as f:
        f.write(inact_df.dropna().to_latex())

    make_fig2(t_plot, gils_plot)
    make_fig3(vm_errors, act_errors)
    make_fig4(vm_errors, act_errors, t_plot, gils_plot)
    make_fig5(t_plot, gils_plot)