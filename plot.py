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
from pinn import PINN2D as PINN
from utils import get_mesh_data, get_boundary, load_config
from scar_data_module import ScarPINNDataModule
import numpy as np
import torch


EXPERIMENT_GILS = np.linspace(0.01, 0.1, 19)
DATA_SPLIT = {"train": [0.01, 0.015, 0.02, 0.03, 0.035, 0.04, 0.05, 0.055, 0.06, 0.07, 0.075, 0.08],
              "interpolation": [0.025, 0.045, 0.065],
              "extrapolation": [0.085, 0.09, 0.095, 0.1]}

ACT_LEVELS = 15
EXPERIMENT_DIR="/work/adam2/SCAR-EP-PINN/experiments/run_20250909_193150"
RESULTS_DIR = "./results"
ANIM_OUTDIR = os.path.join(RESULTS_DIR, "animations")
ANIM_CODEC = "libx264"
BITRATE=1000
FILE_extension = ".svg"  # 

COND_PATH = Path("./data/conductivities_variation_scar_full.npy")
VM_FEM_PATH = Path("./data/vm_variation_scar_full.npy")

#Constants
DT_MS = 5.0                    # time step in ms
TIME_NORMALIZER = 12.9         # Time Units (TU) in ms, as in Aliev-Panfilov model
COND_SCALE = 12.9 / 1.4        # to convert conductivities from S/m to mm^2/TU 
XY_MICRO_TO_MILLI = 1e-3

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

class MeshPlotData(object):
    def __init__(self, cmap, vmin, vmax):
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

        self.vertices, self.triangles, _ = get_mesh_data()
        self.vertices *= 1e-3

        self.triang = tri.Triangulation(self.vertices[:,0], 
                                        self.vertices[:,1],
                                        self.triangles)

class VMMeshPlotData(MeshPlotData):
    def __init__(self, t_plot):
        super().__init__('cividis', -80, 20)

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

    def _get_data(self, exp_idx):
        # FEM solution at t_plot
        v_fem = self.vm_FEM[self.frame, :, exp_idx]

        # PINN prediction
        v_pinn = self.vm_PIMM[self.frame, :, exp_idx]
        return v_fem, v_pinn, self.triang

    def comparative_plot(self, ax_FEM, ax_PINN, exp_idx):
        fem_vm, pinn_vm, triang = self._get_data(exp_idx)
        
        pcm = ax_FEM.tripcolor(triang,
                               fem_vm, 
                               shading='gouraud',
                               cmap=self.cmap,
                               vmin=self.vmin,
                               vmax=self.vmax,
                               rasterized = True)

        ax_PINN.tripcolor(triang,
                          pinn_vm,
                          shading='gouraud',
                          cmap=self.cmap,
                          vmin=self.vmin,
                          vmax=self.vmax,
                          rasterized = True)
        return pcm

class ActMeshPlotData(MeshPlotData):
    def __init__(self):
        self.activations_all_FEM = np.load("./data/activations_all_FEM.npy")
        self.activations_all_PINN = np.load("./data/activations_all_PINN.npy")

        max_activation_time = np.nanmax(self.activations_all_FEM)
        
        super().__init__('plasma', 
                          0,
                          max_activation_time)
        
    def _get_data(self, exp_idx):
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

    def comparative_plot(self, ax_PINN, ax_FEM, exp_idx):
        fem_activation_time_masked, triang_fem, pinn_activation_time_masked, triang_pinn = self._get_data(exp_idx)
        
        #Todo: Needs fixing/rasterization due to large vector graphic image sizes

        pcm = ax_FEM.tricontourf(triang_fem,
                                fem_activation_time_masked, 
                                levels=ACT_LEVELS,
                                cmap=self.cmap,
                                vmin=self.vmin,
                                vmax=self.vmax)

        ax_PINN.tricontourf(triang_pinn,
                            pinn_activation_time_masked,
                            levels=ACT_LEVELS,
                            cmap=self.cmap,
                            vmin=self.vmin,
                            vmax=self.vmax)
        return pcm

def plot_mesh_boundary(ax):
    boundary, lines = get_boundary()
    boundary = boundary * 1e-3
    x_boundary = boundary[:, 0]
    y_boundary = boundary[:, 1]

    lines_array = np.array(boundary[lines])
    x_coords = lines_array[:, :, 0] 
    y_coords = lines_array[:, :, 1]  

    ax.plot(x_coords.T, y_coords.T, linewidth=3, color="k")

def plot_meshdata_vertical(actplotdata,
                           vmplotdata, 
                           gils_plot,
                           figsize,
                           gridspec_kw,
                           filename,
                           zoom = None) -> None:
    set_plot_env()

    g_il_values = EXPERIMENT_GILS
    expernums = expernums_from_gils(gils_plot)

    n_exp = len(expernums)

    # create n_exp × 4 grid, share axes
    fig, axs = plt.subplots(nrows=n_exp,
                            ncols=4, 
                            figsize = figsize,
                            sharex=True, 
                            sharey=True,
                            gridspec_kw=gridspec_kw)
    
    # Adjust the bottom margin to make space for colorbars
    fig.subplots_adjust(bottom=0.15)  # Increase bottom margin

    indicies = range(n_exp)
    for i, expernum, gil in zip(indicies, expernums, gils_plot):
        # left: Act
        ax_actPINN = axs[i,0]
        ax_actFEM = axs[i,1]
        
        # right: Vm
        ax_VmPINN = axs[i,2]
        ax_VmFEM = axs[i,3]

        pcm_act = actplotdata.comparative_plot(ax_actPINN,
                                               ax_actFEM,
                                               expernum)

        pcm_vm = vmplotdata.comparative_plot(ax_VmPINN,
                                             ax_VmFEM,
                                             expernum)

        #Add Mesh boundaries to Act data
        for ax in [ax_actPINN, ax_actFEM]:
            plot_mesh_boundary(ax)        
        
        # vertical row-label: bold g_il including units
        # 1) draw the y-axis label in default color
        axs[i,0].set_ylabel("y [mm]",
                           fontsize=20,
                           rotation=90,
                           labelpad=30,
                           color="black",
                           va="center")

        # decide color based on category
        txt_color = get_dataset_color(gil)
        g_str = get_gstr(gil)

        #Add row label with g_il^s value
        axs[i,0].text(x =-0.40,
                      y = 0.5,
                      s = g_str,
                      transform = axs[i,0].transAxes,
                      fontsize = 1.2*plt.rcParams['axes.labelsize'],
                      va="center", 
                      ha = "center",
                      rotation = 90,
                      color=txt_color)

    #Remove unneeded ticks and spines
    for ax in axs[:, 1:].flat:
        ax.tick_params(left=False, labelleft=False)
        ax.spines['left'].set_visible(False)

    for ax in axs[:-1, :].flat:
        ax.tick_params(bottom = False, labelbottom = False)
        ax.spines['bottom'].set_visible(False)
    
    for ax in axs.flat:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if zoom:
        plt.ylim(zoom[0],zoom[1])
        plt.xlim(zoom[2],zoom[3])
    else:
        for ax in axs.flat:
            ax.set_aspect('equal')

        pad = 1
        ymin = vmplotdata.vertices[:,1].min() - pad
        ymax = vmplotdata.vertices[:,1].max() + pad
        plt.ylim(ymin, ymax)

    #Column titles PINN or FEM
    for ax, coltitle in zip(axs[0,:], ["PINN", "FEM"]*2):
        ax.set_title(coltitle,
                    fontsize = 1.5 * plt.rcParams['axes.labelsize'],
                    fontweight ='bold')
    
    # x-axis labels on bottom row
    for ax in axs[-1,:]:
        ax.set_xlabel("x [mm]", fontsize=20)

    # Add colorbar for the first and second columns
    cbar_ax1 = fig.add_axes([0.125, 0.08, 0.35, 0.02])  # [left, bottom, width, height]
    
    cbar1 = fig.colorbar(pcm_act,
                         cax=cbar_ax1,
                         orientation='horizontal')

    cbar1.set_label(r"$AT$ [ms]", fontsize = 1.2*plt.rcParams['axes.labelsize'])

    # Add colorbar for the third and fourth columns
    cbar_ax2 = fig.add_axes([0.55, 0.08, 0.35, 0.02])  # [left, bottom, width, height]
    
    cbar2 = fig.colorbar(pcm_vm, 
                         cax=cbar_ax2,
                         orientation='horizontal')

    cbar2.set_label(r"$V_m$ [mV]", fontsize = 1.2*plt.rcParams['axes.labelsize'])

    #add_cbar_to_plotgrids(fig, axs, pcm, plotdata.cbar_label)

    plt.savefig(f"{RESULTS_DIR}/{filename}{FILE_extension}",
                 bbox_inches='tight') #dpi=300
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

def calculate_inactivation_dice_coefficient(isnanFEM, isnanPINN):
    intersection = np.sum(np.logical_and(isnanFEM, isnanPINN), axis=0)
    sum_isnan = np.sum(isnanFEM, axis=0) + np.sum(isnanPINN, axis=0)
    dice_coefficient = 2 * intersection / sum_isnan
    return dice_coefficient

def plot_fem_data(t_plot, gil_vals, filename, zoom) -> None:
    expernums = expernums_from_gils(gil_vals)
    vertices, triangles, _ = get_mesh_data()
    # scar_vertices = np.load("./data/gregion3_coordinates.npy")
    vertices = vertices * 1e-3
    # scar_vertices = scar_vertices * 1e-3

    V_FEM = np.load("./data/vm_variation_scar_full.npy")
    activation_FEM_all = np.load("./data/activations_all_FEM.npy")
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    n_timepoints = V_FEM.shape[0]
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
        tc_fem = ax_AT.tricontourf(triang, 
                                   fem_activation_time_masked,
                                   levels=ACT_LEVELS,
                                   cmap='plasma',
                                   vmin=0,
                                   vmax=max_activation_time)
        if i == 0:
            tc_fem0 = tc_fem

        ax_AT.set_aspect('equal')
        ax_AT.scatter(x_boundary, y_boundary, color='black', s=1)
        # aspect ratio
        ax_AT.set_aspect('equal')

        # Plot
        ax_V = axs_grid[1, :].flatten()[i]
        # Plot Vm
        v = V_FEM[:, :, expernum]
        v = v[int(t_plot/5), :]

        V_plot = ax_V.tripcolor(triang_v,
                                v,
                                shading='gouraud',
                                vmin=-80,
                                vmax=20,
                                cmap=plt.cm.cividis,
                                rasterized = True)

        ax_AT.set_title(r'$\bf{g_{{il}}^s}$ = '+f'{gil}' + r' S/m', fontsize=24, fontweight='bold')
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
    plt.savefig(f"{RESULTS_DIR}/{filename}{FILE_extension}", bbox_inches='tight')
    plt.close(fig)

def plot_fem_zoom(t_plot, gil_vals, zoom, filename) -> None:
    """
    Plot 4x4 grid of 4 experiments with zoomed in region of FEM data
    """
    set_plot_env()
    
    expernums = expernums_from_gils(gil_vals)

    vertices, triangles, _ = get_mesh_data()
    # scar_vertices = np.load("./data/gregion3_coordinates.npy")
    vertices = vertices * 1e-3
    V_FEM = np.load("./data/vm_variation_scar_full.npy")
    triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    # plot 4 experiments at t_plot
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)

    positions = range(len(gil_vals))

    for i, expernum, gil_val in zip(positions, expernums, gil_vals):
        v = V_FEM[:, :, expernum]
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
    
    fig.savefig(f"{RESULTS_DIR}/{filename}{FILE_extension}",
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

    plt.savefig(f"{RESULTS_DIR}/{filename}{FILE_extension}",
                bbox_inches='tight')

def make_fig2(t_plot, gils_plot):
    zoom = [50, 75, 15, 50]
    
    plot_fem_data(t_plot,
                  gils_plot,
                  "Fig2_a",
                  zoom)

    plot_fem_zoom(t_plot,
                  gils_plot,
                  zoom,
                  "Fig2_c")

def make_fig3(vm_errors, act_errors):
    plot_combined_errors(act_errors, 
                         vm_errors,
                         "Fig3")

def make_fig4(t_plot, gils_plot):
    figsize = (27, 24)
    gridspec_kw = {'hspace': 0.01, 'wspace': 0.01}

    plot_meshdata_vertical(ActMeshPlotData(),
                           VMMeshPlotData(t_plot),
                           gils_plot[::-1],
                           figsize,
                           gridspec_kw,
                           "Fig4")

def make_fig5(t_plot, gils_plot):
    zoom = [55, 75, 15, 45]
    figsize = (35, 27)

    gridspec_kw = {'hspace': 0.1, 'wspace': 0.1}

    plot_meshdata_vertical(ActMeshPlotData(),
                           VMMeshPlotData(t_plot),
                           gils_plot[::-1],
                           figsize,
                           gridspec_kw,
                           "Fig5.svg",
                           zoom = zoom)

def get_error_dfs(vm_errors, act_errors):
    Delta_vm_df = pd.DataFrame(np.array([EXPERIMENT_GILS, vm_errors]).T, columns = ["gils", "Delta_vm"])

    Delta_at_df = pd.DataFrame(np.array([EXPERIMENT_GILS, act_errors]).T, columns = ["gils", "Delta_at"])

    gil_sets = [get_dataset_label(gil) for gil in EXPERIMENT_GILS]
    
    Delta_vm_df["set"] = gil_sets
    Delta_at_df["set"] = gil_sets
    return Delta_vm_df, Delta_at_df

# ---------- Animations  ----------


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _triangulation_from_utils():
    vertices, triangles, _ = get_mesh_data()
    vertices = vertices * 1e-3
    triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    return vertices, triang

def _apply_axes_style(ax, zoom):
    if zoom is not None:
        ax.set_ylim(zoom[0], zoom[1])
        ax.set_xlim(zoom[2], zoom[3])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")

def animation_PINN_FEM(exp_idx,
                       *,
                       zoom=None,
                       fps=20,
                       bitrate=BITRATE,
                       filename=None):
    """
    Side-by-side animation of PINN vs FEM for experiment index exp_idx.
    Uses saved arrays: ./data/vm_PINN.npy and ./data/vm_variation_scar_full.npy
    """
    _ensure_dir(ANIM_OUTDIR)
    vertices, triang = _triangulation_from_utils()

    v_FEM_all  = np.load("./data/vm_variation_scar_full.npy")  # (T, V, E)
    v_PINN_all = np.load("./data/vm_PINN.npy")                  # (T, V, E)

    v_FEM  = v_FEM_all[:, :, exp_idx]
    v_PINN = v_PINN_all[:, :, exp_idx]
    g_il = EXPERIMENT_GILS[exp_idx]

    T_len, V_count = v_FEM.shape
    assert v_PINN.shape == (T_len, V_count)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    im1 = ax1.tripcolor(triang, v_PINN[0, :], shading="gouraud", cmap=plt.cm.cividis, vmin=-80, vmax=20)
    im2 = ax2.tripcolor(triang, v_FEM[0,  :], shading="gouraud", cmap=plt.cm.cividis, vmin=-80, vmax=20)


    cbar = fig.colorbar(im2, ax=[ax1, ax2],  # span both axes
                    orientation="vertical",
                    fraction=0.046, pad=0.04)
    cbar.set_label(r"$V_m$ [mV]")
    _apply_axes_style(ax1, zoom)
    _apply_axes_style(ax2, zoom)
    ax1.set_title(r"$\bf{PINN}$: $g_{il}$="+f"{g_il:.3f}S/m", fontsize=20)
    ax2.set_title(r"$\bf{FEM}$: $g_{il}$="+f"{g_il:.3f}S/m",  fontsize=20)

    def update(frame):
        v_pred = v_PINN[frame, :]
        v_true = v_FEM[frame,  :]

        im1.set_array(v_pred)
        im2.set_array(v_true)

        # Per-frame RL1 (consistent with your older title behavior)
        denom = np.sum(np.abs(v_true))
        RL1 = (np.sum(np.abs(v_true - v_pred)) / denom) if denom > 0 else np.nan

        ax1.set_title(r"$\bf{PINN}$: $g_{il}$="+f"{g_il:.3f}S/m" + f"\n t={frame*5:.0f} ms, RL1={RL1:.3f}", fontsize=20)
        ax2.set_title(r"$\bf{FEM}$: $g_{il}$="+f"{g_il:.3f}S/m"  + f"\n t={frame*5:.0f} ms",               fontsize=20)
        return im1, im2

    ani = animation.FuncAnimation(fig, update, frames=T_len, interval=1, blit=False)

    if filename is None:
        filename = f"S{exp_idx+1}_vm_pinn_fem_gil_{g_il:.3f}.mp4"
    outpath = os.path.join(ANIM_OUTDIR, filename)
    ani.save(outpath, writer=animation.FFMpegWriter(fps=fps, codec=ANIM_CODEC, bitrate=bitrate))
    plt.close(fig)


def animation_PINN(exp_idx,
                   *,
                   zoom=None,
                   fps=20,
                   bitrate=BITRATE,
                   filename=None):
    """
    PINN-only animation for experiment index exp_idx.
    """
    _ensure_dir(ANIM_OUTDIR)
    vertices, triang = _triangulation_from_utils()

    v_PINN_all = np.load("./data/vm_PINN.npy")  # (T, V, E)
    v_PINN = v_PINN_all[:, :, exp_idx]
    T_len, _ = v_PINN.shape
    g_il = EXPERIMENT_GILS[exp_idx]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.tripcolor(triang, v_PINN[0, :], shading="gouraud", cmap=plt.cm.cividis, vmin=-80, vmax=20)
    plt.colorbar(im, ax=ax, label=r"$V_m$ [mV]")
    _apply_axes_style(ax, zoom)
    ax.set_title(r"$\bf{PINN}$: $g_{il}$="+f"{g_il:.3f}S/m", fontsize=20)

    def update(frame):
        im.set_array(v_PINN[frame, :])
        ax.set_title(r"$\bf{PINN}$: $g_{il}$="+f"{g_il:.3f}S/m" + f"\n t={frame*5:.0f} ms", fontsize=20)
        return im,

    ani = animation.FuncAnimation(fig, update, frames=T_len, interval=1, blit=False)

    if filename is None:
        filename = f"S{exp_idx+1}_vm_pinn_gil_{g_il:.3f}.mp4"
    outpath = os.path.join(ANIM_OUTDIR, filename)
    ani.save(outpath, writer=animation.FFMpegWriter(fps=fps, codec=ANIM_CODEC, bitrate=bitrate))
    plt.close(fig)


def animation_FEM(exp_idx,
                  *,
                  zoom=None,
                  fps=20,
                  bitrate=BITRATE,
                  filename=None):
    """
    FEM-only animation for experiment index exp_idx.
    """
    _ensure_dir(ANIM_OUTDIR)
    vertices, triang = _triangulation_from_utils()

    v_FEM_all = np.load("./data/vm_variation_scar_full.npy")  # (T, V, E)
    v_FEM = v_FEM_all[:, :, exp_idx]
    T_len, _ = v_FEM.shape
    g_il = EXPERIMENT_GILS[exp_idx]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.tripcolor(triang, v_FEM[0, :], shading="gouraud", cmap=plt.cm.cividis, vmin=-80, vmax=20)
    plt.colorbar(im, ax=ax, label=r"$V_m$ [mV]")
    _apply_axes_style(ax, zoom)
    ax.set_title(r"$\bf{FEM}$: $g_{il}$="+f"{g_il:.3f}S/m", fontsize=20)

    def update(frame):
        im.set_array(v_FEM[frame, :])
        ax.set_title(r"$\bf{FEM}$: $g_{il}$="+f"{g_il:.3f}S/m" + f"\n t={frame*5:.0f} ms", fontsize=20)
        return im,

    ani = animation.FuncAnimation(fig, update, frames=T_len, interval=1, blit=False)

    if filename is None:
        filename = f"S{exp_idx+1}_vm_fem_gil_{g_il:.3f}.mp4"
    outpath = os.path.join(ANIM_OUTDIR, filename)
    ani.save(outpath, writer=animation.FFMpegWriter(fps=fps, codec=ANIM_CODEC, bitrate=bitrate))
    plt.close(fig)



def make_animations(gil_vals=None, zoom=None):
    """
    If gil_vals is None, animates all experiments. Otherwise accepts a list of g_il values.
    """
    if gil_vals is None:
        exp_indices = range(len(EXPERIMENT_GILS))
    else:
        exp_indices = expernums_from_gils(gil_vals)

    for exp_idx in exp_indices:
        print(f"Animating experiment index {exp_idx}")
        animation_PINN_FEM(exp_idx, zoom=zoom)
        animation_PINN(exp_idx, zoom=zoom)
        animation_FEM(exp_idx,  zoom=zoom)

def get_v_data(pinn: PINN,epoch) -> np.ndarray:
    """Compute PINN voltage predictions V_PINN with shape (T, V, E)."""
    vertices, triangles, _ = get_mesh_data()
    vertices = vertices * XY_MICRO_TO_MILLI
    V_FEM = np.load(VM_FEM_PATH)  
    cond_all = np.load(COND_PATH)* COND_SCALE

    device = pinn.device
    x = vertices[:, 0]
    y = vertices[:, 1]

    T_len, V_count, E_count = V_FEM.shape
    n_timepoints = V_FEM.shape[0]
    t_start = 0.0
    t_end = (n_timepoints - 1) * DT_MS
    t=np.linspace(t_start, t_end, n_timepoints) / TIME_NORMALIZER
    V_PINN = np.zeros_like(V_FEM)
    exp_indexes=[3,18,17,16,15]
    g_il_vals = EXPERIMENT_GILS[exp_indexes]
    errors = np.zeros((5, ))  # (E, T, V)
    i=0
    with torch.no_grad():
        for exp_idx, g_il in zip(exp_indexes, g_il_vals):
            print(f"Experiment {exp_idx}")
            conductivities = cond_all[:, :, :, exp_idx]
            sigma_a = conductivities[:, 0, 0]
            sigma_b = conductivities[:, 0, 1]
            sigma_c = conductivities[:, 1, 0]
            sigma_d = conductivities[:, 1, 1]

            # Base per-vertex features
            base = np.column_stack((x, y, sigma_a, sigma_b, sigma_c, sigma_d)).astype(np.float32)
            # Compute predictions for one timepoint at a time for memory efficiency
            for ti, t_curr in enumerate(t):
                t_col = np.full((V_count, 1), t_curr, dtype=np.float32)
                exp_col = np.full((V_count, 1), g_il, dtype=np.float32)
                data = np.hstack((base[:, :2], t_col, base[:, 2:], exp_col))  # (V, 8)
                data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
                v_pred = pinn.predict(data_tensor)[:, 0].detach().cpu().numpy()
                V_PINN[ti, :, exp_idx] = v_pred

            # Optional diagnostics
            v_true = V_FEM[:, :, exp_idx]
            abs_sum = np.sum(np.abs(v_true - V_PINN[:, :, exp_idx]))
            v_sum = np.sum(np.abs(v_true))
            errors[i] = abs_sum / v_sum
            i+=1
            
            print("L1 relative error:", abs_sum / v_sum)
            
    #Save errors


    return errors

def plot_error_epochs():

    
    experiment_path =EXPERIMENT_DIR
    config_path = experiment_path + "/config.yaml"
    config = load_config(str(config_path))

    n_epochs_data = config["OPTIM"]["phase1"]["epochs"]
    n_epochs_physics = config["OPTIM"]["phase2"]["epochs"]
    n_epochs = n_epochs_data + n_epochs_physics
    #if error files exist, load them
    if os.path.exists(f"{experiment_path}/l1_errors_epochs.npy") and os.path.exists(f"{experiment_path}/l1_errors_avg_epochs.npy"):
        error = np.load(f"{experiment_path}/l1_errors_epochs.npy")
        avg_error = np.load(f"{experiment_path}/l1_errors_avg_epochs.npy")
        exp_indexes=[3,18,17,16,15]
        for i,g_il in enumerate(EXPERIMENT_GILS[exp_indexes]):
            print(f"Experiment {i} G_il={g_il}: Relative L1 Error: {error[i,:]}")
            plt.plot(np.arange(1,n_epochs+1), error[i,:], label=f"G_il={g_il}")

        plt.plot(np.arange(1,n_epochs+1), avg_error, label=f"Average", color='black', linewidth=2, linestyle='--')

        plt.axvline(x=n_epochs_data, color='red', linestyle='--', label='Physics start')
        
        plt.xlabel("Epoch")
        plt.ylabel("Relative L1 Error")
        plt.yscale('log')
        plt.title("Relative L1 Error vs Epoch for Different G_il Values")
        plt.legend()
        figname=f"/l1_errors_epoch{FILE_extension}"
        plt.savefig(RESULTS_DIR + figname)
        return
    #Else compute them
    
    else:
        dm = ScarPINNDataModule(config)
        dm.setup_data()
        min, max = dm.get_minmax_tensors()
        
        error = np.zeros((5,n_epochs))
        avg_error = np.zeros(n_epochs)  
        pinn = PINN(config, min, max)

        for i in range(1,n_epochs+1):
            print(f"##########################Epoch {i}################################")
            
            if i <= n_epochs_data:
                model_name = f"/model_data_epoch_{i}.pth"
                print(f"Loading model from data phase: {model_name}")
            else:
                model_name = f"/model_Adam_epoch_{i-n_epochs_data}.pth"
                print(f"Loading model from physics phase: {model_name}")
            model_path = experiment_path + model_name

            # Detect device: prefer CUDA, then MPS, then CPU
            if torch.cuda.is_available():
                map_location = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                map_location = torch.device("mps")
            else:
                map_location = torch.device("cpu")


            state_dict = torch.load(str(model_path), map_location=map_location)
            pinn.load_state_dict(state_dict)
            pinn.eval()


            



            
            error[:,i-1] = get_v_data(pinn,i)
            avg_error[i-1] = np.mean(error[:,i-1])
    #Save errors if files do not exist
    if not os.path.exists(f"{experiment_path}/l1_errors_epochs.npy"):
        np.save(f"{experiment_path}/l1_errors_epochs.npy", error)
    if not os.path.exists(f"{experiment_path}/l1_errors_avg_epochs.npy"):
        np.save(f"{experiment_path}/l1_errors_avg_epochs.npy", avg_error)
    
    exp_indexes=[3,18,17,16,15]
    for i,g_il in enumerate(EXPERIMENT_GILS[exp_indexes]):
        print(f"Experiment {i} G_il={g_il}: Relative L1 Error: {error[i,:]}")
        plt.plot(np.arange(1,n_epochs+1), error[i,:], label=f"G_il={g_il}")

    plt.plot(np.arange(1,n_epochs+1), avg_error, label=f"Average", color='black', linewidth=2, linestyle='--')

    plt.axvline(x=n_epochs_data, color='red', linestyle='--', label='Physics start')
    
    plt.xlabel("Epoch")
    plt.ylabel("Relative L1 Error")
    plt.yscale('log')
    plt.title("Relative L1 Error vs Epoch for Different G_il Values")
    plt.legend()
    figname=f"/l1_errors_epoch{FILE_extension}"
    plt.savefig(RESULTS_DIR + figname)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
def plot_loss():
    experiment_path=EXPERIMENT_DIR
    losses = np.load(f"{experiment_path}/losses.npy")
    losses_test=np.load(f"{experiment_path}/losses_test.npy")
    iteration_list = losses[:,0]
    window_size = 2000
    loss_mse_smooth = moving_average(losses[:,1], window_size)
    loss_mse_test_smooth = moving_average(losses_test[:,1], window_size)
    MSE_test_min_iteration = iteration_list[np.argmin(losses_test[:,1])]
    plt.plot(iteration_list, losses[:,1],color='gray', alpha=0.7, label=f"MSE")

    plt.plot(iteration_list, losses_test[:,1],color='lightgreen', alpha=0.7, label=f"MSE test")
    plt.plot(iteration_list[:-window_size], loss_mse_test_smooth[:-1],color='green', label=f"MSE test smooth")
    plt.plot(iteration_list[:-window_size], loss_mse_smooth[:-1],color='black', label=f"MSE smooth")
    
    plt.scatter(MSE_test_min_iteration, np.min(losses_test[:,1]), color='red', label="Min MSE test")
    plt.yscale('log')
    #x axis label
    plt.xlabel("Iteration")
    #y axis label
    plt.ylabel("MSE")
    #title
    plt.title("Training and Validation Loss")
    #save figure
    plt.savefig(f"{RESULTS_DIR}/losses{FILE_extension}")


if __name__ == "__main__":
    t_plot = 430
    gils_plot = [0.01, 0.02, 0.025, 0.1]
    
    vm_errors = get_vm_errordata()
    act_errors, inact_df = get_act_errordata()

    with open(f"{RESULTS_DIR}/inactivation_data.txt", "w") as f:
        f.write(inact_df.dropna().to_latex())
    #Figures
    #make_fig2(t_plot, gils_plot)
    #make_fig3(vm_errors, act_errors)
    #make_fig4(t_plot, gils_plot)
    #make_fig5(t_plot, gils_plot)
    
    #Animations
    #make_animations()

    #Loss 
    plot_loss()
    plot_error_epochs()

    #For interactive analysis
    Delta_vm_df, Delta_at_df = get_error_dfs(vm_errors, act_errors)

    #from IPython import embed; embed()
