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

EXPERIMENT_GILS = np.linspace(0.01, 0.1, 19)
DATA_SPLIT = {"train": [0.01, 0.015, 0.02, 0.03, 0.035, 0.04, 0.05, 0.055, 0.06, 0.07, 0.075, 0.08],
              "interpolation": [0.025, 0.045, 0.065],
              "extrapolation": [0.085, 0.09, 0.095, 0.1]}

def expernums_from_gils(gil_vals):
    return [np.where(np.isclose(EXPERIMENT_GILS, gil))[0][0] for gil in gil_vals]

def get_dataset_color(gil):
    if np.isclose(gil, DATA_SPLIT["train"]).any():
        return 'g', "Training"
    elif np.isclose(gil, DATA_SPLIT["interpolation"]).any():
        return 'b', "Interpolation"
    elif np.isclose(gil, DATA_SPLIT["extrapolation"]).any():
        return 'r', "Extrapolation"
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

def activation_map_GT(pinn, model, experiments, frame_4, experiment_path, filename) -> None:

    # g_il_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # g_il_values = np.linspace(0, 9, 10)
    g_il_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045,
                   0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
    g_il_vals = g_il_values
    extra = False
    long = False
    isotropic = True
    vertices, triangles, _ = get_data(
        extra=extra, long=long, isotropic=isotropic)
    scar_vertices = np.load("./data/gregion3_coordinates.npy")
    vertices = vertices * 1e-3
    scar_vertices = scar_vertices * 1e-3
    vm_all = np.load("./data/vm_variation_scar_full.npy")
    activations_all_FEM = np.load("./data/activations_all_FEM.npy")
    print("VM shape", vm_all.shape)
    conductivities_all = np.load(
        "./data/conductivities_variation_scar_full.npy")

    device = pinn.device
    x = vertices[:, 0]
    y = vertices[:, 1]
    n_timepoints = vm_all.shape[0]
    dt = 5
    t_start = 0
    t_end = (n_timepoints - 1) * dt
    t = np.linspace(t_start, t_end, n_timepoints) / \
        12.9

    activation_threshold = -40  # Define your activation threshold

    # Prepare boundary for plotting
    boundary, _ = get_boundary()
    boundary = boundary * 1e-3
    x_boundary = boundary[:, 0]
    y_boundary = boundary[:, 1]

    # Create the figure and subplots
    fig, axs_grid = plt.subplots(
        2, 4, figsize=(24, 12), sharex=True, sharey=True)
    axs = axs_grid.flatten()

    for i, experiment in enumerate(experiments):
        conductivities = conductivities_all[:, :, :, experiment] * 12.9 / 1.4
        sigma_a = conductivities[:, 0, 0]
        sigma_b = conductivities[:, 0, 1]
        sigma_c = conductivities[:, 1, 0]
        sigma_d = conductivities[:, 1, 1]
        exp_ind = np.ones(len(x)) * g_il_values[experiment]

       
        fem_activation_time = activations_all_FEM[:, experiment]
        # Triangulate and mask invalid data for FEM and PINN
        triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

        

        fem_activation_time_masked = ma.masked_invalid(fem_activation_time)
        pinn_activation_time_masked = ma.masked_invalid(pinn_activation_time)

        tri_mask_fem = np.any(
            np.isnan(fem_activation_time[triang.triangles]), axis=1)
        tri_mask_pinn = np.any(
            np.isnan(pinn_activation_time[triang.triangles]), axis=1)

        triang.set_mask(tri_mask_fem)

        # Plot FEM activation map
        ax_fem = axs_grid[1, :].flatten()[i]
        tc_fem = ax_fem.tricontourf(triang, fem_activation_time_masked, levels=15,
                                    cmap='plasma', vmin=0, vmax=max_activation_time)
        if i == 0:
            tc_fem0 = tc_fem

        ax_fem.set_aspect('equal')
        ax_fem.scatter(x_boundary, y_boundary, color='black', s=1)

        triang.set_mask(tri_mask_pinn)

        # Plot PINN activation map
        ax_pinn = axs_grid[0, :].flatten()[i]
        tc_pinn = ax_pinn.tricontourf(triang, pinn_activation_time_masked, levels=15,
                                      cmap='plasma', vmin=0, vmax=max_activation_time)
        ax_pinn.set_title(
            r'$\bf{g_{{il}}}$ = '+f'{g_il_vals[experiment]}' + r' S/m', fontsize=24, fontweight='bold')
        # add L1 error to middle of all pinn plots
        ax_pinn.text(0.5, 0.5, f'$RL^1$ = {l1_error:.3f}', fontsize=20,
                     ha='center', va='center', transform=ax_pinn.transAxes)
        ax_pinn.set_aspect('equal')
        ax_pinn.scatter(x_boundary, y_boundary, color='black', s=1)

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    cbar = fig.colorbar(tc_fem0, cax=cbar_ax,
                        label='Activation time [ms]')
    cbar_ax.tick_params(labelsize=22)
    cbar.set_label('Activation time [ms]', fontsize=30)

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

    axs_grid[0, 0].set_ylabel(r'$\bf{PINN}$'+'\n\ny [mm]', fontsize=24)
    axs_grid[1, 0].set_ylabel(r'$\bf{FEM}$'+'\n\ny [mm]', fontsize=24)
    for ax in axs_grid[-1, :]:
        ax.set_xlabel('x [mm]', fontsize=24)
        ax.tick_params(bottom=True, labelbottom=True, labelsize=22)
        ax.spines['bottom'].set_visible(True)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust for colorbar
    plt.savefig(f"./results/{filename}", bbox_inches='tight')
    plt.close(fig)

def activation_vertical_zoom(experiments, filename, zoom) -> None:
    set_plot_env()
    g_il_vals = EXPERIMENT_GILS
    indices_experiments = np.arange(len(g_il_values)-6, dtype=int)
    sampled_indices = indices_experiments[(
        indices_experiments + 1) % 4 != 0]
    sampled_indices = sampled_indices[:12]
    extrapolation_gil = g_il_values[-5:]
    train_gil       = g_il_values[sampled_indices]
    interpolation_idx=np.array([3,7,11])
    interpolation_gil = g_il_values[interpolation_idx]
    #extra = False
    #long = False
    #isotropic = True
    vertices, triangles, _ = get_data()
    #scar_vertices = np.load("./data/gregion3_coordinates.npy")
    vertices = vertices * 1e-3
    #scar_vertices = scar_vertices * 1e-3
    vm_all = np.load("./data/vm_variation_scar_full.npy")
    activations_all_FEM = np.load("./data/activations_all_FEM.npy")
    activations_all_PINN = np.load("./data/activations_all_PINN.npy")
    print("VM shape", vm_all.shape)
    conductivities_all = np.load(
        "./data/conductivities_variation_scar_full.npy")
    
    activation_threshold = -40

    x = vertices[:, 0]
    y = vertices[:, 1]
    boundary, _ = get_boundary()
    boundary = boundary * 1e-3
    x_boundary = boundary[:, 0]
    y_boundary = boundary[:, 1]

    n_timepoints = vm_all.shape[0]
    dt = 5
    t_start = 0
    t_end = (n_timepoints - 1) * dt
    t = np.linspace(t_start, t_end, n_timepoints) / 12.9

    n_exp = len(experiments)
    fig, axs = plt.subplots(
    nrows=n_exp, ncols=2,
    figsize=(12, 3.7*n_exp),
    sharex=True, sharey=True,
    gridspec_kw={'hspace': 0.001})
    triang = tri.Triangulation(vertices[:,0], vertices[:,1], triangles)
    set_plot_env()
    i=0
    max_activation_time = np.nanmax(activations_all_FEM)
    for i,exp_idx in enumerate(experiments):
        fem_activation_time = activations_all_FEM[:, exp_idx]
        pinn_activation_time = activations_all_PINN[:, exp_idx]
        fem_activation_time_masked = ma.masked_invalid(fem_activation_time)
        pinn_activation_time_masked = ma.masked_invalid(pinn_activation_time)
        triang_pinn = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
        triang_fem = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
        tri_mask_fem = np.any(
            np.isnan(fem_activation_time[triang.triangles]), axis=1)
        tri_mask_pinn = np.any(
            np.isnan(pinn_activation_time[triang.triangles]), axis=1)
        #triang_pinn.set_mask(tri_mask_pinn)
        #triang_fem.set_mask(tri_mask_fem)
        # left: PINN
        ax = axs[i,0]
        pcm = ax.tripcolor(triang, pinn_activation_time_masked,
                                      cmap='plasma', vmin=0, vmax=max_activation_time)
        ax.scatter(x_boundary, y_boundary, color='black', s=1)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_title("PINN", fontsize=25, fontweight='bold')

        # right: FEM
        ax2 = axs[i,1]
        ax2.tripcolor(triang, fem_activation_time_masked,
                                    cmap='plasma', vmin=0, vmax=max_activation_time)
        ax2.scatter(x_boundary, y_boundary, color='black', s=1)
        ax2.set_aspect('equal')
        if i == 0:
            ax2.set_title("FEM", fontsize=25, fontweight='bold')

        # hide spines
        for aa in (ax, ax2):
            for sp in aa.spines.values():
                sp.set_visible(False)
                
                if i == n_exp-1:
                    axs[i,0].spines['bottom'].set_visible(True)
                    axs[i,1].spines['bottom'].set_visible(True)
        #axs[i,0].spines['left'].set_visible(True)
        # vertical row-label: bold g_il including units
        # 1) draw the y-axis label in default color
        #ax.set_ylabel(
        #    "y [mm]",
        #    fontsize=20,
        #    rotation=90,
        #    labelpad=30,
        #    color="black",
        #    va="center"
        #)
        # decide color based on category
        gil = g_il_values[exp_idx]
        if exp_idx in sampled_indices:
            txt_color = 'g'
        elif exp_idx in interpolation_idx:
            txt_color = 'b'
        else:
            # everything else is extrapolation
            txt_color = 'r'

        # bold, colored g_il text
        g_str = get_gstr(gil) # rf"$\mathbf{{g_{{il}}={gil:.3f}\,\mathbf{{S/m}}}}$"
        """
        ax.text(
            x=-0.40, y=0.5, s=g_str,
            transform=ax.transAxes,
            fontsize=22,
            va="center", ha="center",
            rotation=90,
            color=txt_color
        )
        """
        # ticks: y only on left column; x only on bottom row
        ax.tick_params(
            left=False, labelleft=False,
            bottom=(i==n_exp-1), labelbottom=(i==n_exp-1)
        )
        ax2.tick_params(
            left=False, labelleft=False,
            bottom=(i==n_exp-1), labelbottom=(i==n_exp-1)
        )
        #aspect ratio
        ax.set_aspect('equal')
        ax2.set_aspect('equal')

        # x-axis labels on bottom row
        if i == n_exp-1:
            ax.set_xlabel("x [mm]", fontsize=20)
            ax2.set_xlabel("x [mm]", fontsize=20)
        plt.ylim(zoom[0],zoom[1])
        plt.xlim(zoom[2],zoom[3])
        
        
    fig.subplots_adjust(
        left=0.20,
        right=0.88,
        top=0.9,    # ⬅️ push the subplots down to 75% of the figure
        bottom=0.1,
        hspace=0.001,
        wspace=0.15
    )

    # … draw your tripcolor plots …

    # now add the horizontal colorbar *over* the space you freed up:
    cb = fig.colorbar(
        pcm,
        ax=axs,
        orientation='horizontal',
        location='bottom',    # requires mpl ≥3.2
        fraction=0.05,     # bar thickness
        pad=0.07,          # small pad from the top of the subplot area
        label=r"$AT$ [ms]",
    )
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.xaxis.set_label_position('bottom')
    #plt.tight_layout()
    #plt.tight_layout(pad=0.5, h_pad=0.02)

    #Figure title
    #fig.suptitle(
    #    "Activation time ", fontsize=30, fontweight='bold')
    plt.savefig(f"./results/{filename}",
                dpi=300)
    plt.close(fig)

def plot_vm_vertical_zoom(
        t_plot,        # time in same units as dt=5
        gils_plot,   # list of experiment conductivities
        filename,zoom
    ) -> None:
    set_plot_env()

    g_il_values = np.array([
        0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
        0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
        0.075, 0.08, 0.085, 0.09, 0.095, 0.1
    ])
    indices_experiments = np.arange(len(g_il_values)-6, dtype=int)
    sampled_indices = indices_experiments[(
        indices_experiments + 1) % 4 != 0]
    sampled_indices = sampled_indices[:12]
    extrapolation_gil = g_il_values[-5:]
    train_gil       = g_il_values[sampled_indices]
    interpolation_idx=np.array([3,7,11])
    interpolation_gil = g_il_values[interpolation_idx]

    vertices, triangles, _ = get_data()
    vertices *= 1e-3

    vm_FEM   = np.load("./data/vm_variation_scar_full.npy")            # (T, V, E)
    vm_PINN  = np.load("./data/vm_PINN.npy")            # (T, V, E)
    cond_all = np.load("./data/conductivities_variation_scar_full.npy")
    cond_all *= 12.9/1.4

    dt = 5
    T  = vm_FEM.shape[0]
    t  = np.linspace(0, (T-1)*dt, T)/12.9
    frame = int(t_plot/dt)
    if frame < 0 or frame >= T:
        raise ValueError("t_plot out of range")

    n_exp = len(experiments)


    # create n_exp × 2 grid, share axes
    fig, axs = plt.subplots(
        nrows=n_exp, ncols=2,
        figsize=(12, 3.7*n_exp),
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0.001}
    )
    # reduce vertical whitespace between rows
    

    triang = tri.Triangulation(vertices[:,0], vertices[:,1], triangles)
    set_plot_env()
    for i, exp_idx in enumerate(experiments):
        # FEM solution at t_plot
        v_fem = vm_FEM[frame, :, exp_idx]

        # PINN prediction
        v_pinn = vm_PINN[frame, :, exp_idx]

        # left: PINN
        ax = axs[i,0]
        pcm = ax.tripcolor(triang, v_pinn,
                           shading='gouraud', cmap='cividis',
                           vmin=-80, vmax=20)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_title("PINN", fontsize=25, fontweight='bold')

        # right: FEM
        ax2 = axs[i,1]
        ax2.tripcolor(triang, v_fem,
                      shading='gouraud', cmap='cividis',
                      vmin=-80, vmax=20)
        ax2.set_aspect('equal')
        if i == 0:
            ax2.set_title("FEM", fontsize=25, fontweight='bold')

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
        ax.set_ylabel(
            "y [mm]",
            fontsize=20,
            rotation=90,
            labelpad=30,
            color="black",
            va="center"
        )
        # decide color based on category
        gil = g_il_values[exp_idx]
        if exp_idx in sampled_indices:
            txt_color = 'g'
        elif exp_idx in interpolation_idx:
            txt_color = 'b'
        else:
            # everything else is extrapolation
            txt_color = 'r'

        # bold, colored g_il text
        g_str = get_gstr(gil) #rf"$\mathbf{{g_{{il}}={gil:.3f}\,\mathbf{{S/m}}}}$"
        ax.text(
            x=-0.40, y=0.5, s=g_str,
            transform=ax.transAxes,
            fontsize=22,
            va="center", ha="center",
            rotation=90,
            color=txt_color
        )

        # ticks: y only on left column; x only on bottom row
        ax.tick_params(
            left=True, labelleft=True,
            bottom=(i==n_exp-1), labelbottom=(i==n_exp-1)
        )
        #aspect ratio
        ax.set_aspect('equal')
        ax2.set_aspect('equal')
        ax2.tick_params(
            left=False, labelleft=False,
            bottom=(i==n_exp-1), labelbottom=(i==n_exp-1)
        )

        # x-axis labels on bottom row
        if i == n_exp-1:
            ax.set_xlabel("x [mm]", fontsize=20)
            ax2.set_xlabel("x [mm]", fontsize=20)
    plt.ylim(zoom[0],zoom[1])
    plt.xlim(zoom[2],zoom[3])
    """
    # single shared colorbar
    h = 0.70/2
    w = 0.02*2
    # center it vertically in the original colorbar region:
    y0 = 0.15 + (0.70 - h)/2
    cax = fig.add_axes([1, y0, w, h])
    fig.colorbar(pcm, cax=cax, label=r"$V_m$ [mV]")
    """
    # pull the titles (and grid) down a bit so there's room for the bar above them
    # right after you create fig, axs:
    fig.subplots_adjust(
        left=0.20,
        right=0.88,
        top=0.9,    # ⬅️ push the subplots down to 75% of the figure
        bottom=0.1,
        hspace=0.001,
        wspace=0.15
    )

    # … draw your tripcolor plots …

    # now add the horizontal colorbar *over* the space you freed up:
    cb = fig.colorbar(
        pcm,
        ax=axs,
        orientation='horizontal',
        location='bottom',    # requires mpl ≥3.2
        fraction=0.05,     # bar thickness
        pad=0.07,          # small pad from the top of the subplot area
        label=r"$V_m$ [mV]"
    )
    #set colorbar ticks
    #cb.set_ticks([-80, -60, -40, -20, 0, 20])
    
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.xaxis.set_label_position('bottom')
    
    #plt.tight_layout()
    #plt.tight_layout(pad=0.5, h_pad=0.02)
    
    #fig.suptitle(
    #    "Voltage ", fontsize=30, fontweight='bold', y=0.95)

    plt.savefig(f"./results/{filename}",
                dpi=300)
    plt.close(fig)

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
        cond_all = np.load("./data/conductivities_variation_scar_full.npy")
        cond_all *= 12.9/1.4

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

def plot_meshdata_vertical(plotdata, gils_plot, filename) -> None:
    set_plot_env()

    g_il_values = EXPERIMENT_GILS
    expernums = expernums_from_gils(gils_plot)

    n_exp = len(expernums)

    # create n_exp × 2 grid, share axes
    fig, axs = plt.subplots(nrows=n_exp,
                            ncols=2, 
                            figsize=(15, 24), #5*n_exp
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
        txt_color, txt_label = get_dataset_color(gil)

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
        plt.ylim(plotdata.vertices[:,1].min(), plotdata.vertices[:,1].max()-1)

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

def get_act_errordata():
    vertices, triangles, _ = get_data()
    #scar_vertices = np.load("./data/gregion3_coordinates.npy")
    vertices = vertices * 1e-3
    #scar_vertices = scar_vertices * 1e-3
    vm_all = np.load("./data/vm_variation_scar_full.npy")
    activations_all_FEM = np.load("./data/activations_all_FEM.npy")
    activations_all_PINN = np.load("./data/activations_all_PINN.npy")
    print("VM shape", vm_all.shape)
    
    x = vertices[:, 0]
    y = vertices[:, 1]

    n_timepoints = vm_all.shape[0]
    dt = 5
    t_start = 0
    t_end = (n_timepoints - 1) * dt
    t = np.linspace(t_start, t_end, n_timepoints) / 12.9

    relative_L1_errors = np.nansum(
        np.abs(activations_all_PINN - activations_all_FEM), axis=0) / \
        np.nansum(np.abs(activations_all_FEM), axis=0)
    return relative_L1_errors

def plot_errorbars_horizontal(errors, xlabel, filename):
    set_plot_env()
    g_il_values = EXPERIMENT_GILS
    
    bar_height = 0.005

    fig, ax = plt.subplots(figsize=(12, 24))

    count_train = count_interp = count_extra = 0
    
    # Collect all data for a single bar chart
    bar_colors = []
    bar_labels = []

    for g_il in g_il_values:
        barcolor, label = get_dataset_color(g_il)
        bar_colors.append(barcolor)
        bar_labels.append(label)

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
    
    fig.savefig(f"./results/{filename}", bbox_inches='tight')

def data_split():
    gil_all = EXPERIMENT_GILS

    # Evaluation data
    gil_evaluation_up_to_0_07 = [value for idx, value in enumerate(
        gil_all) if value <= 0.07 and idx % 3 == 2]
    gil_evaluation_beyond_0_075 = [value for value in gil_all if value > 0.075]
    gil_evaluation = gil_evaluation_up_to_0_07 + gil_evaluation_beyond_0_075

    # Training data
    gil_training = [value for value in gil_all if value not in gil_evaluation]

    # Create the plot
    plt.figure(figsize=(12, 4))  # Adjusted height for clarity

    # Plot training data
    plt.bar(gil_training, [0.6] * len(gil_training),
            color='blue', width=0.003, label='Training')

    # Plot evaluation data
    plt.bar(gil_evaluation, [0.6] * len(gil_evaluation),
            color='orange', width=0.003, label='Evaluation')

    # Set labels and title
    plt.xlabel('$g_{il}$ [S/m]', fontsize=25)
    # plt.title('Data Splitting of $g_{il}$ Values')
    plt.yticks([])  # Hide y-axis ticks
    # Show all g_il values on x-axis
    plt.xticks([0.01, 0.04, 0.07, 0.1])
    # size x-axis labels
    plt.tick_params(axis='x', which='major', labelsize=20)
    plt.minorticks_on()

    # Place the legend above the bars
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)

    # Adjust layout for clarity
    plt.tight_layout()
    # Save the plot
    plt.savefig("./results/data_split.pdf", bbox_inches='tight')


def make_fig2(t_plot, gils_plot):
    data_split()
    
    zoom = [50, 75, 15, 50]
    
    plot_fem_data(t_plot,
                  gils_plot,
                  "Fig2_a.svg",
                  zoom)

    plot_fem_zoom(t_plot,
                   gils_plot,
                   zoom,
                   "Fig2_c.svg")

def make_fig3(t_plot, gils_plot):
    vm_errors = get_vm_errordata()
    plot_errorbars_horizontal(vm_errors, 
                              r'$ \mathrm{\Delta}_{V_m}$',
                              "Fig3_a.svg")

    plot_meshdata_vertical(VMMeshPlotData(t_plot),
                          gils_plot[::-1],
                          "Fig3_b.svg")

def make_fig4(gils_plot):
    act_errors = get_act_errordata()
    plot_errorbars_horizontal(act_errors, 
                              r'$ \mathrm{\Delta}_{AT}$',
                              "Fig4_a.svg")

    plot_meshdata_vertical(ActMeshPlotData(),
                          gils_plot[::-1],
                          "Fig4_b.svg")
                          
def make_fig5(t_plot, gils_plot):
    zoom = [55, 75, 15, 45]
    #plot_vm_vertical_zoom(t_plot, experiments, "Fig5_a.pdf", zoom)
    #activation_vertical_zoom( experiments, "Fig5_b.pdf", zoom)

if __name__ == "__main__":
    t_plot = 430
    gils_plot = [0.01, 0.02, 0.025, 0.1]
    
    #make_fig2(t_plot, gils_plot)
    make_fig3(t_plot, gils_plot)
    make_fig4(gils_plot)
    #make_fig5(t_plot, gils_plot)