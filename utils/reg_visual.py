import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def verify_cald_theory_refined():
    """
    Refined Visualization for CALD Theoretical Verification.
    Fixes overlapping issues and improves publication-quality aesthetics.
    """
    
    # ==========================================
    # 1. Physics & Simulation Setup (Same as before)
    # ==========================================
    
    def potential(x):
        u_sharp = -2.0 * np.exp(- (x + 2)**2 / (2 * 0.15**2))
        u_flat  = -1.5 * np.exp(- (x - 2)**2 / (2 * 0.8**2))
        confinement = 0.05 * x**2 
        return u_sharp + u_flat + confinement

    def grad_potential(x):
        du_sharp = -2.0 * np.exp(- (x + 2)**2 / (2 * 0.15**2)) * (- (x + 2) / 0.15**2)
        du_flat  = -1.5 * np.exp(- (x - 2)**2 / (2 * 0.8**2)) * (- (x - 2) / 0.8**2)
        d_conf = 0.1 * x
        return du_sharp + du_flat + d_conf

    def complexity_temperature(x):
        # High temp at sharp minimum (-2), Low temp at flat minimum (2)
        c_sharp = 30.0 * np.exp(- (x + 2)**2 / (2 * 0.5**2))
        c_flat  = 1.0 * np.exp(- (x - 2)**2 / (2 * 1.0**2))
        base_temp = 1.0
        return base_temp + c_sharp + c_flat

    # Simulation Params
    n_steps = 200000
    dt = 0.005
    n_particles = 2000
    
    # Run Simulation
    print("Running Simulation (This may take a few seconds)...")
    x_sgld = np.zeros(n_particles)
    x_cald = np.zeros(n_particles)
    
    history_sgld = []
    history_cald = []
    
    for t in range(n_steps):
        # SGLD
        noise_sgld = np.random.normal(0, np.sqrt(2 * dt * 1.0), n_particles)
        x_sgld = x_sgld - grad_potential(x_sgld) * dt + noise_sgld
        
        # CALD
        T_x = complexity_temperature(x_cald)
        noise_cald = np.random.normal(0, 1.0, n_particles) * np.sqrt(2 * T_x * dt)
        x_cald = x_cald - grad_potential(x_cald) * dt + noise_cald
        
        if t % 50 == 0 and t > n_steps // 2:
            history_sgld.append(x_sgld.copy())
            history_cald.append(x_cald.copy())

    data_sgld = np.concatenate(history_sgld)
    data_cald = np.concatenate(history_cald)

    # Theoretical Calculations
    x_grid = np.linspace(-6, 6, 1000)
    u_grid = potential(x_grid)
    T_grid = complexity_temperature(x_grid)
    
    # Gibbs (SGLD)
    p_gibbs = np.exp(-u_grid / 1.0)
    p_gibbs /= np.trapz(p_gibbs, x_grid)
    
    # Fokker-Planck (CALD)
    integrand = grad_potential(x_grid) / T_grid
    integral_term = np.cumsum(integrand) * (x_grid[1] - x_grid[0])
    p_cald_theory = (1.0 / T_grid) * np.exp(-integral_term)
    p_cald_theory /= np.trapz(p_cald_theory, x_grid)

    # ==========================================
    # 2. Advanced Plotting (Aesthetics Overhaul)
    # ==========================================
    
    # Set style params for academic paper
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'font.family': 'sans-serif', # Or 'serif' for LaTeX look
        'grid.alpha': 0.3,
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    
    # Colors (Academic Palette)
    col_loss = '#2C3E50'    # Dark Blue/Grey
    col_temp = '#E74C3C'    # Alizarin Red
    col_sgld_hist = '#95A5A6' # Concrete Grey
    col_sgld_line = '#2C3E50' # Dark Line
    col_cald_hist = '#1ABC9C' # Turquoise/Teal
    col_cald_line = '#16A085' # Darker Teal

    # --- Plot A: Landscape & Temperature ---
    ax1 = axes[0]
    
    # 1. Plot Loss
    ln1 = ax1.plot(x_grid, u_grid, color=col_loss, linewidth=2.5, label=r'Loss Landscape $U(\theta)$')
    ax1.set_xlabel(r'Parameter Space $\theta$', fontweight='bold')
    ax1.set_ylabel('Loss / Energy', color=col_loss, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=col_loss)
    
    # 2. Plot Temperature (Twin Axis)
    ax1b = ax1.twinx()
    ln2 = ax1b.plot(x_grid, T_grid, color=col_temp, linestyle='--', linewidth=2.5, label=r'Temperature $T(\theta) \propto C_{inv}$')
    ax1b.set_ylabel('Adaptive Temperature', color=col_temp, fontweight='bold')
    ax1b.tick_params(axis='y', labelcolor=col_temp)
    ax1b.spines['right'].set_color(col_temp)
    ax1b.spines['left'].set_color(col_loss)
    
    # 3. Clean Annotations (Avoid Overlap)
    # Use vertical lines to mark minima positions
    ax1.axvline(x=-2, color='gray', linestyle=':', alpha=0.5, ymax=0.9)
    ax1.axvline(x=2, color='gray', linestyle=':', alpha=0.5, ymax=0.9)
    
    # Text labels placed at the TOP margin, away from curves
    ax1.text(-2, max(u_grid) + 0.2, "Sharp Minimum\n(Deep)", ha='center', va='bottom', 
             color=col_loss, fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    ax1.text(2, max(u_grid) + 0.2, "Flat Minimum\n(Wide)", ha='center', va='bottom', 
             color=col_loss, fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    # Combined Legend (Upper Center)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=True, fancybox=True, framealpha=0.9)
    ax1.set_title(r'$\bf{A.}$ Landscape & Adaptive Temperature', pad=25)
    ax1.grid(True)

    # --- Plot B: Theoretical Verification ---
    ax2 = axes[1]
    
    # 1. Histograms (Semi-transparent, no edges for cleanliness)
    ax2.hist(data_sgld, bins=100, density=True, color=col_sgld_hist, alpha=0.4, 
             label='SGLD Simulation', edgecolor='none')
    ax2.hist(data_cald, bins=100, density=True, color=col_cald_hist, alpha=0.5, 
             label='CALD Simulation (Ours)', edgecolor='none')
    
    # 2. Theoretical Curves (Thick, crisp lines)
    ax2.plot(x_grid, p_gibbs, color='black', linestyle='--', linewidth=2, 
             label=r'Gibbs Theory ($e^{-U}$)')
    ax2.plot(x_grid, p_cald_theory, color=col_cald_line, linestyle='-', linewidth=3, 
             label=r'Fokker-Planck Theory ($\rho_{ss}$)')
    
    # 3. "Mass Shift" Arrow Annotation (Visualizing Implicit Regularization)
    # Draw an arrow from the Sharp Peak to the Flat Peak to show the effect
    peak_sharp_x = -2
    peak_flat_x = 2
    height_arrow = 0.25
    
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color=col_cald_line)
    a1 = patches.FancyArrowPatch((peak_sharp_x + 0.5, height_arrow), (peak_flat_x - 0.5, height_arrow),
                                 connectionstyle="arc3,rad=-0.2", **kw)
    ax2.add_patch(a1)
    
    ax2.text(0, height_arrow + 0.02, "Entropic Force\n(Implicit Regularization)", 
             ha='center', va='bottom', color=col_cald_line, fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 4. Labels & Legend
    ax2.set_xlabel(r'Parameter Space $\theta$', fontweight='bold')
    ax2.set_ylabel(r'Probability Density', fontweight='bold')
    ax2.set_title(r'$\bf{B.}$ Theoretical Integrity Verification', pad=15)
    
    # Better Legend Position (Outside to avoid any overlap)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.95, fontsize=10)
    ax2.grid(True)
    
    # Add axis limits to keep things tight
    ax2.set_xlim(-6, 6)
    ax1.set_xlim(-4.5, 4.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # Make room for annotation above axes if needed
    
    save_path = 'cald_theory_refined.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Refined plot saved to '{save_path}'")

if __name__ == "__main__":
    verify_cald_theory_refined()