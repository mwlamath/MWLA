import numpy as np
import matplotlib.pyplot as plt
import csv

def run_mwla_test():
    # --- 1. Setup Grid and Parameters ---
    nx = 256            # Number of grid points
    L = 2.0 * np.pi     # Domain length
    dx = L / nx
    nu = 0.05           # CHANGED: 0.05 allows shock formation (transient growth)
    dt = 0.001          # Time step
    T_final = 2.0       # Final time
    
    x = np.linspace(0, L, nx, endpoint=False)
    
    # Initial Condition: A sine wave
    u = np.sin(x)
    
    # Storage for the MWLA Invariant
    history_Xi = []
    history_time = []
    
    # --- 2. The Solver Loop ---
    u_hat = np.fft.fft(u)
    k = 2 * np.pi * np.fft.fftfreq(nx, d=L/nx)
    
    steps = int(T_final / dt)
    
    print(f"Running MWLA Audit on Burgers' Equation for {steps} steps...")
    
    for n in range(steps):
        # Derivatives in Fourier space
        u_x_hat = 1j * k * u_hat
        u_xx_hat = -k**2 * u_hat
        
        # 2a. Calculate Standard Derivatives (Real Space)
        u_curr = np.fft.ifft(u_hat).real
        u_x = np.fft.ifft(u_x_hat).real
        u_xx = np.fft.ifft(u_xx_hat).real
        
        # 2b. Compute MWLA "Ledger" Entries
        # E2 (Gradient Energy) ~ Integral |grad u|^2
        E2 = np.sum(u_x**2) * dx
        
        # E3 (L3 Norm) ~ Integral |u|^3
        E3 = np.sum(np.abs(u_curr)**3) * dx
        
        # 2c. Update Step (Simple Euler for demo)
        nonlinear = u_curr * u_x
        diffusion = nu * u_xx
        rhs = -nonlinear + diffusion
        
        u_new = u_curr + dt * rhs
        
        # MWLA Invariant Xi(t) = E2 + E3
        Xi = E2 + E3
        
        history_Xi.append(Xi)
        history_time.append(n * dt)
        
        # Update state
        u_hat = np.fft.fft(u_new)

    # --- 3. Save Data to CSV ---
    csv_filename = "mwla_1d_data.csv"
    print(f"Saving data to {csv_filename}...")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "Xi"])
        for t, val in zip(history_time, history_Xi):
            writer.writerow([t, val])
    
    # --- 4. Generate and Save Plot ---
    png_filename = "1DburgerMWLA.png"
    plt.figure(figsize=(10, 6))
    plt.plot(history_time, history_Xi, label=r'Ledger Invariant $\Xi(t)$', color='green', linewidth=2)
    plt.title(f"1D MWLA Audit: Burgers' Equation (Viscosity $\\nu={nu}$)")
    plt.xlabel("Time")
    plt.ylabel(r"Invariant Magnitude $\Xi$")
    plt.axhline(y=history_Xi[0], color='r', linestyle='--', label="Initial State")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Interpret Results for the Graph text
    if max(history_Xi) > history_Xi[0]:
        plt.text(0.5, max(history_Xi)*0.9, "Transient Growth (Shock Formation)", fontsize=12, color='orange', fontweight='bold', ha='center')
    
    plt.savefig(png_filename)
    print(f"Graph saved as {png_filename}")
    plt.show()

if __name__ == "__main__":
    run_mwla_test()