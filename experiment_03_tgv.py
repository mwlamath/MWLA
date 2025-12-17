import numpy as np
import matplotlib.pyplot as plt
import csv

def run_3d_mwla():
    # --- Parameters ---
    N = 64              # Grid resolution (NxNxN)
    L = 2 * np.pi       # Domain size
    nu = 0.005          # Viscosity
    dt = 0.005          # Time step
    T_final = 2.0       # Final time
    
    # Fourier Grid
    k = np.fft.fftfreq(N, L/(2*np.pi*N))
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1e-10 # Avoid singularity
    
    # De-aliasing Mask
    kmax_dealias = 2.0/3.0 * np.max(k)
    mask = (np.abs(kx) < kmax_dealias) & (np.abs(ky) < kmax_dealias) & (np.abs(kz) < kmax_dealias)

    # --- Initial Condition: Taylor-Green Vortex ---
    # Standard benchmark for 3D energy dissipation
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u) # z-component starts at 0
    
    # Transform to spectral space
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)
    
    history_Xi = []
    history_time = []
    
    print(f"Running 3D MWLA Audit (Grid: {N}^3). This may take a minute...")
    
    steps = int(T_final / dt)
    
    for n in range(steps):
        # 1. Compute nonlinear terms in Real Space
        
        # Get real velocities
        ur = np.fft.ifftn(u_hat).real
        vr = np.fft.ifftn(v_hat).real
        wr = np.fft.ifftn(w_hat).real
        
        # Compute Vorticity (curl u) in Spectral
        omega_x_hat = 1j * (ky * w_hat - kz * v_hat)
        omega_y_hat = 1j * (kz * u_hat - kx * w_hat)
        omega_z_hat = 1j * (kx * v_hat - ky * u_hat)
        
        # Vorticity in Real Space
        omega_x = np.fft.ifftn(omega_x_hat).real
        omega_y = np.fft.ifftn(omega_y_hat).real
        omega_z = np.fft.ifftn(omega_z_hat).real
        
        # Cross Product: u x omega
        cx = vr*omega_z - wr*omega_y
        cy = wr*omega_x - ur*omega_z
        cz = ur*omega_y - vr*omega_x
        
        # Transform Cross Product back
        cx_hat = np.fft.fftn(cx)
        cy_hat = np.fft.fftn(cy)
        cz_hat = np.fft.fftn(cz)
        
        # 2. Project onto Divergence-Free Space (Leray Projection)
        # P(f) = f - k(k.f)/k^2
        k_dot_c = kx*cx_hat + ky*cy_hat + kz*cz_hat
        
        rhs_u = cx_hat - kx * k_dot_c / k2
        rhs_v = cy_hat - ky * k_dot_c / k2
        rhs_w = cz_hat - kz * k_dot_c / k2
        
        # Apply Viscosity
        rhs_u -= nu * k2 * u_hat
        rhs_v -= nu * k2 * v_hat
        rhs_w -= nu * k2 * w_hat
        
        # 3. Time Step (Forward Euler for simplicity/speed)
        u_hat += dt * rhs_u * mask
        v_hat += dt * rhs_v * mask
        w_hat += dt * rhs_w * mask
        
        # --- MWLA AUDIT ---
        # E2: Enstrophy (integral of vorticity squared)
        E2 = 0.5 * np.sum(omega_x**2 + omega_y**2 + omega_z**2) * (L/N)**3
        
        # E3: Kinetic Energy L3 norm (approximated)
        vel_mag = np.sqrt(ur**2 + vr**2 + wr**2)
        E3 = np.sum(vel_mag**3) * (L/N)**3
        
        # MWLA Invariant
        Xi = E2 + E3
        
        history_Xi.append(Xi)
        history_time.append(n*dt)
        
        if n % 50 == 0:
            print(f"Step {n}/{steps} | Xi: {Xi:.4f}")

    # --- CSV Output ---
    csv_filename = "mwla_3d_data.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "Xi"])
        for t, val in zip(history_time, history_Xi):
            writer.writerow([t, val])
    print(f"Data saved to {csv_filename}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(history_time, history_Xi, 'r-', linewidth=2, label=r'3D Ledger Invariant $\Xi(t)$')
    plt.title(r"3D Vortex Stretching (Taylor-Green): $\Xi(t)$ Evolution")
    plt.xlabel("Time")
    plt.ylabel("Invariant Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Corrected Filename
    plt.savefig("3DvortextaylorgreenMWLA.png") 
    plt.show()
    print("3DvortextaylorgreenMWLA.png generated.")

if __name__ == "__main__":
    run_3d_mwla()