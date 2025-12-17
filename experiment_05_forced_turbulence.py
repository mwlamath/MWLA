import numpy as np
import matplotlib.pyplot as plt
import csv

def run_forced_turbulence():
    print("--> Running Forced Turbulence (Statistically Stationary Test)...")
    
    # --- Parameters ---
    N = 64              # Moderate resolution
    L = 2 * np.pi
    nu = 0.002          # Low viscosity
    dt = 0.005
    T_final = 20.0      # Long run to see long-term stability
    forcing_scale = 0.1 # Strength of the stirring
    
    # Grid
    k = np.fft.fftfreq(N, L/(2*np.pi*N))
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1e-10
    
    # De-aliasing
    kmax_dealias = 2.0/3.0 * np.max(k)
    mask = (np.abs(kx) < kmax_dealias) & (np.abs(ky) < kmax_dealias) & (np.abs(kz) < kmax_dealias)

    # Initial Condition: Random Noise (Start from nothing!)
    np.random.seed(42)
    u_hat = np.random.randn(N,N,N) + 1j*np.random.randn(N,N,N)
    v_hat = np.random.randn(N,N,N) + 1j*np.random.randn(N,N,N)
    w_hat = np.random.randn(N,N,N) + 1j*np.random.randn(N,N,N)
    
    # Filter to low wavenumbers only (Large scale structures)
    k_init_mask = (k2 < 4**2) 
    u_hat *= k_init_mask
    v_hat *= k_init_mask
    w_hat *= k_init_mask
    
    # Project to make divergence free
    k_dot_u = kx*u_hat + ky*v_hat + kz*w_hat
    u_hat -= kx * k_dot_u / k2
    v_hat -= ky * k_dot_u / k2
    w_hat -= kz * k_dot_u / k2

    history_Xi = []
    history_time = []
    
    steps = int(T_final / dt)
    
    for n in range(steps):
        # 1. Standard Navier-Stokes Step
        ur, vr, wr = np.fft.ifftn(u_hat).real, np.fft.ifftn(v_hat).real, np.fft.ifftn(w_hat).real
        
        # Spectral Vorticity
        ox_hat = 1j*(ky*w_hat - kz*v_hat)
        oy_hat = 1j*(kz*u_hat - kx*w_hat)
        oz_hat = 1j*(kx*v_hat - ky*u_hat)
        
        ox, oy, oz = np.fft.ifftn(ox_hat).real, np.fft.ifftn(oy_hat).real, np.fft.ifftn(oz_hat).real
        
        # Cross Product
        cx = vr*oz - wr*oy
        cy = wr*ox - ur*oz
        cz = ur*oy - vr*ox
        
        cx_hat, cy_hat, cz_hat = np.fft.fftn(cx), np.fft.fftn(cy), np.fft.fftn(cz)
        
        # Leray Projection
        k_dot_c = kx*cx_hat + ky*cy_hat + kz*cz_hat
        rhs_u = cx_hat - kx * k_dot_c / k2
        rhs_v = cy_hat - ky * k_dot_c / k2
        rhs_w = cz_hat - kz * k_dot_c / k2
        
        # Viscosity
        rhs_u -= nu * k2 * u_hat
        rhs_v -= nu * k2 * v_hat
        rhs_w -= nu * k2 * w_hat
        
        # 2. THE FORCING TERM (Stirring the pot)
        # We add energy to low wavenumbers (k < 2.5) constantly
        forcing_mask = (k2 > 0.1) & (k2 < 2.5**2)
        
        rhs_u += forcing_scale * u_hat * forcing_mask
        rhs_v += forcing_scale * v_hat * forcing_mask
        rhs_w += forcing_scale * w_hat * forcing_mask
        
        # Time Step
        u_hat += dt * rhs_u * mask
        v_hat += dt * rhs_v * mask
        w_hat += dt * rhs_w * mask
        
        # Audit
        E2 = 0.5 * np.sum(ox**2 + oy**2 + oz**2) * (L/N)**3
        E3 = np.sum(np.sqrt(ur**2 + vr**2 + wr**2)**3) * (L/N)**3
        history_Xi.append(E2 + E3)
        history_time.append(n*dt)
        
        if n % 50 == 0:
            print(f"Step {n}/{steps}: Xi = {history_Xi[-1]:.2f}")

    # --- Save CSV ---
    print("Saving 'forced_turbulence_data.csv'...")
    with open('forced_turbulence_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "Xi"])
        for t, val in zip(history_time, history_Xi):
            writer.writerow([t, val])

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(history_time, history_Xi, 'k-', linewidth=1, label=r'Forced Turbulence $\Xi(t)$')
    # Calculate mean of second half to show stationarity
    stationary_mean = np.mean(history_Xi[int(len(history_Xi)/2):])
    plt.axhline(y=stationary_mean, color='r', linestyle='--', label='Statistically Stationary Mean')
    
    plt.title(r"Long-Time Stability: Forced Turbulence Test")
    plt.xlabel("Time")
    plt.ylabel(r"Ledger Invariant $\Xi$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Figure_Forced.png")
    plt.show()
    print("Graph saved as 'Figure_Forced.png'.")

if __name__ == "__main__":
    run_forced_turbulence()