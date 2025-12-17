import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # You might need: pip install pandas

def run_simulation(N, nu, label):
    print(f"--> Running Case: {label} (Grid: {N}^3, Viscosity: {nu})...")
    
    # --- Setup ---
    L = 2 * np.pi
    dt = 0.005 if N <= 32 else 0.002 # Smaller dt for higher res
    T_final = 1.5 # Short run is enough to see the peak or crash
    
    # Grid
    k = np.fft.fftfreq(N, L/(2*np.pi*N))
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1e-10
    
    # De-aliasing
    kmax_dealias = 2.0/3.0 * np.max(k)
    mask = (np.abs(kx) < kmax_dealias) & (np.abs(ky) < kmax_dealias) & (np.abs(kz) < kmax_dealias)

    # Initial Condition (Taylor-Green)
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)
    
    history_Xi = []
    history_time = []
    
    steps = int(T_final / dt)
    
    # --- Time Loop ---
    for n in range(steps):
        # 1. Vorticity in Spectral
        omega_x_hat = 1j * (ky * w_hat - kz * v_hat)
        omega_y_hat = 1j * (kz * u_hat - kx * w_hat)
        omega_z_hat = 1j * (kx * v_hat - ky * u_hat)
        
        # 2. Cross Product in Real
        ur, vr, wr = np.fft.ifftn(u_hat).real, np.fft.ifftn(v_hat).real, np.fft.ifftn(w_hat).real
        ox, oy, oz = np.fft.ifftn(omega_x_hat).real, np.fft.ifftn(omega_y_hat).real, np.fft.ifftn(omega_z_hat).real
        
        cx = vr*oz - wr*oy
        cy = wr*ox - ur*oz
        cz = ur*oy - vr*ox
        
        cx_hat, cy_hat, cz_hat = np.fft.fftn(cx), np.fft.fftn(cy), np.fft.fftn(cz)
        
        # 3. Leray Projection
        k_dot_c = kx*cx_hat + ky*cy_hat + kz*cz_hat
        rhs_u = cx_hat - kx * k_dot_c / k2
        rhs_v = cy_hat - ky * k_dot_c / k2
        rhs_w = cz_hat - kz * k_dot_c / k2
        
        # 4. Viscosity (or lack thereof)
        if nu > 0:
            rhs_u -= nu * k2 * u_hat
            rhs_v -= nu * k2 * v_hat
            rhs_w -= nu * k2 * w_hat
            
        u_hat += dt * rhs_u * mask
        v_hat += dt * rhs_v * mask
        w_hat += dt * rhs_w * mask
        
        # Audit
        E2 = 0.5 * np.sum(ox**2 + oy**2 + oz**2) * (L/N)**3
        E3 = np.sum(np.sqrt(ur**2 + vr**2 + wr**2)**3) * (L/N)**3
        history_Xi.append(E2 + E3)
        history_time.append(n*dt)
        
        # "Safety Valve" - Stop if it blows up
        if history_Xi[-1] > 20000: 
            print("    ! BLOW-UP DETECTED ! Stopping early.")
            history_Xi.append(np.nan) # Mark end
            history_time.append((n+1)*dt)
            break

    return history_time, history_Xi

# --- Main Driver ---
if __name__ == "__main__":
    plt.figure(figsize=(10, 6))
    
    # 1. Standard Run
    # INCREASED VISCOSITY to 0.05 per our previous discussion (to show clear gap)
    t1, xi1 = run_simulation(N=32, nu=0.05, label="Standard (N=32)") 
    plt.plot(t1, xi1, 'g-', linewidth=2, label=r'MWLA (Standard $\nu=0.05$)')
    
    # 2. High-Res Run (Test Grid Convergence)
    t2, xi2 = run_simulation(N=64, nu=0.05, label="High-Res (N=64)")
    plt.plot(t2, xi2, 'b--', linewidth=2, label=r'Grid Check (N=64)')
    
    # 3. Euler Run (Test Sensitivity)
    t3, xi3 = run_simulation(N=32, nu=0.0, label="Euler (Inviscid)")
    plt.plot(t3, xi3, 'r-', linewidth=2, label=r'Euler Limit ($\nu=0$)')

    # --- SAVE RAW DATA ---
    max_len = max(len(xi1), len(xi2), len(xi3))
    def pad(arr, length):
        return arr + [np.nan]*(length - len(arr))

    df = pd.DataFrame({
        'time': pad(t1, max_len), # Using t1 as base time
        'Xi_Standard': pad(xi1, max_len),
        'Xi_HighRes': pad(xi2, max_len),
        'Xi_Euler': pad(xi3, max_len)
    })
    
    df.to_csv("mwla_validation_data.csv", index=False)
    print("SAVED: mwla_validation_data.csv")

    plt.title("MWLA Validation Suite: Grid Independence & Inviscid Limit")
    plt.xlabel("Time")
    plt.ylabel(r"Ledger Invariant $\Xi(t)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("sim3datavalidation.png")
    plt.show()
    print("DONE. Saved 'sim3datavalidation.png'.")