import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Correction taille des polices matplotlib
mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10
})
plt.rcParams['figure.dpi'] = 100

CHEMIN_FICHIER = r"C:\Users\Dorian\Downloads\CH1.CSV"

# OPTION D'AFFICHAGE DU SPECTRE
AFFICHAGE_DENSITE = True # True  → dBm / Hz & False → dBm / bin

# =========================================================
# FONCTIONS D'INTERACTION
# =========================================================
def zoom_in():
    x_min, x_max = ax2.get_xlim()
    center = (x_min + x_max) / 2
    span = (x_max - x_min) * 0.5
    ax2.set_xlim(center - span/2, center + span/2)
    canvas.draw()

def zoom_out():
    x_min, x_max = ax2.get_xlim()
    center = (x_min + x_max) / 2
    span = (x_max - x_min) * 2
    ax2.set_xlim(center - span/2, center + span/2)
    canvas.draw()

def move_left():
    x_min, x_max = ax2.get_xlim()
    shift = (x_max - x_min) * 0.2
    ax2.set_xlim(x_min - shift, x_max - shift)
    canvas.draw()

def move_right():
    x_min, x_max = ax2.get_xlim()
    shift = (x_max - x_min) * 0.2
    ax2.set_xlim(x_min + shift, x_max + shift)
    canvas.draw()

# =========================================================
# FONCTION PRINCIPALE
# =========================================================
def lancer_programme():

    fichier = filedialog.askopenfilename(
        title="Sélectionner un fichier Excel ou CSV",
        filetypes=[("Fichiers Excel/CSV", "*.xlsx *.xls *.csv")]
    )

    try:
        try:
            df = pd.read_excel(fichier)
        except Exception:
            df = pd.read_csv(fichier, sep=',', engine='python')

        if df.shape[1] < 2:
            raise ValueError("Le fichier doit contenir au moins deux colonnes.")

        temps = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        signal = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        valid = temps.notna() & signal.notna()
        temps, signal = temps[valid], signal[valid]

        if len(temps) < 2:
            raise ValueError("Données insuffisantes.")

        signal = signal - np.mean(signal)

        ax1.clear()
        ax1.plot(temps, signal, color='#0258BA', linewidth=2.0)
        ax1.set_facecolor('#F8F9FA')
        ax1.set_title("Signal temporel", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Temps (s)")
        ax1.set_ylabel("Amplitude (V)")
        ax1.grid(True, linestyle='--', alpha=1)

        dt = np.mean(np.diff(temps))
        if dt <= 0 or np.isnan(dt):
            raise ValueError("Pas de temps invalide")

        fs = 1 / dt
        N = len(signal)
        R = 50.0

        window = np.hanning(N)
        signal_win = signal * window
        coherent_gain = np.mean(window)

        fft_vals = np.fft.fft(signal_win)
        freqs = np.fft.fftfreq(N, d=dt)

        half = N // 2
        freqs = freqs[:half]
        fft_vals = fft_vals[:half]

        fft_amplitude = np.abs(fft_vals) / N
        fft_amplitude /= coherent_gain
        fft_amplitude[1:] *= 2

        fft_rms = fft_amplitude / np.sqrt(2)
        puissance = (fft_rms ** 2) / R
        puissance[puissance <= 0] = 1e-20

        fft_dBm = 10 * np.log10(puissance / 1e-3)

        df = fs / N
        puissance_hz = puissance / df
        fft_dBm_Hz = 10 * np.log10(puissance_hz / 1e-3)

        spectre = fft_dBm_Hz if AFFICHAGE_DENSITE else fft_dBm

        ax2.clear()
        ax2.plot(freqs, spectre, color='#e53935', linewidth=2.0)
        ax2.set_facecolor('#f8f9fa')
        ax2.set_title("Spectre fréquentiel (FFT)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Fréquence (Hz)")
        ax2.set_ylabel("Puissance (dBm)" if AFFICHAGE_DENSITE else "Puissance (dBm / bin)")
        ax2.grid(True, linestyle='--', alpha=1)

        Amax = np.max(spectre)
        idx_peak = np.argmax(spectre)
        f_centrale = freqs[idx_peak]

        seuil = Amax - 6
        indices = np.where(spectre >= seuil)[0]

        if len(indices) >= 2:
            f_min = freqs[indices[0]]
            f_max = freqs[indices[-1]]
            bande_passante = f_max - f_min
            bande_passante_prct = (bande_passante / f_centrale) * 100
        else:
            bande_passante = np.nan
            bande_passante_prct = np.nan

        ax2.axvline(f_centrale, color='black', linestyle='--', alpha=0.8)
        ax2.axhline(seuil, color='black', linestyle='--', alpha=0.7)

        if not np.isnan(bande_passante):
            ax2.axvspan(f_min, f_max, color='orange', alpha=0.25)

        texte = (
            f"Fréquence centrale : {f_centrale*1e-6:.2f} MHz\n"
            f"f_max : {f_max*1e-6:.2f} MHz\n"
            f"f_min : {f_min*1e-6:.2f} MHz\n"
            f"Bande passante (%) : {bande_passante_prct:.2f} %"
        )

        ax2.text(
            0.98, 0.95, texte,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85)
        )

        ax2.set_xlim(0, max(freqs) / 4)
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Erreur", f"{e}")

# =========================================================
# INTERFACE GRAPHIQUE
# =========================================================
fenetre = tk.Tk()
fenetre.title("Analyseur de Signal - Oscilloscope & FFT")
fenetre.geometry("950x800")
fenetre.configure(bg="#e9ecef")

cadre_principal = tk.Frame(fenetre, bg="#dee2e6", bd=3, relief="ridge")
cadre_principal.pack(expand=True, fill="both", padx=30, pady=30)

conteneur = tk.Frame(cadre_principal, bg="#dee2e6")
conteneur.pack(expand=True, fill="both", padx=20, pady=20)

def enregistrer_image():
    fichier = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("Image JPEG", "*.jpg"), ("Image PNG", "*.png")],
        title="Enregistrer le graphique"
    )
    if fichier:
        fig.savefig(fichier, dpi=300, bbox_inches="tight")

bouton_lancer = tk.Button(
    conteneur, text="▶", bg="#2ecc71", fg="white",
    font=("Arial", 24, "bold"), command=lancer_programme,
    width=3, height=1
)
bouton_lancer.grid(row=0, column=0, padx=20, pady=230, sticky="n")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
fig.subplots_adjust(hspace=0.4)
fig.patch.set_facecolor('#dee2e6')

canvas = FigureCanvasTkAgg(fig, master=conteneur)
canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=(10, 0), sticky="nsew")

cadre_zoom = tk.Frame(conteneur, bg="#dee2e6")
cadre_zoom.grid(row=1, column=1, pady=10)

btn_style = {"font": ("Arial", 14, "bold"), "width": 4, "height": 1}

tk.Button(cadre_zoom, text="−", command=zoom_out, bg="#6c757d", fg="white", **btn_style).grid(row=0, column=2, padx=5)
tk.Button(cadre_zoom, text="+", command=zoom_in, bg="#6c757d", fg="white", **btn_style).grid(row=0, column=1, padx=5)
tk.Button(cadre_zoom, text="←", command=move_left, bg="#6c757d", fg="white", **btn_style).grid(row=0, column=0, padx=5)
tk.Button(cadre_zoom, text="→", command=move_right, bg="#6c757d", fg="white", **btn_style).grid(row=0, column=3, padx=5)
tk.Button(cadre_zoom, text="💾", command=enregistrer_image, bg="#17a2b8", fg="white", **btn_style).grid(row=0, column=4, padx=5)

conteneur.columnconfigure(1, weight=1)
conteneur.rowconfigure(0, weight=1)

fenetre.mainloop()
