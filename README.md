# Projet Transducteur ultrasonore

Le projet consiste à concevoir un transducteur ultrasonore sans plomb à partir d’une céramique KNN, dans le cadre du BUT Mesures Physiques et en collaboration avec l’entreprise Vermon. Il s’inscrit dans une démarche répondant aux exigences environnementales de la directive RoHS et vise à maîtriser l’ensemble des étapes de réalisation : synthèse du matériau, caractérisations, assemblage du dispositif et analyse du signal obtenu.

## *Script interface graphique*

``` bash 
import tkinter as tk  # Bibliothèque  interface graphique
from tkinter import filedialog, messagebox  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Intégration matplotlib dans Tkinter
import matplotlib.pyplot as plt  # Création de graphiques
import matplotlib as mpl  # Configuration matplotlib
import pandas as pd  # Lecture fichiers Excel-CSV
import numpy as np  # Calculs numériques (FFT et mean)

# ---------------- CONFIGURATION GRAPHIQUE ----------------

# Ajuste les tailles de police dans les graphiques
mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10
})
plt.rcParams['figure.dpi'] = 100  # Résolution des figures

# ---------------- OPTION D'AFFICHAGE ----------------

AFFICHAGE_DENSITE = True  
# True  → affiche en dBm/Hz (densité spectrale)
# False → affiche en dBm (par bin FFT)

# ---------------- FONCTIONS DE NAVIGATION ----------------

def zoom_in():
    """Zoom avant sur le spectre"""
    x_min, x_max = ax2.get_xlim()
    center = (x_min + x_max) / 2  # centre de la zone affichée
    span = (x_max - x_min) * 0.5  # réduit la largeur (zoom)
    ax2.set_xlim(center - span/2, center + span/2)
    canvas.draw()

def zoom_out():
    """Zoom arrière"""
    x_min, x_max = ax2.get_xlim()
    center = (x_min + x_max) / 2
    span = (x_max - x_min) * 2  # agrandit la vue
    ax2.set_xlim(center - span/2, center + span/2)
    canvas.draw()

def move_left():
    """Déplace la vue vers la gauche"""
    x_min, x_max = ax2.get_xlim()
    shift = (x_max - x_min) * 0.2
    ax2.set_xlim(x_min - shift, x_max - shift)
    canvas.draw()

def move_right():
    """Déplace la vue vers la droite"""
    x_min, x_max = ax2.get_xlim()
    shift = (x_max - x_min) * 0.2
    ax2.set_xlim(x_min + shift, x_max + shift)
    canvas.draw()

# ---------------- FONCTION PRINCIPALE ----------------

def lancer_programme():
    """Charge un fichier, calcule la FFT et affiche les résultats"""

    # Ouvre une fenêtre pour sélectionner un fichier
    fichier = filedialog.askopenfilename(
        title="Sélectionner un fichier Excel ou CSV",
        filetypes=[("Fichiers Excel/CSV", "*.xlsx *.xls *.csv")]
    )

    try:
        # Lecture du fichier (Excel sinon CSV)
        try:
            df = pd.read_excel(fichier)
        except Exception:
            df = pd.read_csv(fichier, sep=',', engine='python')

        # Vérifie qu'il y a au moins 2 colonnes
        if df.shape[1] < 2:
            raise ValueError("Le fichier doit contenir au moins deux colonnes.")

        # Récupération des colonnes (temps et signal)
        temps = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        signal = pd.to_numeric(df.iloc[:, 1], errors='coerce')

        # Supprime les valeurs invalides
        valid = temps.notna() & signal.notna()
        temps, signal = temps[valid], signal[valid]

        if len(temps) < 2:
            raise ValueError("Données insuffisantes.")

        # Supprime la composante continue (centrage)
        signal = signal - np.mean(signal)

        # ---------------- SIGNAL TEMPOREL ----------------

        ax1.clear()
        ax1.plot(temps, signal, color='#0258BA', linewidth=2.0)
        ax1.set_facecolor('#F8F9FA')
        ax1.set_title("Signal temporel", fontweight='bold')
        ax1.set_xlabel("Temps (s)")
        ax1.set_ylabel("Amplitude (V)")
        ax1.grid(True, linestyle='--')

        # ---------------- PARAMÈTRES FFT ----------------

        dt = np.mean(np.diff(temps))  # pas de temps moyen

        if dt <= 0 or np.isnan(dt):
            raise ValueError("Pas de temps invalide")

        fs = 1 / dt  # fréquence d'échantillonnage
        N = len(signal)  # nombre de points
        R = 50.0  # impédance (ohms)

        # ---------------- FENÊTRAGE ----------------

        window = np.hanning(N)  # fenêtre de Hanning
        signal_win = signal * window
        coherent_gain = np.mean(window)  # correction amplitude

        # ---------------- FFT ----------------

        fft_vals = np.fft.fft(signal_win)
        freqs = np.fft.fftfreq(N, d=dt)

        # On garde seulement la moitié positive
        half = N // 2
        freqs = freqs[:half]
        fft_vals = fft_vals[:half]

        # ---------------- AMPLITUDE ----------------

        fft_amplitude = np.abs(fft_vals) / N
        fft_amplitude /= coherent_gain
        fft_amplitude[1:] *= 2  # correction spectre simple face

        # Conversion en RMS
        fft_rms = fft_amplitude / np.sqrt(2)

        # ---------------- PUISSANCE ----------------

        puissance = (fft_rms ** 2) / R
        puissance[puissance <= 0] = 1e-20  # éviter log(0)

        fft_dBm = 10 * np.log10(puissance / 1e-3)

        # ---------------- DENSITÉ SPECTRALE ----------------

        df = fs / N  # largeur d'un bin
        puissance_hz = puissance / df
        fft_dBm_Hz = 10 * np.log10(puissance_hz / 1e-3)

        # Choix affichage
        spectre = fft_dBm_Hz if AFFICHAGE_DENSITE else fft_dBm

        # ---------------- AFFICHAGE FFT ----------------

        ax2.clear()
        ax2.plot(freqs, spectre, color='#e53935', linewidth=2.0)
        ax2.set_facecolor('#f8f9fa')
        ax2.set_title("Spectre fréquentiel (FFT)", fontweight='bold')
        ax2.set_xlabel("Fréquence (Hz)")
        ax2.set_ylabel("Puissance (dBm/Hz)" if AFFICHAGE_DENSITE else "Puissance (dBm)")
        ax2.grid(True, linestyle='--')

        # ---------------- BANDE PASSANTE (-6 dB) ----------------

        Amax = np.max(spectre)
        seuil = Amax - 6  # seuil -6 dB

        indices = np.where(spectre >= seuil)[0] # Recherche de f_min et f_max

        if len(indices) >= 2:
            f_min = freqs[indices[0]]
            f_max = freqs[indices[-1]]

            f_centrale = (f_min + f_max) / 2
            bande_passante = f_max - f_min
            bande_passante_prct = (bande_passante / f_centrale) * 100
        else:
            f_centrale = np.nan # Si il n'y en a pas ou impossibilité de la calculer on affiche "nan"
            bande_passante = np.nan
            bande_passante_prct = np.nan

        # Ajout des repères graphiques
        ax2.axvline(f_centrale, color='black', linestyle='--') # Ligne montrant la frèquence centrale
        ax2.axhline(seuil, color='black', linestyle='--') # Ligne montrant le seuil à -6 dB

        if not np.isnan(bande_passante):
            ax2.axvspan(f_min, f_max, color='orange', alpha=0.25) # Zone correspondant à la bande passante

        # Texte d'information
        texte = (
            f"Fréquence centrale : {f_centrale*1e-6:.2f} MHz\n"
            f"f_max : {f_max*1e-6:.2f} MHz\n"
            f"f_min : {f_min*1e-6:.2f} MHz\n"
            f"Bande passante (%) : {bande_passante_prct:.2f} %"
        ) # Tout est convertit en MHz pour une étude en Hz il faut retirer la conversion car elle n'est pas automatique

        ax2.text(
            0.98, 0.95, texte,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        # Limite affichage fréquence
        ax2.set_xlim(0, max(freqs) / 4)

        canvas.draw()

    except Exception as e:
        messagebox.showerror("Erreur", f"{e}")

# ---------------- INTERFACE GRAPHIQUE ----------------

fenetre = tk.Tk()
fenetre.title("Analyseur de Signal - Oscilloscope & FFT")
fenetre.geometry("950x800")
fenetre.configure(bg="#e9ecef")

# Cadre principal
cadre_principal = tk.Frame(fenetre, bg="#dee2e6", bd=3, relief="ridge")
cadre_principal.pack(expand=True, fill="both", padx=30, pady=30)

conteneur = tk.Frame(cadre_principal, bg="#dee2e6")
conteneur.pack(expand=True, fill="both", padx=20, pady=20)

# Fonction pour sauvegarder une image
def enregistrer_image():
    fichier = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("Image JPEG", "*.jpg"), ("Image PNG", "*.png")],
        title="Enregistrer le graphique"
    )
    if fichier:
        fig.savefig(fichier, dpi=300, bbox_inches="tight")

# Bouton lancement
bouton_lancer = tk.Button(
    conteneur, text="▶", bg="#2ecc71", fg="white",
    font=("Arial", 30, "bold"), command=lancer_programme
)
bouton_lancer.grid(row=0, column=0, padx=18, pady=200)

# Création des graphiques
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
fig.subplots_adjust(hspace=0.4)
fig.patch.set_facecolor('#dee2e6')

# Intégration dans Tkinter
canvas = FigureCanvasTkAgg(fig, master=conteneur)
canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

# Boutons de contrôle
cadre_zoom = tk.Frame(conteneur, bg="#dee2e6")
cadre_zoom.grid(row=1, column=1, pady=10)

btn_style = {"font": ("Arial", 14, "bold"), "width": 4}

tk.Button(cadre_zoom, text="−", command=zoom_out, bg="#6c757d", fg="white", **btn_style).grid(row=0, column=2)
tk.Button(cadre_zoom, text="+", command=zoom_in, bg="#6c757d", fg="white", **btn_style).grid(row=0, column=1)
tk.Button(cadre_zoom, text="←", command=move_left, bg="#6c757d", fg="white", **btn_style).grid(row=0, column=0)
tk.Button(cadre_zoom, text="→", command=move_right, bg="#6c757d", fg="white", **btn_style).grid(row=0, column=3)
tk.Button(cadre_zoom, text="💾", command=enregistrer_image, bg="#17a2b8", fg="white", **btn_style).grid(row=0, column=4)

# Ajustement layout
conteneur.columnconfigure(1, weight=1)
conteneur.rowconfigure(0, weight=1)

# Lancement de l'application
fenetre.mainloop()
```