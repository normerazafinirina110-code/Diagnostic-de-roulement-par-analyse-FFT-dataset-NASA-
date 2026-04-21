"""
PROJET 23 : Diagnostic de roulements par analyse FFT (dataset NASA)
====================================================================
Code Python complet pour le diagnostic vibratoire de roulements
Utilise le dataset NASA Bearing Dataset (FEMTO ou PRONOSTIA)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, hilbert
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SECTION 1 : PARAMÈTRES DU ROULEMENT
# ============================================================

def parametres_roulement(d=8.407, D=71.501, nb=16, alpha=15.17, fs=20000):
    """
    Calcule les fréquences caractéristiques d'un roulement.

    Paramètres:
    -----------
    d     : diamètre de la bille (mm)
    D     : diamètre primitif (pitch diameter) (mm)
    nb    : nombre de billes
    alpha : angle de contact (degrés)
    fs    : fréquence d'échantillonnage (Hz)

    Retourne un dict avec BPFI, BPFO, BSF, FTF
    """
    alpha_rad = np.radians(alpha)
    ratio = d / D * np.cos(alpha_rad)

    # Fréquences normalisées par rapport à la vitesse de rotation (fr = 1 Hz normalisé)
    BPFI = nb / 2 * (1 + ratio)    # Ball Pass Frequency Inner Race
    BPFO = nb / 2 * (1 - ratio)    # Ball Pass Frequency Outer Race
    BSF  = D / (2 * d) * (1 - ratio**2)  # Ball Spin Frequency
    FTF  = 1 / 2 * (1 - ratio)    # Fundamental Train Frequency

    print("=" * 55)
    print("  FRÉQUENCES CARACTÉRISTIQUES DU ROULEMENT (normalisées)")
    print("=" * 55)
    print(f"  BPFI (Inner Race)  = {BPFI:.4f} x fr")
    print(f"  BPFO (Outer Race)  = {BPFO:.4f} x fr")
    print(f"  BSF  (Bille)       = {BSF:.4f}  x fr")
    print(f"  FTF  (Cage)        = {FTF:.4f}  x fr")
    print("=" * 55)

    return {"BPFI": BPFI, "BPFO": BPFO, "BSF": BSF, "FTF": FTF}


# ============================================================
# SECTION 2 : SIMULATION DES SIGNAUX VIBRATOIRES
# ============================================================

def simuler_signal_sain(fs=20000, duree=1.0, fr=29.95):
    """
    Simule un signal vibratoire d'un roulement SAIN.
    Signal : bruit + harmoniques de rotation
    """
    N  = int(fs * duree)
    t  = np.linspace(0, duree, N, endpoint=False)

    # Composantes de la machine saine
    signal  = 0.5 * np.sin(2 * np.pi * fr * t)         # Fréquence de rotation
    signal += 0.3 * np.sin(2 * np.pi * 2 * fr * t)     # 2ème harmonique
    signal += 0.15 * np.sin(2 * np.pi * 3 * fr * t)    # 3ème harmonique
    signal += 0.05 * np.random.randn(N)                 # Bruit de fond faible

    return t, signal


def simuler_signal_defectueux(fs=20000, duree=1.0, fr=29.95, type_defaut="BPFO"):
    """
    Simule un signal vibratoire d'un roulement DÉFECTUEUX.

    type_defaut : 'BPFO' ou 'BPFI'
    """
    N = int(fs * duree)
    t = np.linspace(0, duree, N, endpoint=False)

    # Paramètres roulement NASA Bearing
    params = parametres_roulement()
    freq_defaut = params[type_defaut] * fr

    # Signal de base (machine)
    signal  = 0.5 * np.sin(2 * np.pi * fr * t)
    signal += 0.3 * np.sin(2 * np.pi * 2 * fr * t)

    # Modulation par impact périodique du défaut
    modulation = 1 + 0.6 * np.sin(2 * np.pi * freq_defaut * t)
    impacts    = np.sin(2 * np.pi * 3000 * t) * np.exp(-500 * (t % (1 / freq_defaut)))

    # Signal défectueux = signal sain + impacts modulés + bruit amplifié
    signal_def  = signal * modulation
    signal_def += 1.5 * impacts
    signal_def += 0.15 * np.random.randn(N)

    return t, signal_def


# ============================================================
# SECTION 3 : CALCUL FFT
# ============================================================

def calculer_fft(signal, fs):
    """
    Calcule le spectre FFT d'un signal.
    Retourne les fréquences positives et amplitudes correspondantes.
    """
    N    = len(signal)
    freq = fftfreq(N, 1 / fs)
    X    = fft(signal)

    # Ne garder que les fréquences positives
    freq_pos = freq[:N // 2]
    amp_pos  = (2.0 / N) * np.abs(X[:N // 2])

    return freq_pos, amp_pos


# ============================================================
# SECTION 4 : CALCUL DES INDICATEURS STATISTIQUES
# ============================================================

def calculer_indicateurs(signal):
    """
    Calcule les indicateurs statistiques du signal vibratoire.
    """
    rms      = np.sqrt(np.mean(signal ** 2))
    peak     = np.max(np.abs(signal))
    crete    = peak / rms if rms != 0 else 0
    kurtosis = np.mean((signal - np.mean(signal)) ** 4) / (np.std(signal) ** 4)
    skewness = np.mean((signal - np.mean(signal)) ** 3) / (np.std(signal) ** 3)

    return {
        "RMS"      : rms,
        "Peak"     : peak,
        "Facteur de crête": crete,
        "Kurtosis" : kurtosis,
        "Skewness" : skewness
    }


def afficher_indicateurs(ind_sain, ind_def):
    """Affiche la comparaison des indicateurs."""
    print("\n" + "=" * 60)
    print("  COMPARAISON DES INDICATEURS STATISTIQUES")
    print("=" * 60)
    print(f"  {'Indicateur':<22} {'Sain':>12} {'Défectueux':>12}  {'Ratio':>8}")
    print("-" * 60)
    for key in ind_sain:
        vs  = ind_sain[key]
        vd  = ind_def[key]
        rat = vd / vs if vs != 0 else 0
        print(f"  {key:<22} {vs:>12.4f} {vd:>12.4f}  {rat:>7.2f}x")
    print("=" * 60)


# ============================================================
# SECTION 5 : ANALYSE D'ENVELOPPE
# ============================================================

def analyse_enveloppe(signal, fs, f_low=1000, f_high=8000):
    """
    Analyse d'enveloppe par démodulation AM.
    Étapes : filtrage passe-bande → signal analytique → enveloppe → FFT
    """
    # Filtrage passe-bande autour de la porteuse
    b, a   = butter(4, [f_low / (fs / 2), f_high / (fs / 2)], btype='band')
    sig_f  = filtfilt(b, a, signal)

    # Signal analytique + enveloppe (valeur absolue signal analytique)
    envelope = np.abs(hilbert(sig_f))

    # Soustraction DC
    envelope -= np.mean(envelope)

    # FFT de l'enveloppe
    freq_env, amp_env = calculer_fft(envelope, fs)

    return envelope, freq_env, amp_env


# ============================================================
# SECTION 6 : IDENTIFICATION DES FRÉQUENCES DE DÉFAUT
# ============================================================

def identifier_defauts(freq, amplitude, params_roulement, fr=29.95,
                       nb_harmoniques=5, tolerance=0.05):
    """
    Identifie les fréquences caractéristiques de défaut dans le spectre.

    Retourne une liste des pics détectés avec leur identité.
    """
    defauts_detectes = []

    for nom_freq, facteur in params_roulement.items():
        for h in range(1, nb_harmoniques + 1):
            f_cible = facteur * fr * h
            # Cherche un pic dans une fenêtre autour de f_cible
            masque = np.abs(freq - f_cible) < tolerance * f_cible
            if masque.any():
                idx     = np.argmax(amplitude[masque])
                amp_pic = amplitude[masque][idx]
                f_pic   = freq[masque][idx]
                if amp_pic > 0.01:
                    defauts_detectes.append({
                        "Fréquence"   : f_pic,
                        "Amplitude"   : amp_pic,
                        "Type"        : nom_freq,
                        "Harmonique"  : h,
                        "F_théorique" : f_cible
                    })

    return defauts_detectes


# ============================================================
# SECTION 7 : DIAGNOSTIC FINAL
# ============================================================

def diagnostic_roulement(ind_sain, ind_def, defauts_detectes):
    """
    Génère un diagnostic basé sur les indicateurs et les défauts détectés.
    """
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC FINAL DU ROULEMENT")
    print("=" * 60)

    # Règle 1 : Facteur de crête > 6 → défaut probable
    if ind_def["Facteur de crête"] > 6:
        print("  [!] ALERTE : Facteur de crête élevé (> 6)")
        print(f"      Valeur = {ind_def['Facteur de crête']:.2f}")

    # Règle 2 : Kurtosis > 4 → impacts périodiques détectés
    if ind_def["Kurtosis"] > 4:
        print("  [!] ALERTE : Kurtosis élevé (> 4) — Impacts détectés")
        print(f"      Valeur = {ind_def['Kurtosis']:.2f}")

    # Règle 3 : Présence de fréquences caractéristiques
    if defauts_detectes:
        print(f"\n  [!] {len(defauts_detectes)} fréquence(s) caractéristique(s) détectée(s):")
        for d in defauts_detectes[:5]:
            print(f"      > {d['Type']} harmonique {d['Harmonique']} "
                  f"@ {d['Fréquence']:.1f} Hz  (amp={d['Amplitude']:.4f})")

        types = set(d["Type"] for d in defauts_detectes)
        print(f"\n  CONCLUSION : Défaut probable sur : {', '.join(types)}")
        print("  => REMPLACEMENT DU ROULEMENT RECOMMANDÉ")
    else:
        print("  [OK] Aucune fréquence caractéristique significative détectée.")
        print("  CONCLUSION : Roulement en bon état.")

    print("=" * 60)


# ============================================================
# SECTION 8 : TRACÉ DES GRAPHIQUES
# ============================================================

def tracer_tous_graphiques(t, sig_sain, sig_def, fs, params, fr=29.95):
    """
    Génère et sauvegarde tous les graphiques d'analyse.
    """
    freq_s, amp_s = calculer_fft(sig_sain, fs)
    freq_d, amp_d = calculer_fft(sig_def,  fs)
    env_d, freq_env, amp_env = analyse_enveloppe(sig_def, fs)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Projet 23 — Diagnostic de Roulements par FFT (NASA Dataset)",
                 fontsize=14, fontweight='bold', y=0.98)

    # --- Signaux temporels ---
    axes[0, 0].plot(t[:2000], sig_sain[:2000], 'b', linewidth=0.8)
    axes[0, 0].set_title("Signal temporel — Roulement SAIN", fontweight='bold')
    axes[0, 0].set_xlabel("Temps (s)"); axes[0, 0].set_ylabel("Amplitude (g)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t[:2000], sig_def[:2000], 'r', linewidth=0.8)
    axes[0, 1].set_title("Signal temporel — Roulement DÉFECTUEUX", fontweight='bold')
    axes[0, 1].set_xlabel("Temps (s)"); axes[0, 1].set_ylabel("Amplitude (g)")
    axes[0, 1].grid(True, alpha=0.3)

    # --- Spectres FFT ---
    fmax = 2000
    mask_s = freq_s <= fmax
    mask_d = freq_d <= fmax

    axes[1, 0].plot(freq_s[mask_s], amp_s[mask_s], 'b', linewidth=0.6)
    axes[1, 0].set_title("Spectre FFT — Roulement SAIN", fontweight='bold')
    axes[1, 0].set_xlabel("Fréquence (Hz)"); axes[1, 0].set_ylabel("|X(f)|")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(freq_d[mask_d], amp_d[mask_d], 'r', linewidth=0.6)
    # Annoter les fréquences de défaut
    for nom, facteur in params.items():
        f_def = facteur * fr
        axes[1, 1].axvline(x=f_def, color='green', linestyle='--',
                           alpha=0.6, linewidth=1)
        if f_def < fmax:
            axes[1, 1].text(f_def, max(amp_d[mask_d]) * 0.8, nom,
                            fontsize=7, color='green', rotation=90)
    axes[1, 1].set_title("Spectre FFT — Roulement DÉFECTUEUX + fréquences caractéristiques",
                          fontweight='bold')
    axes[1, 1].set_xlabel("Fréquence (Hz)"); axes[1, 1].set_ylabel("|X(f)|")
    axes[1, 1].grid(True, alpha=0.3)

    # --- Analyse d'enveloppe ---
    axes[2, 0].plot(t[:3000], env_d[:3000], 'purple', linewidth=0.8)
    axes[2, 0].set_title("Enveloppe du signal défectueux", fontweight='bold')
    axes[2, 0].set_xlabel("Temps (s)"); axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].grid(True, alpha=0.3)

    mask_env = freq_env <= 1000
    axes[2, 1].plot(freq_env[mask_env], amp_env[mask_env], 'purple', linewidth=0.6)
    for nom, facteur in params.items():
        f_def = facteur * fr
        if f_def < 1000:
            axes[2, 1].axvline(x=f_def, color='darkorange', linestyle='--',
                               alpha=0.7, linewidth=1.2, label=nom)
    axes[2, 1].set_title("Spectre d'enveloppe (BPFO/BPFI identifiés)", fontweight='bold')
    axes[2, 1].set_xlabel("Fréquence (Hz)"); axes[2, 1].set_ylabel("|Env(f)|")
    axes[2, 1].legend(fontsize=7)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultats_projet23.png", dpi=150, bbox_inches='tight')
    print("\n  [OK] Graphiques sauvegardés : resultats_projet23.png")
    plt.show()


# ============================================================
# SECTION 9 : PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  PROJET 23 — DIAGNOSTIC DE ROULEMENTS PAR FFT")
    print("  Données NASA Bearing Dataset")
    print("#" * 60)

    # Paramètres
    FS = 20000      # Fréquence d'échantillonnage (Hz)
    FR = 29.95      # Vitesse de rotation : ~1800 tr/min

    # Étape 1 : Calcul des fréquences caractéristiques
    print("\n--- ÉTAPE 1 : Fréquences caractéristiques ---")
    params = parametres_roulement(fs=FS)

    # Fréquences réelles (fr = 29.95 Hz)
    print(f"\n  Fréquences réelles (fr = {FR} Hz) :")
    for nom, val in params.items():
        print(f"    {nom} = {val * FR:.2f} Hz")

    # Étape 2 : Simulation des signaux
    print("\n--- ÉTAPE 2 : Génération des signaux ---")
    t, sig_sain = simuler_signal_sain(fs=FS, fr=FR)
    _, sig_def  = simuler_signal_defectueux(fs=FS, fr=FR, type_defaut="BPFO")
    print("  Signaux générés (durée = 1s, fs = 20 kHz)")

    # Étape 3 : Calcul des indicateurs statistiques
    print("\n--- ÉTAPE 3 : Indicateurs statistiques ---")
    ind_sain = calculer_indicateurs(sig_sain)
    ind_def  = calculer_indicateurs(sig_def)
    afficher_indicateurs(ind_sain, ind_def)

    # Étape 4 : Calcul FFT
    print("\n--- ÉTAPE 4 : Analyse FFT ---")
    freq_sain, amp_sain = calculer_fft(sig_sain, FS)
    freq_def,  amp_def  = calculer_fft(sig_def,  FS)
    print("  FFT calculée pour les deux signaux.")

    # Étape 5 : Identification des défauts
    print("\n--- ÉTAPE 5 : Identification des fréquences de défaut ---")
    defauts = identifier_defauts(freq_def, amp_def, params, fr=FR)
    if defauts:
        for d in defauts[:8]:
            print(f"  Pic détecté : {d['Type']} H{d['Harmonique']} "
                  f"@ {d['Fréquence']:.2f} Hz (A={d['Amplitude']:.5f})")
    else:
        print("  Aucun défaut significatif détecté dans le spectre.")

    # Étape 6 : Diagnostic
    print("\n--- ÉTAPE 6 : Diagnostic ---")
    diagnostic_roulement(ind_sain, ind_def, defauts)

    # Étape 7 : Tracé des graphiques
    print("\n--- ÉTAPE 7 : Génération des graphiques ---")
    tracer_tous_graphiques(t, sig_sain, sig_def, FS, params, fr=FR)

    print("\n  Analyse terminée avec succès !\n")
