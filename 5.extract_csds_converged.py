import pandas as pd

def main():
    # -------------------------------------------------------------
    # 1) Fichier d'entrée (doit être dans le même dossier que ce script)
    # -------------------------------------------------------------
    input_csv = "CSDS_parameters_with_CSDS_model.csv"

    # -------------------------------------------------------------
    # 2) Lecture du CSV complet
    # -------------------------------------------------------------
    print(f"Lecture du fichier : {input_csv}")
    df = pd.read_csv(input_csv)

    # Vérifier que la colonne de convergence existe
    if "csds_converged" not in df.columns:
        raise KeyError(
            "La colonne 'csds_converged' est absente du fichier. "
            "Assure-toi que le script d'ajustement CSDS a bien ajouté cette colonne."
        )

    # -------------------------------------------------------------
    # 3) Filtrage : ne garder que les lignes qui ont convergé
    # -------------------------------------------------------------
    df_converged = df[df["csds_converged"] == True].copy()

    n_total = len(df)
    n_conv = len(df_converged)

    print(f"Nombre total de lignes           : {n_total}")
    print(f"Nombre de lignes convergées      : {n_conv}")
    if n_total > 0:
        print(f"Proportion de cas convergés      : {n_conv / n_total:.2%}")

    # -------------------------------------------------------------
    # 4) Sauvegarde dans un nouveau CSV
    # -------------------------------------------------------------
    output_csv = "CSDS_parameters_CONVERGED_ONLY.csv"
    df_converged.to_csv(output_csv, index=False)

    print(f"\nFichier sauvegardé : {output_csv}")
    print("Ce fichier ne contient que les échantillons où csds_converged == True.")

if __name__ == "__main__":
    main()
