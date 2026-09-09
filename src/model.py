import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.prep_data import prepare_matrices

def train_mlp_model():
    print("🚀 Chargement et préparation des matrices...")
    X_train, X_test, y_train, y_test, scaler = prepare_matrices()

    print("\n🧠 Définition et entraînement de l'architecture MLP...")
    # Configuration du Perceptron Multicouche
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),  # 2 couches cachées (32 et 16 neurones)
        activation='relu',            # Activation ReLU
        solver='adam',                # Optimiseur Adam
        max_iter=500,                 # Nombre max d'itérations
        random_state=42,              # Reproductibilité
        early_stopping=True,          # Arrêt précoce pour éviter le Surapprentissage (Overfitting)
        n_iter_no_change=20
    )

    # Entraînement du modèle sur les données Train
    mlp.fit(X_train, y_train)

    print("✅ Entraînement terminé !")
    print(f"🔹 Nombre d'itérations effectuées : {mlp.n_iter_}")
    print(f"🔹 Perte finale (Loss) : {mlp.loss_:.4f}")

    # Sauvegarde du modèle entraîné
    joblib.dump(mlp, "mlp_model.joblib")
    print("💾 Modèle sauvegardé sous 'mlp_model.joblib'.")

    return mlp, X_test, y_test

if __name__ == "__main__":
    train_mlp_model()