import json
import zipfile
import numpy as np
from tqdm import tqdm
from joblib import dump
from datetime import datetime
import os

# Paths
zip_path = "data/dota_games.zip"
heroes_json_path = "data/heroes.json"
model_save_dir = "models/"
os.makedirs(model_save_dir, exist_ok=True)


# --------- HERO DATA LOADING ---------
def load_hero_names(path):
    """Load hero names into a dictionary."""
    with open(path, "r") as f:
        heroes = json.load(f)
    return {hero["id"]: hero["api_name"] for hero in heroes}


hero_mapping = load_hero_names(heroes_json_path)


# --------- DATA LOADER CLASS ---------
class DotaDataLoader:
    def __init__(self, zip_path, hero_mapping, batch_size=1000, games_to_process=None):
        self.zip_path = zip_path
        self.hero_mapping = hero_mapping
        self.batch_size = batch_size
        self.games_to_process = games_to_process
        self.hero_ids = sorted(hero_mapping.keys())  # Consistent hero indexing
        self.hero_index = {hero_id: i for i,
                           hero_id in enumerate(self.hero_ids)}

    def load_batches(self):
        """Yields batches of X and y."""
        with zipfile.ZipFile(self.zip_path, "r") as zip_file:
            json_files = [
                name for name in zip_file.namelist() if name.endswith(".json")
            ]
            if self.games_to_process:
                json_files = json_files[: self.games_to_process]

            X_batch = np.zeros(
                (self.batch_size, len(self.hero_ids) + 6), dtype=np.float32
            )
            y_batch = []
            batch_index = 0

            for file_name in tqdm(json_files, desc="Loading Batches"):
                try:
                    with zip_file.open(file_name, "r") as f:
                        game_data = json.load(f)
                        result = game_data["result"]

                        # Match-level features
                        match_features = [
                            result["radiant_score"],
                            result["dire_score"],
                            result["duration"],
                            result["tower_status_radiant"],
                            result["tower_status_dire"],
                            result["game_mode"],
                        ]

                        # Hero presence
                        radiant_heroes = []
                        dire_heroes = []
                        for player in result["players"]:
                            if player["player_slot"] < 128:
                                radiant_heroes.append(player["hero_id"])
                            else:
                                dire_heroes.append(player["hero_id"])

                        row = np.zeros(len(self.hero_ids) +
                                       6, dtype=np.float32)
                        for hero_id in radiant_heroes:
                            row[self.hero_index[hero_id]] = 1
                        for hero_id in dire_heroes:
                            row[self.hero_index[hero_id]] = -1
                        row[-6:] = match_features

                        X_batch[batch_index] = row
                        y_batch.append(1 if result["radiant_win"] else 0)

                        batch_index += 1
                        if batch_index == self.batch_size:
                            yield X_batch[:batch_index], np.array(y_batch)
                            X_batch = np.zeros(
                                (self.batch_size, len(self.hero_ids) + 6),
                                dtype=np.float32,
                            )
                            y_batch = []
                            batch_index = 0
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")

            if batch_index > 0:  # Yield remaining data
                yield X_batch[:batch_index], np.array(y_batch)


# --------- MANUAL LOGISTIC REGRESSION WITH MINI-BATCH ---------
class ManualLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        return -np.mean(
            y_true * np.log(y_pred + 1e-9) + (1 - y_true) *
            np.log(1 - y_pred + 1e-9)
        )

    def train(self, data_loader):
        """Train the model using mini-batch gradient descent."""
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for X_batch, y_batch in data_loader.load_batches():
                if self.weights is None:
                    self.weights = np.zeros(X_batch.shape[1], dtype=np.float32)

                linear_pred = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.sigmoid(linear_pred)

                error = y_pred - y_batch
                dw = np.dot(X_batch.T, error) / len(y_batch)
                db = np.sum(error) / len(y_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def save_model(self):
        """Save model to disk."""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f"{model_save_dir}manual_logreg_{timestamp}.joblib"
        dump({"weights": self.weights, "bias": self.bias}, filename)
        print(f"Model saved to: {filename}")


# --------- MAIN EXECUTION ---------
if __name__ == "__main__":
    # Initialize loader and model
    batch_size = 1000
    epochs = 5
    learning_rate = 0.01

    loader = DotaDataLoader(zip_path, hero_mapping, batch_size=batch_size)
    model = ManualLogisticRegression(
        learning_rate=learning_rate, epochs=epochs)

    # Train the model
    print("Training the model...")
    model.train(loader)

    # Save the trained model
    print("Saving the model...")
    model.save_model()

    # Example usage: Predict win probability for a draft
    radiant_heroes = [7, 9]
    dire_heroes = [8, 4, 129]
    test_features = np.zeros(len(loader.hero_ids) + 6, dtype=np.float32)
    for hero_id in radiant_heroes:
        test_features[loader.hero_index[hero_id]] = 1
    for hero_id in dire_heroes:
        test_features[loader.hero_index[hero_id]] = -1

    test_features[-6:] = [48, 14, 1937, 1983, 4, 22]  # Example match features
    win_prob = model.predict_proba(test_features)
    print(f"Win Probability for Radiant: {win_prob:.2%}")
