import json
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import dump, load
from datetime import datetime
import os

# Paths
zip_path = "data/dota_games.zip"
heroes_json_path = "data/heroes.json"
model_save_path = "models/herostat.pkl"
os.makedirs("models", exist_ok=True)


# Load hero names
def load_hero_names(path):
    with open(path, "r") as f:
        heroes = json.load(f)
    return {hero["id"]: hero["api_name"] for hero in heroes}


hero_mapping = load_hero_names(heroes_json_path)


# Data Loader
class DotaDataLoader:
    def __init__(self, zip_path, games_to_process=None):
        self.zip_path = zip_path
        self.games_to_process = games_to_process

    def load_games(self):
        X, y = [], []
        hero_stats = {}
        hero_pairs = {}

        with zipfile.ZipFile(self.zip_path, "r") as zip_file:
            json_files = [
                name for name in zip_file.namelist() if name.endswith(".json")
            ]
            if self.games_to_process:
                json_files = json_files[: self.games_to_process]

            for file_name in tqdm(json_files, desc="Loading Games"):
                try:
                    with zip_file.open(file_name, "r") as f:
                        game_data = json.load(f)
                        result = game_data["result"]

                        match_features = {
                            "radiant_score": result["radiant_score"],
                            "dire_score": result["dire_score"],
                            "duration": result["duration"],
                            "tower_status_radiant": result["tower_status_radiant"],
                            "tower_status_dire": result["tower_status_dire"],
                            "lobby_type": result["lobby_type"],
                            "game_mode": result["game_mode"],
                        }

                        # Player-level features: KDA aggregation
                        radiant_kda, dire_kda = 0, 0
                        radiant_heroes, dire_heroes = [], []

                        for player in result["players"]:
                            kda = (player["kills"] + player["assists"]) / max(
                                1, player["deaths"]
                            )
                            hero_id = player["hero_id"]
                            if player["player_slot"] < 128:
                                radiant_kda += kda
                                radiant_heroes.append(hero_id)
                            else:
                                dire_kda += kda
                                dire_heroes.append(hero_id)

                            # Hero stats tracking
                            hero_stats.setdefault(
                                hero_id, {"win_count": 0, "pick_count": 0}
                            )
                            hero_stats[hero_id]["pick_count"] += 1
                            if result["radiant_win"] and player["player_slot"] < 128:
                                hero_stats[hero_id]["win_count"] += 1
                            elif (
                                not result["radiant_win"]
                                and player["player_slot"] >= 128
                            ):
                                hero_stats[hero_id]["win_count"] += 1

                        # Hero synergy tracking
                        for hero in radiant_heroes:
                            hero_pairs.setdefault(hero, {})
                            for paired_hero in radiant_heroes:
                                if hero != paired_hero:
                                    hero_pairs[hero][paired_hero] = (
                                        hero_pairs[hero].get(
                                            paired_hero, 0) + 1
                                    )

                        for hero in dire_heroes:
                            hero_pairs.setdefault(hero, {})
                            for paired_hero in dire_heroes:
                                if hero != paired_hero:
                                    hero_pairs[hero][paired_hero] = (
                                        hero_pairs[hero].get(
                                            paired_hero, 0) + 1
                                    )

                        match_features["radiant_kda"] = radiant_kda
                        match_features["dire_kda"] = dire_kda

                        for hero_id in radiant_heroes:
                            match_features[f"hero_{hero_id}"] = 1
                        for hero_id in dire_heroes:
                            match_features[f"hero_{hero_id}"] = -1

                        X.append(match_features)
                        y.append(1 if result["radiant_win"] else 0)

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        return X, y, hero_stats, hero_pairs


# Manual Feature Scaling
def manual_scaling(X):
    X_np = np.array(X)
    mean = np.mean(X_np, axis=0)
    std = np.std(X_np, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    scaled_X = (X_np - mean) / std
    return scaled_X, mean, std


# Manual Logistic Regression
class ManualLogisticRegression:
    def __init__(self, learning_rate=0.005, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        X, y = np.array(X), np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.01

        for epoch in tqdm(range(self.epochs), desc="Training Model"):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)

            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 100 == 0:
                loss = -np.mean(
                    y * np.log(y_pred + 1e-9) + (1 - y) *
                    np.log(1 - y_pred + 1e-9)
                )
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def save_model(self, feature_columns, mean, std, hero_stats, hero_pairs):
        model_data = {
            "weights": self.weights,
            "bias": self.bias,
            "columns": feature_columns,
            "mean": mean,
            "std": std,
            "hero_stats": hero_stats,
            "hero_pairs": hero_pairs,
        }
        dump(model_data, model_save_path)
        print(f"Model saved to: {model_save_path}")


# HeroStats Inference Class
class HeroStats:
    def __init__(
        self, weights, bias, feature_columns, mean, std, hero_stats, hero_pairs
    ):
        self.weights = weights
        self.bias = bias
        self.feature_columns = feature_columns
        self.mean = mean
        self.std = std
        self.hero_stats = hero_stats
        self.hero_pairs = hero_pairs

    def predict_win_probability(self, radiant_heroes, dire_heroes):
        draft = {f"hero_{hid}": 1 for hid in radiant_heroes}
        draft.update({f"hero_{hid}": -1 for hid in dire_heroes})
        draft_df = pd.DataFrame([draft]).reindex(
            columns=self.feature_columns, fill_value=0
        )
        scaled_draft = (draft_df.values - self.mean) / self.std
        linear_pred = np.dot(scaled_draft, self.weights) + self.bias
        return 1 / (1 + np.exp(-linear_pred))[0]

    def get_winrate(self, hero_id):
        hero_data = self.hero_stats.get(
            hero_id, {"win_count": 0, "pick_count": 0})
        return hero_data["win_count"] / max(1, hero_data["pick_count"])

    def get_pickrate(self, hero_id):
        hero_data = self.hero_stats.get(hero_id, {"pick_count": 0})
        return hero_data["pick_count"]

    def get_best_paired_with_hero(self, hero_id):
        pairs = self.hero_pairs.get(hero_id, {})
        return max(pairs, key=pairs.get, default=None)

    def recommend_next_hero(self, current_heroes, opposing_heroes, is_dire=False):
        """
        Recommend the best hero to pick next based on win probability.
        """
        available_heroes = (
            set(self.hero_stats.keys()) -
            set(current_heroes) - set(opposing_heroes)
        )
        best_hero, max_prob = None, 0

        for hero_id in available_heroes:
            test_heroes = current_heroes + [hero_id]
            win_prob = (
                self.predict_win_probability(test_heroes, opposing_heroes)
                if not is_dire
                else self.predict_win_probability(opposing_heroes, test_heroes)
            )

            if win_prob > max_prob:
                best_hero, max_prob = hero_id, win_prob

        return best_hero

    def recommend_full_draft(self, current_heroes, is_radiant=True):
        """
        Recommend a full team of 5 heroes optimized for win probability and synergy.
        """
        draft = current_heroes[:]
        while len(draft) < 5:
            next_hero = self.recommend_next_hero(
                draft, [], is_dire=not is_radiant)
            draft.append(next_hero)
        return draft

    @staticmethod
    def load(path_to_model):
        model_data = load(path_to_model)
        return HeroStats(
            model_data["weights"],
            model_data["bias"],
            model_data["columns"],
            model_data["mean"],
            model_data["std"],
            model_data["hero_stats"],
            model_data["hero_pairs"],
        )


# --------- MAIN EXECUTION ---------
if __name__ == "__main__":
    loader = DotaDataLoader(zip_path, games_to_process=None)
    X_raw, y, hero_stats, hero_pairs = loader.load_games()
    df_X = pd.DataFrame(X_raw).fillna(0)

    X_scaled, mean, std = manual_scaling(df_X.values)
    model = ManualLogisticRegression(learning_rate=0.005, epochs=1000)
    model.train(X_scaled, y)

    model.save_model(df_X.columns, mean, std, hero_stats, hero_pairs)

    hero_model = HeroStats.load(model_save_path)
    radiant_heroes = [7, 9]
    dire_heroes = [8, 4, 129]
    win_prob = hero_model.predict_win_probability(radiant_heroes, dire_heroes)
    print(f"Win Probability for Radiant: {win_prob:.2%}")
