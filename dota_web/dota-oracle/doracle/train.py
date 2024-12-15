import json
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load  # For saving and loading models
import os
import logging

# Set up logging
logging.basicConfig(
    filename="data_loading_errors.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Paths
zip_path = "data/dota_games.zip"
heroes_json_path = "data/heroes.json"
model_path = "trained_draft_model.joblib"  # Path to save/load the trained model
games_to_process = None  # Set to None for all


class HeroStats:
    def __init__(self, hero_mapping, model, X_columns):
        self.hero_mapping = hero_mapping
        self.model = model
        self.X_columns = X_columns

    def predict_win_probability(self, radiant_heroes, dire_heroes):
        """Predict the win probability for a draft."""
        draft = {"hero_" + str(hero_id): 1 for hero_id in radiant_heroes}
        draft.update({"hero_" + str(hero_id): -1 for hero_id in dire_heroes})
        draft_df = pd.DataFrame([draft]).reindex(
            columns=self.X_columns, fill_value=0)
        probability = self.model.predict_proba(draft_df)[0][1]
        return probability

    def recommend_next_hero(self, radiant_heroes, dire_heroes):
        """Recommend the next best hero for Radiant to maximize win probability."""
        max_prob = 0
        best_hero = None

        available_heroes = (
            set(self.hero_mapping.keys()) -
            set(radiant_heroes) - set(dire_heroes)
        )
        for hero_id in available_heroes:
            test_radiant = radiant_heroes + [hero_id]
            win_prob = self.predict_win_probability(test_radiant, dire_heroes)
            if win_prob > max_prob:
                max_prob = win_prob
                best_hero = hero_id

        return best_hero, max_prob


class DraftAssistant:
    def __init__(self, zip_path, heroes_json_path, model_path, games_to_process=None):
        self.zip_path = zip_path
        self.heroes_json_path = heroes_json_path
        self.model_path = model_path
        self.games_to_process = games_to_process
        self.hero_mapping = self.load_hero_names()
        self.model = None
        self.X_columns = None

    def load_hero_names(self):
        with open(self.heroes_json_path, "r") as f:
            heroes = json.load(f)
        return {hero["id"]: hero["api_name"] for hero in heroes}

    def load_games(self):
        X = []
        y = []

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

                        # Skip games without 'players' field
                        if (
                            "result" not in game_data
                            or "players" not in game_data["result"]
                        ):
                            logging.warning(
                                f"Missing 'players' in {file_name}")
                            continue

                        radiant_heroes = []
                        dire_heroes = []

                        for player in game_data["result"]["players"]:
                            if player["player_slot"] < 128:  # Radiant team
                                radiant_heroes.append(player["hero_id"])
                            else:  # Dire team
                                dire_heroes.append(player["hero_id"])

                        # Create a single feature vector (radiant heroes as +1, dire heroes as -1)
                        match_features = {
                            "hero_" + str(hero_id): 1 for hero_id in radiant_heroes
                        }
                        match_features.update(
                            {"hero_" + str(hero_id): -
                             1 for hero_id in dire_heroes}
                        )
                        X.append(match_features)
                        y.append(1 if game_data["result"]
                                 ["radiant_win"] else 0)

                except (KeyError, json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"Error processing {file_name}: {e}")
                    continue

        return X, y

    def preprocess_data(self, X):
        df = pd.DataFrame(X).fillna(0)
        self.X_columns = df.columns
        return df

    def train_and_save_model(self):
        print("Loading game data...")
        X_raw, y = self.load_games()

        print("Preprocessing data...")
        X = self.preprocess_data(X_raw)

        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

        # Save the model
        dump((model, self.X_columns), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the saved model."""
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            self.model, self.X_columns = load(self.model_path)
            return HeroStats(self.hero_mapping, self.model, self.X_columns)
        else:
            print("No saved model found. Train a model first.")
            return None


# Main execution
if __name__ == "__main__":
    assistant = DraftAssistant(
        zip_path, heroes_json_path, model_path, games_to_process)

    # Check if model exists, otherwise train and save
    if not os.path.exists(model_path):
        assistant.train_and_save_model()

    # Load the trained model
    hero_stats = assistant.load_model()

    if hero_stats:
        # Example draft
        radiant_heroes = [7, 9, 44]  # Partial radiant draft
        dire_heroes = [8, 4, 129]  # Dire team draft

        # Predict win probability
        win_prob = hero_stats.predict_win_probability(
            radiant_heroes, dire_heroes)
        print(f"Win Probability for Radiant: {win_prob:.2f}")

        # Recommend next hero
        next_hero, updated_prob = hero_stats.recommend_next_hero(
            radiant_heroes, dire_heroes
        )
        print(
            f"Recommended Next Hero: {hero_stats.hero_mapping[next_hero]} (Win Probability: {
                updated_prob:.2f})"
        )
