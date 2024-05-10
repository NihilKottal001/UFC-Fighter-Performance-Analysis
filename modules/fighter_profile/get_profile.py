from typing import Any
import pandas as pd

class Profile:
    def __init__(self, filename, df) -> None:
        self.filename = filename
        self.df = df

    def merge_profile(self):
        profile_dataset = pd.read_csv(self.filename)

        columns = list(self.df.columns)
        columns.extend(['fighter_' + col for col in profile_dataset.columns if col != 'fighter'])
        columns.extend(['opponent_' + col for col in profile_dataset.columns if col != 'fighter'])
        combined_df = pd.DataFrame(columns=columns)

        for index, row in self.df.iterrows():
            try:
                values = row.tolist()
                fighter_row = profile_dataset[profile_dataset['fighter'] == row['fighter']]
                fighter_row = fighter_row.drop(columns=['fighter'])
                values.extend(fighter_row.iloc[0].tolist())


                opponent_row = profile_dataset[profile_dataset['fighter'] == row['opponent']]
                opponent_row = opponent_row.drop(columns=['fighter'])
                values.extend(opponent_row.iloc[0].tolist())


                combined_df.loc[len(combined_df)] = values

            except:
                pass
        '''if self.profile_df is not None:
            profile_columns = list(self.profile_df)
            profile_columns.extend(['fighter_' + col for col in profile_dataset.columns if col != 'fighter'])
            profile_columns.extend(['opponent_' + col for col in profile_dataset.columns if col != 'fighter'])
            profile_data = pd.DataFrame(columns=profile_columns)
        for index, row in self.profile_df.iterrows():
            try:
                profile_values = row.tolist()
                fighter_row = profile_dataset[profile_dataset['fighter'] == row['fighter']]
                fighter_row = fighter_row.drop(columns=['fighter'])
                profile_values.extend(fighter_row.iloc[0].tolist())


                opponent_row = profile_dataset[profile_dataset['fighter'] == row['opponent']]
                opponent_row = opponent_row.drop(columns=['fighter'])
                profile_values.extend(opponent_row.iloc[0].tolist())

                profile_data.loc[len(profile_data)] = profile_values
            except:
                pass'''

        return combined_df

