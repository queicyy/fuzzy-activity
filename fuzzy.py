import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

df = pd.read_csv('movie_dataset.csv')

df = df[['title', 'popularity', 'vote_average', 'vote_count', 'revenue']].dropna()

df_normalized = df.copy()
cols_to_normalize = ['popularity', 'vote_average', 'vote_count', 'revenue']
df_normalized[cols_to_normalize] = (df[cols_to_normalize] - df[cols_to_normalize].min()) / (df[cols_to_normalize].max() - df[cols_to_normalize].min())

popularity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'popularity')
vote_avg = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'vote_avg')
vote_count = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'vote_count')
revenue = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'revenue')
score = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'score')

for var in [popularity, vote_avg, vote_count, revenue]:
    var.automf(3, names=['low', 'medium', 'high'])

score.automf(5, names=['very_low', 'low', 'medium', 'high', 'very_high'])

rules = [
    ctrl.Rule(popularity['high'] & vote_avg['high'], score['very_high']),
    ctrl.Rule(vote_count['high'] & revenue['high'], score['very_high']),
    ctrl.Rule(popularity['medium'] & vote_avg['medium'], score['medium']),
    ctrl.Rule(popularity['low'] | vote_avg['low'], score['low']),
    ctrl.Rule(revenue['low'] & vote_count['low'], score['very_low'])
]

scoring_system = ctrl.ControlSystem(rules)
scoring = ctrl.ControlSystemSimulation(scoring_system)

scores = []
for idx, row in df_normalized.iterrows():
    scoring.input['popularity'] = row['popularity']
    scoring.input['vote_avg'] = row['vote_average']
    scoring.input['vote_count'] = row['vote_count']
    scoring.input['revenue'] = row['revenue']
    scoring.compute()
    scores.append(scoring.output['score'])

df['fuzzy_score'] = scores

df_ranked = df.sort_values(by='fuzzy_score', ascending=False)

print(df_ranked[['title', 'fuzzy_score']].head(10))