import pandas as pd
import numpy as np

# Sample training data as a list of dictionaries
data = [
    {'Origin': 'Japan', 'Manufacturer': 'Honda', 'Color': 'Blue', 'Decade': '1980', 'Type': 'Economy', 'Example Type': 'Positive'},
    {'Origin': 'Japan', 'Manufacturer': 'Toyota', 'Color': 'Green', 'Decade': '1970', 'Type': 'Sports', 'Example Type': 'Negative'},
    {'Origin': 'Japan', 'Manufacturer': 'Toyota', 'Color': 'Blue', 'Decade': '1990', 'Type': 'Economy', 'Example Type': 'Positive'},
    {'Origin': 'USA', 'Manufacturer': 'Chrysler', 'Color': 'Red', 'Decade': '1980', 'Type': 'Economy', 'Example Type': 'Negative'},
    {'Origin': 'Japan', 'Manufacturer': 'Honda', 'Color': 'White', 'Decade': '1980', 'Type': 'Economy', 'Example Type': 'Positive'}
]

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Extract attributes and target
attributes = df.columns[:-1]  # All columns except the last one
target = df.columns[-1]  # The last column

# Initialize the specific and general hypotheses
S = ['0'] * len(attributes)
G = [['?'] * len(attributes)]

# Candidate Elimination algorithm
for i, row in df.iterrows():
    if row[target] == 'Positive':
        for j in range(len(attributes)):
            if S[j] == '0':
                S[j] = row[j]
            elif S[j] != row[j]:
                S[j] = '?'
        G = [g for g in G if all((g[j] == '?' or g[j] == row[j]) for j in range(len(attributes)))]
    else:
        new_G = []
        for g in G:
            for j in range(len(attributes)):
                if g[j] == '?':
                    for value in df[attributes[j]].unique():
                        if value != row[j]:
                            new_g = g.copy()
                            new_g[j] = value
                            if all((S[k] == '?' or new_g[k] == '?' or S[k] == new_g[k]) for k in range(len(attributes))):
                                new_G.append(new_g)
        G = new_G

# Output the final S and G
print("Final specific hypothesis (S):", S)
print("Final general hypotheses (G):", G)
