import pandas as pd

# Define the dataset
data = {
    'Example': [1, 2, 3, 4],
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Air Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'Enjoy Sport': ['Yes', 'Yes', 'No', 'Yes']
}

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)

# Extract features and labels
features = df.columns[1:-1]
target = df.columns[-1]

# Initialize the most specific hypothesis (S) and the most general hypothesis (G)
S = ['0'] * len(features)
G = [['?'] * len(features)]

# Function to update the S and G sets
def candidate_elimination(examples):
    global S, G
    for i, example in examples.iterrows():
        if example[target] == 'Yes':
            # Update S
            for j in range(len(S)):
                if S[j] == '0':
                    S[j] = example[features[j]]
                elif S[j] != example[features[j]]:
                    S[j] = '?'
            
            # Remove hypotheses from G that are inconsistent with the example
            G = [g for g in G if all((g[k] == '?' or g[k] == example[features[k]]) for k in range(len(g)))]
        
        else:
            # Add general hypotheses to G
            G_new = []
            for g in G:
                for j in range(len(g)):
                    if g[j] == '?':
                        hypothesis = g[:]
                        hypothesis[j] = '0' if S[j] == '?' else S[j]
                        G_new.append(hypothesis)
            G.extend(G_new)

            # Remove hypotheses from G that are too specific
            G = [g for g in G if not any((s != '?' and (s != g[k] and g[k] != '?')) for k, s in enumerate(S))]
    
    return S, G

# Run the Candidate-Elimination algorithm
S, G = candidate_elimination(df)

print("Most specific hypothesis S:")
print(S)

print("\nMost general hypotheses G:")
for g in G:
    print(g)
