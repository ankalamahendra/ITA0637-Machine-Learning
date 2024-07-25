import numpy as np

# Define the training data
data = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
])

# Initialize the hypothesis
hypothesis = ['0'] * (data.shape[1] - 1)  # Start with the most specific hypothesis

# Implement Find-S algorithm
for example in data:
    if example[-1] == 'Yes':  # For positive examples
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':  # If attribute value is '0', take the attribute value from the example
                hypothesis[i] = example[i]
            elif hypothesis[i] != example[i]:  # If attribute value contradicts, generalize to '?'
                hypothesis[i] = '?'

# Print the most specific hypothesis
print("The most specific hypothesis is:", hypothesis)
