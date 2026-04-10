import math

# Define the string of letters
unique_letters = 'IOWA'

# Calculate the number of unique letters
unique_letters_count = len(set(unique_letters))  # Using set to ensure uniqueness

# Calculate the number of arrangements (n!)
arrangements = math.factorial(unique_letters_count)

# Print the result with context
print(f'The number of four-letter code words possible using the letters {unique_letters} is: {arrangements}')