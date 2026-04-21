# filename: read_entropy_change.py

def read_entropy_change(filename='entropy_change_result.txt'):
    with open(filename, 'r') as file:
        content = file.read()
    return content

# Read and print the entropy change result
entropy_change_result = read_entropy_change()

# Save the read result to a new file for clarity
with open('entropy_change_output.txt', 'w') as f:
    f.write(entropy_change_result)

# Also print the result (for debugging or logging purposes)
print(entropy_change_result)  # Note: In non-interactive environment, this will not display but is included for completeness
