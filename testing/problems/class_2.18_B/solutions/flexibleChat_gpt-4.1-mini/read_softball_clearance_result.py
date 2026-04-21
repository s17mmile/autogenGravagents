# filename: read_softball_clearance_result.py

def read_clearance_result(filename='softball_clearance_result.txt'):
    try:
        with open(filename, 'r') as file:
            content = file.read()
        return content.strip()
    except FileNotFoundError:
        return 'Result file not found.'

# Read and print the result
result_text = read_clearance_result()

# Save the result to a new file for verification
with open('softball_clearance_result_read.txt', 'w') as f:
    f.write(result_text + '\n')
