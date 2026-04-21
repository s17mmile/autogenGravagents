# filename: read_vibrational_probability.py

def read_probability(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return 'File not found.'
    except IOError as e:
        return f'Error reading file: {e}'

if __name__ == '__main__':
    filename = 'vibrational_probability.txt'
    result = read_probability(filename)
    print(result)
