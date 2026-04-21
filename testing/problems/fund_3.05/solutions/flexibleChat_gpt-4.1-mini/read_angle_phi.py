# filename: read_angle_phi.py

def read_angle_phi_from_file(filename='angle_phi_result.txt'):
    """
    Reads the angle phi value from the specified file.

    Parameters:
    filename (str): Path to the file containing the angle result.

    Returns:
    float: The numerical angle phi value.

    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the numerical value cannot be parsed.
    """
    try:
        with open(filename, 'r') as file:
            content = file.read().strip()
        # Extract the numerical value from the string
        import re
        match = re.search(r'[-+]?[0-9]*\.?[0-9]+', content)
        if match:
            return float(match.group())
        else:
            raise ValueError('No numerical value found in the file.')
    except FileNotFoundError:
        raise FileNotFoundError(f'File {filename} not found.')

if __name__ == '__main__':
    try:
        angle_phi = read_angle_phi_from_file()
        print(f'Angle phi between vectors a and b: {angle_phi:.4f} degrees')
    except Exception as e:
        print(f'Error: {e}')
