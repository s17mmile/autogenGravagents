# filename: read_boiling_point.py

def read_boiling_point(filename='boiling_point_everest.txt'):
    with open(filename, 'r') as file:
        content = file.read().strip()
    return content

if __name__ == '__main__':
    boiling_point_info = read_boiling_point()
    print(boiling_point_info)
