# filename: read_liftoff_time.py

try:
    with open('liftoff_time.txt', 'r') as f:
        liftoff_time_content = f.read()
except FileNotFoundError:
    liftoff_time_content = 'liftoff_time.txt file not found.'

print(liftoff_time_content)
