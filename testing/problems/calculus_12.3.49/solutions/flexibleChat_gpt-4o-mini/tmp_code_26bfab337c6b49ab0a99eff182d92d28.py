def calculate_work(F, initial_position, final_position):
    # Check if the input vectors are of length 3
    if len(F) != 3 or len(initial_position) != 3 or len(final_position) != 3:
        raise ValueError('All input vectors must be of length 3.')

    # Calculate the displacement vector d
    displacement = [final_position[i] - initial_position[i] for i in range(3)]

    # Calculate the dot product F . d
    work_done = sum(F[i] * displacement[i] for i in range(3))
    return work_done

# Define the force vector F
FORCE_VECTOR = [8, -6, 9]  # in newtons

# Define the initial and final positions
INITIAL_POSITION = [0, 10, 8]
FINAL_POSITION = [6, 12, 20]

# Calculate the work done
work = calculate_work(FORCE_VECTOR, INITIAL_POSITION, FINAL_POSITION)

# Output the work done
print(f'Work done by the force: {work} joules')