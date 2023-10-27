import shlex, subprocess
import time

commands = ["/home/username/miniconda3/envs/env_surrogate3/bin/python 2d_beam_rolling_load_4x2_multistiffness.py"]

def run_command(command, iterations, delay=10):
    for _ in range(iterations):
        args = shlex.split(command)
        subprocess.run(args)
        time.sleep(delay)

if __name__ == "__main__":
    for command in commands:
        print(command)

    user_input = None
    while user_input not in ['y', 'n']:
        user_input = input('Run experiment with above commands? (y/n): ')
        user_input = user_input.lower()[:1]

    if user_input == 'n':
        exit(0)
    
    iterations = int(input('Enter the number of iterations: '))

    for command in commands:
        run_command(command, iterations)