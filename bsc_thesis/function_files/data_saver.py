import numpy as np
import pickle
import os


def create_filename(part1, part2, part3, part4, part5):
    
    filename = f"{part1}_{part2}_{part3}_{part4}_{part5}"
    
    return filename


def save(data, folder, filename):

    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    
    print(f"Data saved successfully to {file_path}")


def load(folder, filename):

    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    print(f"Data loaded successfully from {file_path}")
    return data