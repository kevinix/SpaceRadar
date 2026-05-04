import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MAX_CONTENT_LENGTH = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {'ply', 'pcd', 'xyz', 'xyzn', 'xyzrgb', 'pts'}

PREPROCESS_DEFAULTS = {
    'voxel_size': 0.02,
    'nb_neighbors': 20,
    'std_ratio': 2.0,
}

RECONSTRUCTION_DEFAULTS = {
    'poisson_depth': 9,
    'alpha_radius': 0.5,
    'marching_cubes_resolution': 128,
}
