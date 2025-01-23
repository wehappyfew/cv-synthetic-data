import os
from helpers import load_stl_file, generate_random_rotations, overlay_2d_onto_background, split_data

# Paths
stl_files_path = "assets/3dmodels"
backgrounds_path = "assets/background_images"
rendered_2d_images = "./assets/rendered_2d_images"
synthetic_data = "./assets/synthetic_data"

# number of rotated images to generate from the stl file
# these will be overlayed onto the backgrounds
number_of_rotation_images = 10


# Main function
def main(stl_file):
    # Path to the stl files
    stl_files_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/3dmodels/")

    # get the model name from the stl file
    model_name = stl_file.split("/")[-1].split(".")[0]

    # Load the mesh from STL
    mesh = load_stl_file(stl_files_path + stl_file)

    # Generate and save random rotations
    generate_random_rotations(stl_file, mesh, rendered_2d_images, number_of_rotations=number_of_rotation_images)

    # Get files from both folders
    rendered_2d_files = [os.path.join(f'{rendered_2d_images}/{model_name}', file) for file in os.listdir(f'{rendered_2d_images}/{model_name}') if
                      os.path.isfile(os.path.join(f'{rendered_2d_images}/{model_name}', file))]
    background_files = [os.path.join(backgrounds_path, file) for file in os.listdir(backgrounds_path) if
                        os.path.isfile(os.path.join(backgrounds_path, file))]

    # Overlay the 2D images onto the background
    overlay_2d_onto_background(
        stl_file=stl_file,
        background_paths=background_files,
        image_paths=rendered_2d_files,
        synthetic_data_folder=synthetic_data
    )

    # Split the synthetic data into training and val sets
    split_data(synthetic_data_folder=synthetic_data, split_ratio=0.2)


if __name__ == "__main__":
    main(stl_file='lancet.stl')
    # for stl_file in os.listdir(stl_files_path):
    #     if stl_file.endswith(".stl"):  # Optional: Only process .stl files
    #         main(stl_file=stl_file)



