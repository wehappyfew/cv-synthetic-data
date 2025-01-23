import os
import cv2
import random
import shutil
import trimesh
import numpy as np
import open3d as o3d
from io import BytesIO
from PIL import Image, UnidentifiedImageError

def generate_random_rotations(stl_file, mesh, output_folder, number_of_rotations):
    for _ in range(number_of_rotations):
        # Save the rendered image with random rotation
        save_rendered_image(stl_file, mesh, output_folder, apply_camo='grayscale')


def apply_rotation(mesh):
    # Generate random angles for X, Y, Z between 0 and 360 degrees
    angle_x = random.uniform(0, 360)
    angle_y = random.uniform(0, 360)
    angle_z = random.uniform(0, 360)

    # Rotation matrices for each axis
    rotation_x = trimesh.transformations.rotation_matrix(np.radians(angle_x), [1, 0, 0])
    rotation_y = trimesh.transformations.rotation_matrix(np.radians(angle_y), [0, 1, 0])
    rotation_z = trimesh.transformations.rotation_matrix(np.radians(angle_z), [0, 0, 1])

    # Apply the rotations
    mesh.apply_transform(rotation_x)
    mesh.apply_transform(rotation_y)
    mesh.apply_transform(rotation_z)

    return mesh, angle_x, angle_y, angle_z


def apply_stripe_pattern(mesh, num_stripes=10):
    """
    Apply a random stripe pattern to the mesh vertices.
    :param mesh:
    :param num_stripes:
    :return:

    """
    # Apply a striped pattern using vertex colors
    vertices = np.array(mesh.vertices)
    vertex_colors = np.zeros((len(vertices), 3))

    # Generate stripes by alternating colors along the X axis
    for i, vertex in enumerate(vertices):
        # Stripe pattern based on the x-coordinate of the vertex
        stripe_color = 1 if int((vertex[0] * num_stripes) % 2) == 0 else 0
        vertex_colors[i] = [stripe_color, stripe_color, stripe_color]  # Grey stripes

    return vertex_colors


def apply_grayscale(mesh):
    """
    Apply a random shade of grey to the mesh vertices.
    :param mesh:
    :return:
    """
    # Generate a random shade of grey (between 0.2 to 0.8 for a visible range)
    grey_value = random.uniform(0.2, 0.8)
    grey_color = [grey_value, grey_value, grey_value]  # RGB values for grey

    # Apply grey color to the mesh vertices
    vertex_colors = o3d.utility.Vector3dVector([grey_color] * len(mesh.vertices))

    return vertex_colors


def save_rendered_image(stl_file, mesh, output_folder, apply_camo='grayscale'):
    # get the model name from the stl file
    model_name = stl_file.split("/")[-1].split(".")[0]

    # Create the output folder if it doesn't exist
    output_folder = f'{output_folder}/{model_name}'

    # Apply random rotations to the mesh
    mesh_rotated, angle_x, angle_y, angle_z = apply_rotation(mesh.copy())

    # Convert trimesh to Open3D geometry
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_rotated.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_rotated.faces)

    # Apply a random pattern on top of the rendered 2d image
    if apply_camo == "grayscale":
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(apply_grayscale(mesh_rotated))
    elif apply_camo == "stripes":
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(apply_stripe_pattern(mesh_rotated))

    # Create the Open3D visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=200, height=200)
    vis.add_geometry(o3d_mesh)

    # Set background to white for consistent removal
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background

    # Render and capture the image
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(True)
    image = np.asarray(image)

    # Convert the image to RGBA and make white background transparent
    img = Image.fromarray((image * 255).astype(np.uint8)).convert("RGBA")
    data = img.getdata()

    new_data = []
    for item in data:
        # If pixel is white (255, 255, 255), make it transparent
        if item[:3] == (255, 255, 255):
            new_data.append((255, 255, 255, 0))  # Transparent pixel
        else:
            new_data.append(item)

    img.putdata(new_data)

    # Convert the image to numpy array for cropping
    img_np = np.array(img)

    # Get the bounding box of the non-transparent pixels (not white)
    non_transparent_pixels = np.where(img_np[:, :, 3] > 0)  # alpha channel > 0
    if len(non_transparent_pixels[0]) == 0:
        print("No visible mesh to crop")
        return

    # Find the min and max x and y indices
    min_x = min(non_transparent_pixels[1])
    max_x = max(non_transparent_pixels[1])
    min_y = min(non_transparent_pixels[0])
    max_y = max(non_transparent_pixels[0])

    # Crop the image using the bounding box
    cropped_img = img_np[min_y:max_y, min_x:max_x]

    # Convert back to PIL Image
    cropped_img_pil = Image.fromarray(cropped_img)

    # Save the processed image
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, f"x{angle_x:.2f}_y{angle_y:.2f}_z{angle_z:.2f}.png")

    cropped_img_pil.save(output_path)
    print(f"Saved overlay image with transparency to {output_path}")

    # Cleanup to avoid memory issues
    vis.destroy_window()


def load_stl_file(stl_file_path):
    mesh = trimesh.load_mesh(stl_file_path)
    return mesh


def split_data(synthetic_data_folder, split_ratio=0.2):
    """
    Split the dataset into train and val sets, moving and managing files accordingly.

    Args:
        synthetic_data_folder (str): Path to the folder containing the images and labels.
        split_ratio (float): The ratio of images to move to the validation set (default is 0.2).
    """
    # Define paths
    images_folder = os.path.join(synthetic_data_folder, 'images')
    labels_folder = os.path.join(synthetic_data_folder, 'labels')

    # Get all image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))]

    # Randomly shuffle and select 20% for validation
    val_count = int(len(image_files) * split_ratio)
    val_images = random.sample(image_files, val_count)

    # Split images and labels
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, os.path.splitext(image_file)[0] + '.txt')

        if image_file in val_images:
            # Move the image to validation
            shutil.move(image_path, os.path.join(images_folder, 'val', image_file))

            # Move the corresponding label to validation
            if os.path.exists(label_path):
                shutil.move(label_path, os.path.join(labels_folder, 'val', os.path.basename(label_path)))
        else:
            # Move the image to train
            shutil.move(image_path, os.path.join(images_folder, 'train', image_file))

            # Move the corresponding label to train
            if os.path.exists(label_path):
                shutil.move(label_path, os.path.join(labels_folder, 'train', os.path.basename(label_path)))

    print("Dataset split completed.")


def add_color_jitter(img):
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))  # Random brightness

    return img


def overlay_2d_onto_background(stl_file, background_paths, image_paths, synthetic_data_folder):
    """
    Overlay multiple rendered_image images on multiple background images.

    Parameters:

    Returns:
    - None: Saves the resulting images to the specified output folder.
    """

    # get the model name from the stl file
    model_name = stl_file.split("/")[-1].split(".")[0]

    # Set the class id based on the model name
    class_id_mapping = {"lancet": 0, "shahed136": 1}
    class_id = class_id_mapping.get(model_name, -1)
    if class_id == -1:
        raise ValueError(f"Unknown model name: {model_name}")

    # Create the YOLO-compatible output folders if they do not exist
    folders = ["images/train", "images/val", "labels/train", "labels/val"]
    for folder in folders:
        os.makedirs(f"{synthetic_data_folder}/{folder}", exist_ok=True)

    # Iterate over all combinations of backgrounds and 2d images
    for i, background_path in enumerate(background_paths):
        for j, rendered_image_path in enumerate(image_paths):
            # Open the background image
            try:
                background = Image.open(background_path)
                # Ensure the image has an alpha channel (RGBA mode)
                if background.mode != "RGBA":
                    background = background.convert("RGBA")
            except UnidentifiedImageError:
                print(f"Skipping invalid image file: {background_path}")
                continue

            try:
                # Open the rendered_image image
                rendered_image = Image.open(rendered_image_path)
                # Ensure the rendered_image image has an alpha channel (RGBA mode)
                if rendered_image.mode != "RGBA":
                    rendered_image = rendered_image.convert("RGBA")
            except UnidentifiedImageError:
                print(f"Skipping invalid image file: {rendered_image_path}")
                continue

            # Generate random position within the bounds of the background
            max_x = background.width - rendered_image.width
            max_y = background.height - rendered_image.height
            random_x = random.randint(0, max_x)
            random_y = random.randint(0, max_y)
            # Define the position for the rendered_image with random x and y
            position = (random_x, random_y)

            # Create an overlay image the same size as the background
            overlay = Image.new("RGBA", background.size, (0, 0, 0, 0))
            # get alpha channel mask
            alpha_channel_mask = rendered_image.split()[3]
            # Paste the rendered image onto the overlay at the specified position
            overlay.paste(rendered_image, position, alpha_channel_mask)  # Use the alpha channel as mask
            # Composite the overlay onto the background
            synthetic_image = Image.alpha_composite(background, overlay)

            # Set the maximum resolution
            max_resolution = (800, 800)  # Example: Max width and height of 800px
            # Resize the final image proportionally to fit within max_resolution
            synthetic_image.thumbnail(max_resolution, Image.LANCZOS)

            # Remove the ICC profile by converting the image to a clean RGBA mode
            synthetic_image = synthetic_image.convert("RGBA")

            # Generate a unique filename for the output image
            background_name = os.path.basename(background_path).rsplit(".", 1)[0]
            rendered_image_name = os.path.basename(rendered_image_path).rsplit(".", 1)[0]
            synthetic_image_name = f"{model_name}_{background_name}_{rendered_image_name}"
            synthetic_image_path = os.path.join(f'{synthetic_data_folder}/images', f'{synthetic_image_name}.png')

            # Calculate YOLO annotation values
            x_center = random_x + rendered_image.width / 2
            y_center = random_y + rendered_image.height / 2
            width = rendered_image.width
            height = rendered_image.height

            # Normalize coordinates
            x_center_normalized = x_center / background.width
            y_center_normalized = y_center / background.height
            width_normalized = width / background.width
            height_normalized = height / background.height

            # Write YOLO annotation to file
            annotation_file = f"{synthetic_data_folder}/labels/{synthetic_image_name}.txt"
            annotation = f"{class_id} {x_center_normalized:.6f} {y_center_normalized:.6f} {width_normalized:.6f} {height_normalized:.6f}\n"
            with open(annotation_file, "w") as f:
                f.write(annotation)

            # # Draw the bounding box to be sure
            # draw = ImageDraw.Draw(synthetic_image)
            # x_min = random_x
            # y_min = random_y
            # x_max = random_x + rendered_image.width
            # y_max = random_y + rendered_image.height
            # draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

            # Save the resulting image
            buffer = BytesIO()
            synthetic_image.save(buffer, format="PNG", icc_profile=None)
            buffer.seek(0)

            # Later, write the buffer content to disk
            with open(synthetic_image_path, 'wb') as f:
                f.write(buffer.getvalue())

