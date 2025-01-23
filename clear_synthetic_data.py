import os

def delete_files_in_subfolders(base_path):
    """
    Deletes all files inside any subfolder under the specified base path.

    Args:
        base_path (str): The base directory containing subfolders with files to delete.
    """
    # Walk through each subdirectory under the base path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # Construct the full file path
            file_path = os.path.join(root, file)
            try:
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

#
base_path = "assets/synthetic_data"
delete_files_in_subfolders(base_path)