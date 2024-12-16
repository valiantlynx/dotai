
import zipfile
from tqdm import tqdm


def display_zip_structure(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            print(f"Structure of '{zip_path}':")
            for file_info in tqdm(zip_file.infolist()):
                # Extract file name and size
                file_name = file_info.filename
                file_size = file_info.file_size
                compressed_size = file_info.compress_size

                # Indent structure based on folder levels
                indent_level = file_name.count('/')
                indent = "  " * indent_level
                print(
                    f"{indent}- {file_name} (Size: {file_size} bytes, Compressed: {compressed_size} bytes)")

    except FileNotFoundError:
        print(f"Error: File '{zip_path}' not found.")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid ZIP file.")


# Path to your zip file
zip_path = "dota_games.zip"
display_zip_structure(zip_path)
