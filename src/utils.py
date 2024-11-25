import tarfile
from pathlib import Path
import os
import shutil
import logging
import zipfile
import requests
import tempfile
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm


def extract_dataset(source: str, extract_path: str) -> None:
    """
    Extract a dataset file (tar.gz or zip) to a specified directory, showing progress,
    and removing macOS hidden files. The source can be a local file path or a URL.

    Args:
    source (str): Path to the local file or URL of the dataset file.
    extract_path (str): Path where the dataset should be extracted.

    Raises:
    FileNotFoundError: If the local file is not found.
    requests.exceptions.RequestException: If there's an error downloading the file.
    tarfile.ReadError: If there's an error reading the tar.gz file.
    zipfile.BadZipFile: If there's an error reading the zip file.
    """
    extract_path = Path(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)

    if source.startswith('http://') or source.startswith('https://'):
        print(f"Downloading file from {source}")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            response = requests.get(source, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as pbar:
                for data in response.iter_content(block_size):
                    size = temp_file.write(data)
                    pbar.update(size)
            temp_file_path = temp_file.name
    else:
        temp_file_path = source
        if not Path(temp_file_path).exists():
            raise FileNotFoundError(f"The file {temp_file_path} does not exist.")

    try:
        try:
            with tarfile.open(temp_file_path, 'r:gz') as tar:
                members = tar.getmembers()
                total_members = len(members)
                with tqdm(total=total_members, unit='file', desc="Extracting files") as pbar:
                    for member in members:
                        tar.extract(member, path=extract_path)
                        pbar.update(1)
        except tarfile.ReadError:
            with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                members = zip_ref.infolist()
                total_members = len(members)
                with tqdm(total=total_members, unit='file', desc="Extracting files") as pbar:
                    for member in members:
                        zip_ref.extract(member, path=extract_path)
                        pbar.update(1)

        print(f"Dataset extracted to {extract_path}")
        print(f"Total files extracted: {total_members}")
        print("Removing macOS hidden files...")
        removed_files = 0
        for root, dirs, files in os.walk(extract_path, topdown=False):
            for name in files:
                if name.startswith('._') or name == '.DS_Store':
                    os.remove(os.path.join(root, name))
                    removed_files += 1
            for name in dirs:
                if name == '__MACOSX':
                    shutil.rmtree(os.path.join(root, name))
                    removed_files += 1
        print(f"Removed {removed_files} macOS hidden files/folders")
        print(f"Final number of files: {total_members - removed_files}")

    except (tarfile.ReadError, zipfile.BadZipFile) as e:
        raise type(e)(f"Error reading the file: {temp_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise
    finally:
        if source.startswith('http://') or source.startswith('https://'):
            os.unlink(temp_file_path)


def get_paths_to_files(dir_path: str) -> Tuple[List[str], List[str]]:
    """
    Recursively get file paths and file names in 'train', 'test', and 'val' subdirectories,
    excluding hidden files and files outside these directories.

    Args:
    dir_path (str): The path to the directory to search.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing two lists:
    - The first list contains the full file paths.
    - The second list contains the file names.

    Raises:
    FileNotFoundError: If the specified directory does not exist.
    ValueError: If any of 'train', 'test', or 'val' subdirectories are missing.
    """
    filepaths = []
    fnames = []
    required_dirs = ['train', 'test', 'valid']

    try:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"The directory {dir_path} does not exist.")

        missing_dirs = [d for d in required_dirs if not os.path.isdir(os.path.join(dir_path, d))]
        if missing_dirs:
            raise ValueError(f"The following required directories are missing: {', '.join(missing_dirs)}")

        for subdir in required_dirs:
            subdir_path = os.path.join(dir_path, subdir)
            for dirpath, _, filenames in os.walk(subdir_path):
                for f in filenames:
                    if not f.startswith('.'):  
                        full_path = os.path.join(dirpath, f)
                        filepaths.append(full_path)
                        fnames.append(f)

        if not filepaths:
            print(f"Warning: No files were found in the subdirectories of {dir_path}")

        return filepaths, fnames
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], []

def get_dataset_paths(dataset_dir: str) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Get file paths and names for train, test, and validation sets.

    Args:
    dataset_dir (str): The path to the main dataset directory containing 'train', 'test', and 'val' subdirectories.

    Returns:
    Dict[str, Tuple[List[str], List[str]]]: A dictionary with keys 'train', 'test', and 'val'.
    Each value is a tuple containing two lists:
    - The first list contains the full file paths.
    - The second list contains the file names.

    Raises:
    FileNotFoundError: If the dataset directory or any required subdirectory does not exist.
    PermissionError: If there are insufficient permissions to access the directories.
    ValueError: If no files are found in a subdirectory.
    """
    dataset_paths = {}
    required_splits = ['train', 'test', 'valid']

    try:
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        for split in required_splits:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                raise FileNotFoundError(f"Required subdirectory not found: {split_dir}")

            filepaths = []
            fnames = []

            for dirpath, dirnames, filenames in os.walk(split_dir):
                for f in filenames:
                    if not f.startswith('.'):  
                        filepaths.append(os.path.join(dirpath, f))
                        fnames.append(f)

            if not filepaths:
                raise ValueError(f"No files found in {split_dir}")

            dataset_paths[split] = (filepaths, fnames)

        return dataset_paths

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise


def get_logger(ch_log_level: int = logging.INFO, fh_log_level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger with console and file handlers.

    Args:
        ch_log_level (int): Logging level for the console handler. Default is logging.INFO.
        fh_log_level (int): Logging level for the file handler. Default is logging.INFO.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(ch_log_level)
    
    # File Handler
    fh = logging.FileHandler('training.log')
    fh.setLevel(fh_log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger


def calculate_mean_std(data):
    return np.mean(data), np.std(data)