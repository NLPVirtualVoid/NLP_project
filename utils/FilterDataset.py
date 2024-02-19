"""
This script can filter the APPS dataset and copy-delete large examples to separate folders.
This is useful as the top 100 test examples are nearly 90% of dataset by size.
"""

import os
import numpy as np
import shutil

# THIS NEEDS TO BE SET TO LOCAL TARGET FOR EACH USER
APPS_path = "C:/dev/data/APPS/train/"

def get_dir_size(path='.'):
    """
    This function returns the size of a directory or a file in bytes
    :param path: this is the path for the directory/file of interest
    :return: dir size in bytes
    """
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def get_dir_size_ARRAY(path='.'):
    """
    This function returns a numpy array with the size of all directories
    at the path location in MB
    :param path: this is the path for the directory of interest
    :return: np array with directory sizes
    """

    dsize = {}
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                fs_MB = get_dir_size(entry.path)/np.power(2,20)
                dsize[entry.name] = fs_MB

    return dsize

def remove_examples(APPS_path,train_or_test = 'test',max_folder_size_MB = 10):

    APPS_path = APPS_path.strip()
    if train_or_test == 'test':
        APPS_path_examples = APPS_path + "/test/"
    elif train_or_test == 'train':
        APPS_path_examples = APPS_path + "/train/"
    else:
        raise TypeError("train_or_test must be 'test' or 'train'")

    try:
        dir_sizes = get_dir_size_ARRAY(APPS_path_examples)
    except RuntimeError as error:
        print("Error calling get_dir_size_ARRAY: ",error)

    dir_sizes = dict(sorted(dir_sizes.items(), key=lambda item: item[1], reverse=True))

    assert not len(dir_sizes)==0, "There are no directories in path [%s]" % APPS_path_examples

    assert '0001' in dir_sizes.keys(), "Expecting to find example 0001 in APPS dataset"



    for i,key in enumerate(dir_sizes):
        if dir_sizes[key] > max_folder_size_MB:

            # Check whether the new data folder exists or not
            new_folder_path = APPS_path + "/" + train_or_test + "_large_examples"
            FolderExist = os.path.exists(new_folder_path)
            if not FolderExist:
                # Create a new directory
                os.makedirs(new_folder_path)
                print("New directory created: ",new_folder_path)

            print("[%d] Removing example [%s] with size %d MB from %s dataset" % (i,key,dir_sizes[key],train_or_test))

            source_directory = APPS_path_examples + key
            destination_directory = new_folder_path + "/" + key

            # Copy the source_directory to the destination directory
            try:
                shutil.copytree(source_directory, destination_directory)
            except RuntimeError as error:
                print("Error copying dir ",key," :",error)
            # delete source directory once copied
            try:
                shutil.rmtree(source_directory)
            except RuntimeError as error:
                print("Error deleting dir ",key," :",error)



def main():
    APPS_path = input("Please provide the relative or full path to your copy of the APPS dataset - e.g. 'C:/dev/APPS'?")
    remove_examples(APPS_path,'test', max_folder_size_MB=10)
    remove_examples(APPS_path,'train', max_folder_size_MB=10)


if __name__ == '__main__':
    main()