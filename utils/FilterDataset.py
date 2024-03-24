"""
This script can filter the APPS dataset and copy-delete large examples to separate folders.
This is useful as the top 100 test examples are nearly 90% of dataset by size.
"""

import os
import numpy as np
import shutil
import json
import random
from collections import Counter

random.seed(40)

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

def count_words(s):
    T = s.replace('=',' ')
    T = T.replace('\n',' ')
    T = T.replace(':',' ')
    T = T.replace('(',' ')
    T = T.replace(')',' ')
    T = T.replace('[',' ')
    T = T.replace(']',' ')
    T = T.replace('{', ' ')
    T = T.replace('}', ' ')
    T = T.replace('.',' ')
    T = T.replace(',', ' ')
    return len(T.split())


def problem_catalogue(APPS_path,test=True):

    APPS_path = APPS_path.strip()
    if test:
        APPS_path_examples = APPS_path + "/test/"
    else:
        APPS_path_examples = APPS_path + "/train/"


    try:
        dir_sizes = get_dir_size_ARRAY(APPS_path_examples)
    except RuntimeError as error:
        print("Error calling get_dir_size_ARRAY: ", error)

    #dir_sizes = dict(sorted(dir_sizes.items(), key=lambda item: item[1], reverse=True))

    assert not len(dir_sizes) == 0, "There are no directories in path [%s]" % APPS_path_examples

    assert '0001' in dir_sizes.keys(), "Expecting to find example 0001 in APPS dataset"


    cat = []

    for i,key in enumerate(dir_sizes):

        if not (len(key) == 4) and (key.isnumeric()): #Test folder name is consistent
            continue

        record = {'probID': key, 'size': dir_sizes[key]}

        try:
            question_path = APPS_path_examples + key + '/question.txt'
            if not os.path.isfile(question_path):
                record['question'] = {'linecount': 0,
                                      'wordcount': 0,
                                      'test_count': 0,
                                      'alpha_ratio': 0,
                                      'missing_image': False
                                      }
            else:
                with open(question_path, 'r', encoding='utf-8') as f:
                    question = f.readlines()
                    question = ''.join(question)
                    record['question'] = {'linecount': question.count('\n'),
                                          'wordcount': count_words(''.join(question)),
                                          'test_count': question.split('Examples')[-1].count('Input'),
                                          'alpha_ratio': len([char for char in question if char.isalpha()])/len(question.replace(' ','')),
                                          'missing_image': question.lower().count('[image]') > 0
                                       }
        except RuntimeError as error:
            print("Error reading question.txt for problem [%s] : Error: %s " % (key, error))

        try:
            solution_path = APPS_path_examples + key + '/solutions.json'
            if not os.path.isfile(solution_path):
                record['solutions'] = {'count': 0,
                                      'linecount': 0,
                                      'word_count': 0,
                                      'input_count': 0
                                      }
            else:
                with open(solution_path, 'r', encoding='utf-8') as f:
                    solutions = json.load(f)
                    record['solutions'] = {'count': len(solutions),
                                           'linecount': [s.count('\n') for s in solutions],
                                           'wordcount': [count_words(s) for s in solutions],
                                           'input_count': [s.count('input(')  for s in solutions]
                                           }
        except RuntimeError as error:
            print("Error reading solutions.json for problem [%s] : Error: %s " % (key,error))

        try:
            metadata_path = APPS_path_examples + key + '/metadata.json'
            if not os.path.isfile(metadata_path):
                record['metdata'] = {'difficulty': 'unknown',
                                      'url': 'unknown'}
            else:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    record['metadata'] = metadata
        except RuntimeError as error:
            print("Error reading metadata.json for problem [%s] : Error: %s " % (key, error))

        try:
            input_output_path = APPS_path_examples + key + '/input_output.json'
            if not os.path.isfile(input_output_path):
                record['testcases'] = {'count': 0}
            else:
                with open(input_output_path, 'r', encoding='utf-8') as f:
                    testcases = json.load(f)
                    record['testcases'] = {'count': len(testcases['outputs'])}

        except RuntimeError as error:
            print("Error reading input_output.json for problem [%s] : Error: %s " % (key,error))

        cat.append(record)

    try:
        catalogue_path = APPS_path_examples + 'problem_data.json'
        with open(catalogue_path, 'w') as f:
        # Write the list to the file as JSON
            json.dump(cat, f)
    except RuntimeError as error:
        print("Error writing output.json: %s " % (error))

    return cat

def create_test_set(APPS_path,size,dir_name = None):
    fpath = APPS_path + "/test/" + "problem_data.json"
    with open(fpath, 'r', encoding='utf-8') as f:
        cat = json.load(f) # read problem catalogue

    # Shuffle questions:
    random.shuffle(cat)

    testset = []

    for candidate in cat:

        if not candidate['testcases']['count'] > 10: # Median 10 (2467 > 10)
            continue

        if not candidate['testcases']['count'] < 50:
            continue

        if not candidate['solutions']['count'] > 10: # Median 18 (2849 > 10)
            continue

        if max(candidate['solutions']['wordcount']) > 500:
            continue

        if max(candidate['solutions']['linecount']) > 50:
            continue

        if not max(candidate['solutions']['input_count']) > 0:
            continue

        if not candidate['question']['alpha_ratio'] > 0.75:
            continue

        if candidate['question']['missing_image']:
            continue

        if not candidate['question']['test_count'] > 0:
            continue

        if candidate['question']['wordcount'] > 500:
            continue

        if candidate['question']['linecount'] > 60:
            continue

        testset.append(candidate)

        if len(testset) == size:
            break

    difficulty = [c['metadata']['difficulty'] for c in testset]
    print(len(testset))
    print(Counter(difficulty))

    if not dir_name is None:

        APPS_problems = APPS_path + "/test/"

        new_folder_path = APPS_path + "/" + dir_name + "/"
        FolderExist = os.path.exists(new_folder_path)
        if not FolderExist:
            # Create a new directory
            os.makedirs(new_folder_path)
            print("New directory created: ", new_folder_path)


        # Create a dataset:
        for i,q in enumerate(testset):

            # Check whether the new data folder exists or not

            probID = q['probID']
            print("Writing example [%s] to new dataset" % (probID))

            source_directory = APPS_problems + probID
            destination_directory = new_folder_path + ("/%.04d" % i)

            # Copy the source_directory to the destination directory
            try:
                pass#shutil.copytree(source_directory, destination_directory)
            except RuntimeError as error:
                print("Error copying dir ",probID, " :", error)

        try:
            catalogue_path = new_folder_path + 'problem_data.json'
            with open(catalogue_path, 'w') as f:
            # Write the list to the file as JSON
                json.dump(testset, f)
        except RuntimeError as error:
            print("Error writing output.json: %s " % (error))

    return testset


def main():
    APPS_path = input("Please provide the relative or full path to your copy of the APPS dataset - e.g. 'C:/dev/APPS'?")
    #APPS_path = 'C:/DEV/data/APPS/'

    remove_examples(APPS_path,'test', max_folder_size_MB=10)
    remove_examples(APPS_path,'train', max_folder_size_MB=10)
    problem_catalogue(APPS_path,test=True)
    create_test_set(APPS_path,300,"testset_300")

    

if __name__ == '__main__':
    main()