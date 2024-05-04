"""
This module contains code for running inference using the Cohere API on APPS datset problems.
See *main* for an example.
"""

import os
import utils.APPS_Dataset
import cohere
import time
import re

def insert_after(item, lst, new_item):
    """
    Insert new_item into lst after the first occurrence of item.

    Parameters:
    item: The item after which to insert new_item.
    lst: The list to modify.
    new_item: The item to insert.

    Returns:
    The modified list. If item is not in lst, returns the original list.
    """
    # Find the index of the first occurrence of item
    try:
        index = lst.index(item)
    except ValueError:
        print(f"{item} not found in list.")
        return lst

    # Insert new_item after the first occurrence of item
    lst.insert(index + 1, new_item)

    return lst
class cohereAPI:

    def __init__(self,APPS_path,API_key):

        if not os.path.exists(APPS_path):
            raise ValueError("APPS_path is not a valid directory")

        self.data = utils.APPS_Dataset.AppsDataset(APPS_path,tokenizer=None,max_len=None)

        try:
            self.cohereAPI = cohere.Client(API_key)
        except ValueError as error:
            print(error)
            raise ValueError("Cannot connect to cohere API using key provided")

        # Create new dir for output e.g. .../APPS/Cohere/Train/
        self.output_path = "/".join(insert_after("APPS",APPS_path.split("/"),"Cohere"))
        os.makedirs(self.output_path, exist_ok=True)

    def generate(self,indices = None):

        if indices is None:
            indices = list(range(len(self.data))) # process all problems
        else:
            indices = [idx for idx in indices if idx < len(self.data)] # constrain indices

        for idx in indices:

            problem = self.data[idx]

            if 'ERROR' in problem.keys():
                print("ERROR in generating prompt ...")
                continue

            output_dir = self.output_path + problem['ID']

            inference_start_time = time.time()

            try:
                os.makedirs(output_dir,exist_ok=True)
            except RuntimeError as error:
                print(error)
                print("Failed to makedir : [%s]" % output_dir)
                continue

            try:
                generated_code = self.cohereAPI.generate(prompt=problem['prompt'])
            except RuntimeError as error:
                print(error)
                print("Cohere API call failed ...")
                self.write_fail_log(output_dir,"Cohere API call failed")
                continue

            code_as_string = str(generated_code.data)
            pattern = r'```Python(.*?)```'
            match = re.search(pattern,code_as_string, re.DOTALL)

            # If the pattern was found, print it

            if match:
                output_file = output_dir + "/solution.py"
                try:
                    with open(output_file,'w',encoding='utf-8') as f:
                        f.write(match.group(1))
                except RuntimeError as error:
                    print(error)
                    print("Failed to write solution.py ...")
            else:
                print("No code found in the solution")
                self.write_fail_log(output_dir, "No code found in the API response")

            inference_end_time = time.time()
            print("Inference for problem [%s] : total elapsed time = %.01f" % (problem['ID'],inference_end_time - inference_start_time))

    def write_fail_log(self,output_dir,reason):
        output_file = output_dir + "/FAILED.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(reason)
        except RuntimeError as error:
            print(error)
            print("write_fail_log failed ...")







if __name__ == '__main__':

    # THIS NEEDS TO BE SET TO LOCAL TARGET FOR EACH USER
    APPS_path = "C:/dev/data/APPS/test/"

    API_key = "apikey"

    I = cohereAPI(APPS_path,API_key)
    idx = list(range(1,10))
    I.generate(idx)
