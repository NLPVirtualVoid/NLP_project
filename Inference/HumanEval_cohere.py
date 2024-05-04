"""
This module contains code for running inference using the Cohere API 
See *main* for an example
"""

import os
import cohere
import time
import re
from datasets import load_dataset
from Test.human_eval.data import write_jsonl, read_problems


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

class Test:

    def __init__(self,HE_path,API_key):

        if not os.path.exists(HE_path):
            raise ValueError("HumanEval path is not a valid directory")

        #self.data = load_dataset("openai_humaneval",cache_dir=HE_path,ignore_verifications=True)
        self.data = read_problems()

        try:
            self.cohereAPI = cohere.Client(API_key)
        except ValueError as error:
            print(error)
            raise ValueError("Cannot connect to cohere API using key provided")

        # Create new dir for output e.g. .../APPS/Cohere/Train/
        #self.output_path = "/".join(insert_after("APPS",APPS_path.split("/"),"Cohere"))
        #os.makedirs(self.output_path, exist_ok=True)

    def run_problem(self,indices = None):

        if indices is None:
            indices = list(range(len(self.data))) # process all problems
        else:
            indices = [idx for idx in indices if idx < len(self.data)] # constrain indices

        tasks = [task_id for task_id in self.data]
        solutions = []

        for task in tasks:

            inference_start_time = time.time()

            try:
                prompt = "Please write code for a python function that solves the following problem : "
                prompt = prompt + self.data[task]['prompt']

                # Run prompt against Cohere API:
                generated_code = self.cohereAPI.generate(prompt=prompt)

            except RuntimeError as error:
                print(error)
                print("Cohere API call failed ...")
                #self.write_fail_log(output_dir, "Cohere API call failed")
                continue

            code_as_string = str(generated_code.data)
            pattern = r'```[Pp]ython(.*?)```'
            match = re.search(pattern, code_as_string, re.DOTALL)

            if match:
                print("[%s] Python code generated" % (task))
                solutions.append(dict(task_id = task,completion = match.group(1)))
            else:
                print("[%s] No python found in model response" % (task))
                solutions.append(dict(task_id=task, completion=" "))

            inference_end_time = time.time()
            print("Inference for problem [%s] : total elapsed time = %.01f" % (
            task, inference_end_time - inference_start_time))

        try:
            output_dir = "C:/DEV/data/HumanEval/"
            write_jsonl(output_dir + "samples.jsonl", solutions,append=False)
        except RuntimeError as error:
            print(error)




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
    HE_path = "C:/dev/data/HumanEval/"

    API_key = "apikey"

    I = Test(HE_path,API_key)
    idx = [0,1]
    I.run_problem(idx)
