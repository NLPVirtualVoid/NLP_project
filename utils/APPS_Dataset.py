"""
This module contains utility code for a torch Dataset class that handles APPS problem data
"""

import os
from torch.utils.data import Dataset

class AppsDataset(Dataset):

    def __init__(self, dataset_path, tokenizer = None, max_len = None):

        self.path = dataset_path
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.problems = {}
        with os.scandir(dataset_path) as it:
            for entry in it:
                check_folder = entry.is_dir() and len(entry.name) == 4 and entry.name.isnumeric()

                if check_folder:
                    self.problems[entry.name] = False # Add to problems dictionary - unprocessed

        if tokenizer is not None:
            self.tokenizer = tokenizer

        if max_len is not None:
            self.max_len = max_len

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):

        problem = list(self.problems)[idx]
        print("[%s] Serving ..." % problem)


        problem_path = self.path + problem + "/question.txt"

        try:
            with open(problem_path, "r",encoding='utf-8') as file:
                prompt = file.read()
        except RuntimeError as error:
            print(error)
            print("Failed to read APPS problem")
            return {"ERROR": True }


        preamble = "Provide Python code to solve the following problem : "
        prompt = preamble + prompt

        if self.tokenizer is None:
            if self.max_len is not None:
                prompt = prompt[:self.max_len]

            processed = self.problems[problem]
            self.problems[problem] = True  # processed

            return {
                    'ID': problem,
                    'prompt': prompt,
                    'processed': processed
                    }
        else:
            encoding = self.tokenizer.encode_plus(
                prompt,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                )

            self.problems[problem] = True # processed

            return {
                'text': prompt,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                     }

if __name__ == '__main__':

    # THIS NEEDS TO BE SET TO LOCAL TARGET FOR EACH USER
    APPS_path = "C:/dev/data/APPS/train/"

    APPS = AppsDataset(APPS_path,tokenizer=None,max_len=None)

    for i,prompt in enumerate(APPS):
        if i == 5:
            print(prompt)
            break

