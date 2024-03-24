# Testing on HumanEval dataset

Assuming you have generated a well-formatted ``json`` file with solutions for HumanEval problems you can run the tests
with ``HumanEval.py`` by editing the name of the samples json to your specific solution file.

You will need a copy of the HumanEval dataset present in this directory : ``HumanEval.jsonl.gz``

## Generating solutions using Cohere

To run inference using ``Cohere`` on ``HumanEval`` instantiate a ``Test`` object using the code in ``HumanEval_cohere.py`` 
but ensure that you have the HumanEval dataset present in the ``output`` directory before proceeding. The method ``run_problem``
allows the user to run any problem (or list of problems) in the dataset based on indices.   

## Generating solutions with other LLms

As long as you write your solutions to a ``.jsonl`` file with the right format (see HumanEval github readme) then you
can test it using the code in this directory.
