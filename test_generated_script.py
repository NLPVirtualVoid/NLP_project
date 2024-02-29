# Import JSON and os module
import json
import subprocess

# create list of questions
number_strings = []
lower_bound = 3
upper_bound = 13
for i in range(lower_bound,upper_bound):
        number_strings.append(f"{i:04}")


for i in number_strings:
    script_file = "Output/answer"+i+".py"
    in_out_file = "apps-main/APPS/train/"+i+"/input_output.json"

    with open(script_file, "r") as file:
        code=file.read()
    with open(in_out_file, "r") as file:
        in_out=file.read()

    inputs = json.loads(in_out)['inputs'][0]
    command = ["python", "-c", code]

    # Execute the command
    process = subprocess.run(command, input=inputs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="ascii")

    # Check the return code
    if process.returncode == 0:
        #print("Execution successful")
        #print("Output:")
        #print(process.stdout)
        out_str=subprocess.check_output(command, input=inputs, encoding="ascii", stderr=subprocess.DEVNULL)
        if out_str == json.loads(in_out)['outputs'][0]:
             print(f"Problem {i} success")
        else:
             print(f"Problem {i} failed")
    else:
        print(f"Problem {i} execution failed, error: {str(process.stderr)}")


print("Finished")