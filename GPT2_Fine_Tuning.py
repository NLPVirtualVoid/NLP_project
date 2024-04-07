import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import time

def get_optimal_device():
        if torch.cuda.is_available():
                return "cuda"
        if torch.backends.mps.is_available():
                return "mps"
        return "cpu"

device_name = get_optimal_device() # https://pytorch.org/docs/master/notes/mps.html https://developer.apple.com/metal/pytorch/
device = torch.device(device_name)
print(torch.backends.mps.is_available())



def full_in_out(model, tokenizer, prompt_text="Generate Python code to determine if a number is prime"):
        start_time=time.time()

        if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token


        input_ids=torch.LongTensor(tokenizer.encode(prompt_text,
                           verbose=False,
                           # return_tensors="pt",
                           # max_length=1024,
                           # truncation=True
                          )).unsqueeze(0)
        tokenizer_time=time.time()

        # Create attention mask tensor
        attention_mask = torch.ones_like(input_ids)
        output_ids=model.generate(input_ids,
                          #attention_mask=attention_mask,
                          #pad_token_id=tokenizer.pad_token_id,
                          # parameters to control the length of output
                          max_length = 1024 - len(input_ids),
                          # max_new_tokens = 200,
                          num_beams=2,
                          early_stopping=True,
                          #num_return_sequences=1,
                          eos_token_id=tokenizer.eos_token_id,
                          # Parameters that control the generation strategy used
                          do_sample=False,
                          # Parameters for manipulation of the model output logits
                          #top_k = 2,
                          #temperature=0.1,
                          #repetition_penalty=7.0,
                         )
        output_time=time.time()

        output_str = tokenizer.decode(output_ids[0]) #,skip_special_tokens=True
        decoder_time=time.time()

        print(f"Full I/O:\nTokenizer time: {(tokenizer_time-start_time):.1f}  Output Time: {(output_time-tokenizer_time):.1f}  Decoder Time:{(decoder_time-output_time):.1f}")
        print(f"Total time: {(decoder_time-start_time):.1f}")
        return output_str



def pipeline_in_out(pipeline, prompt_text="Generate Python code to determine if a number is prime"):

        # run inference
        try:
                output_str = pipeline(prompt_text,
                              do_sample=True,
                              num_return_sequences=1,
                              temperature=0.1,
                              #top_k=5,
                              num_beams=5,
                              early_stopping=True,
                              #max_new_tokens=500,
                              truncation=True,
                              max_length=1023,
                              eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=50256,
                              )
        except Exception as err:
                output_str = ["Error", str(err)]


        return output_str



# create list of questions
number_strings = []
lower_bound = 224
upper_bound = 500
for i in range(lower_bound,upper_bound):
        number_strings.append(f"{i:04}")
prompts=[]
print_data = False

#Hendryks model fine tuned on APPS training data
model_path = "apps-main/models/1.5B"
# Setup the transformer model
torch.mps.set_per_process_memory_fraction(0.5) # https://pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction.html#torch.mps.set_per_process_memory_fraction
start_time=time.time()
if print_data: print("Memory used pre-model:",torch.mps.current_allocated_memory())
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_path).to("mps")
print("Memory used post-model:",torch.mps.current_allocated_memory())
end_time=time.time()
print(f"Setup:\nModel & Tokenizer time: {(end_time-start_time):.1f}")

# loop over questions populating prompts list
for i in number_strings:
        # read in question
        question_path = "apps-main/APPS/train/"+i+"/question.txt"
        with open(question_path, "r") as file:
                question_text=file.read()
        # add question to prompts
        prompts.append(question_text)

        if print_data: print("Memory used pre-cache empty:",torch.mps.current_allocated_memory())        
        torch.mps.empty_cache() # https://github.com/pytorch/pytorch/issues/105839
        if print_data: print("Memory used post=cashe empty:",torch.mps.current_allocated_memory())        
        pipeline_start_time=time.time()
        pipeline = transformers.pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        torch_dtype=torch.float32,
                        device="mps",
                        )
        if print_data: print("Memory used post-pipeline:",torch.mps.current_allocated_memory())
        pipeline_end_time=time.time()
        if print_data: print_data: print(f"Pipeline time: {(pipeline_end_time - pipeline_start_time):.1f}")

        # run inference
        inference_start_time=time.time()
        prompt_text = "\nQUESTION:\n"+question_text+"\nANSWER:\n"
        generated_code = pipeline_in_out(pipeline, prompt_text)
        inference_end_time=time.time()

        if generated_code[0] != "Error":
                print(f"Inference {i} output time: {(inference_end_time - inference_start_time):.1f}")
                # split text
                preamble = generated_code[0]['generated_text'].split("ANSWER:")[0]
                code = generated_code[0]['generated_text'].split("ANSWER:")[1]

                f = open("Output/answer"+i+".py", "w")
                f.write(code)
                f.close()
        else:
                print(f"Inference {i} output time: {(inference_end_time - inference_start_time):.1f} ERROR")
                preamble=generated_code[0]+"\n"+generated_code[1]+"\n"+question_text
        g = open("Output/preamble"+i+".txt", "w")
        g.write(preamble)
        g.close() 

print(f"Finished at {time.time()}")

#for code in generated_code:
#    print(code['generated_text'])
# Using the full in/out decoder takes a long time (~30mins) vs only ~15s for pipeline
# generated_code = full_in_out(model, tokenizer, prompt_text)
# print("Full in/out\n",full_in_out(model, tokenizer, prompt_text))
# print("Pipeline\n",generated_code)
#print("Pipeline Test\n", pipeline_in_out(model, tokenizer,"Generate Python code to determine if a number is prime.")[0]['generated_text'])
# prompt_text="Generate Python code to solve the following problem: "+question_text
# exec(generated_code)
