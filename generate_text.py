import csv
import wandb
import torch
import torch.nn.functional

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from evaluate import generate_random_email, accuracy, similarity

# Initialize Model
# model = "TheBloke/Llama-2-70B-GGUF"
# model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model = "meta-llama/Llama-2-70b-chat-hf"
# model = "TheBloke/CodeLlama-70B-Instruct-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model)
# quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
base_model = AutoModelForCausalLM.from_pretrained(
    model,
    # model_file="llama-2-70b.Q5_K_M.gguf",
    # quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto", 
) 

# Calculate probabilities for next token, given context
def get_next_word_probs(input_ids, source_tokens=None):
  with torch.no_grad():
    logits = base_model(input_ids).logits.squeeze()[-1]
  probabilities = torch.nn.functional.log_softmax(logits, dim=0)
  return probabilities

def greedy(context):
    probs = get_next_word_probs(context)
    # Get top token according to probability
    _, top_k_tokens = torch.topk(probs, 10)
    new_token = top_k_tokens[0].unsqueeze(0).unsqueeze(0)
    return new_token

# Constrained Beam Search
def beam_search(initial_context, original_text_tokens, k=5, w=5):
    """
    Explaination:
        This function uses beam search to determine the best sequence while also checking ngrams to ensure
        generated text is in original text
    Given: 
        initial_context: Already generated text+prompts+previous unconstrained answer
        k: Parameter that specifys topk probabilities that will be considered
        w: Pruning parameter for beam search (how many nodes we keep for next iteration)
        original_text_tokens: Token list of original text (text only, no prompts etc.)
    
    Returns:
        sequence with best relative cumulative probability
    """
    # initial_context = tokenizer.encode(initial_context, return_tensors='pt')
    new_tokens = []
    sequences = [(initial_context, 0.0, new_tokens, 0.0)]  # List of tuples (sequence, cumulative_probability, new_tokens)
    
    penalization = False
    alpha = 5 # hyperparameter to penalize from Google Neural Machine Translation 
    if penalization:
        wandb.log({"alpha": alpha})
    wandb.log({"length_penalty": penalization})

    all_possible_sequences = []
    # Find tokens 
    for n in range(20):
        # list of token sequences which have been used and got a new token appended
        used_token_sequences = []
        no_new_tokens = True

        # length penalty
        length_penalty = (((5 + n) ** alpha) / (6 ** alpha)) 

        # Loop through top w sequences 
        for (sequence, cumulative_prob, new_tokens, relative_cumulative_prob) in sequences:
            if n == 0:
                permitted = [[elem] for elem in original_text_tokens]
            else:
                # nested list of permitted tokens
                permitted = ngrams(original_text_tokens, new_tokens)

            # Get the probabilities for the next token (can only be tokens from original_tokens by construction)
            probs = get_next_word_probs(sequence, original_text_tokens)
            top_k_probs, top_k_tokens = torch.topk(probs, k)

            # Create new sequences, for each append topk_tokens according to prob
            for i in range(len(top_k_tokens)):
                # threshold
                # prev_token = tokenizer.decode(new_tokens)
                new_token_list = new_tokens + [top_k_tokens[i].item()]
                for l in permitted:
                    # Checking ngrams
                    if new_token_list == l:
                        no_new_tokens = False
                        used_token_sequences.append(new_tokens)

                        new_item_tensor = top_k_tokens[i].unsqueeze(0).unsqueeze(0)
                        new_sequence = torch.cat((sequence, new_item_tensor), dim=1)
                        
                        new_cumulative_prob = cumulative_prob + top_k_probs[i].item()
                        if penalization == True:
                            new_relative_cumulative_prob = new_cumulative_prob / length_penalty
                        else:
                            new_relative_cumulative_prob = new_cumulative_prob / (n+1)
                        all_possible_sequences.append((new_sequence, new_cumulative_prob, new_token_list, new_relative_cumulative_prob))
                        break # do not append it again if its in permitted multiple times
            

        # If no new sequence has been appended, stop and return token sequence of best sequence according to cumulative prob
        if no_new_tokens:
            best_sequence = max(sequences, key=lambda x: x[3])
            return torch.tensor(best_sequence[2]).unsqueeze(0)

        # delete used entries
        if n != 0:
            all_possible_sequences = [entry for entry in all_possible_sequences if entry[2] not in used_token_sequences]

        # Sort all_possible sequences and select the top-w sequences based on the RELATIVE cumulative probability
        all_possible_sequences.sort(key=lambda x: x[3], reverse=True)
        sequences = all_possible_sequences[:w]

    # return best sequence according to RELATIVE cumulative probability
    best_sequence = max(sequences, key=lambda x: x[3])
    # print(f"Returned Sequence: {tokenizer.decode(best_sequence[2])}")
    return torch.tensor(best_sequence[2]).unsqueeze(0)


# Helper functions
# ngrams
def ngrams(original_text_tokens, new_tokens):
    """
    Explaination:
        creates a list of all possible combinations of next token, given current token(s) sequence
    Given: 
        original_text_tokens: List of tokens in the original text (no prompts etc.)
        new_tokens: currently already appended tokens 
    
    Returns:
        sublist: a nested list in the form of [[new_tokens + found token that comes after this new_tokens sequence in original text]],
        this will contain more lists if there are more possibilities.
    """
    sublists = []
    len_token_list = len(new_tokens)
    len_main_list = len(original_text_tokens)
    
    for i in range(len_main_list - len_token_list + 1):
        if original_text_tokens[i:i+len_token_list] == new_tokens and i+len_token_list < len_main_list:
            sublists.append(original_text_tokens[i:i+len_token_list+1])
    return sublists

# find index of a sublist in list, if it doesn't exist return -1
def find_sublist_index(main_list, sublist):
    try:
        start_index = main_list.index(sublist[0])
        while start_index <= len(main_list) - len(sublist):
            if main_list[start_index:start_index + len(sublist)] == sublist:
                return start_index
            start_index = main_list.index(sublist[0], start_index + 1)
    except ValueError:
        return -1  
    return -1 

def count_occurrences(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

# Templates
def test(text):return f"""
   <s>[INST]<<SYS>>
   Task: Cite the Aktenzahl.
   Instruction: Answer only with the Aktenzahl of the following text.

   Example Answer:
   ###/####
   
   Text: ###
   {text}
   ###
   Answer:[/INST] 
   """

def gptprompt(text): return f"""
   <s>[INST]<<SYS>>
   Task: Cite the Aktenzahl.
   Instruction: Carefully read the provided text and identify any instance of Aktenzahl, which are specific file numbers formatted with two groups of numbers separated by a slash ('/'). The length of the numbers on either side of the slash can vary. Answer only with the Aktenzahl found in the text.

   Example Answer:
   ###/####
   
   Text: ###
   {text}
   ###
   Answer:[/INST]
   """

def test2(text): return f"""
   <s>[INST]<<SYS>>
   Task: Cite the Aktenzahl.
   Instruction: Answer only with the File number of the following text.

   Example Answer:
   123/4567
   Text: ###
   {text}
   ###
   Answer:[/INST] 
   """




# beam search testing 
def beam_search_test(text):
    input_text = tokenizer.encode(test(text),return_tensors='pt')
    og_text_tokens = tokenizer.encode(text)
    newline = 13

    # preprocessing original text (no newline)
    og_text_tokens_no_newline = [num for num in og_text_tokens if num != newline]

    output = beam_search(input_text, og_text_tokens_no_newline).flatten().flatten().tolist()
    print("-" * 80)
    print(f"Answer Beam Search: {tokenizer.decode(output)}")
    return tokenizer.decode(output)

# greedy testing
def greedy_test(text):
    context = tokenizer.encode(test(text),return_tensors='pt')
    # Tokenize eos Token
    eos_token = tokenizer.eos_token_id

    generated_tokens = []
    i = 0
    while True:
        if i < 50: # limit token generation 
            new_token = greedy(context)
            # stopping condition
            if eos_token == new_token:
                print("EOS Token found!")
                break
            context = torch.cat((context, new_token), dim=1)  # Concatenate along columns
            generated_tokens.append(new_token.item())
            # print(tokenizer.decode(new_token.item()))
            i += 1
        else:
            break
    print(f"Answer Greedy: {tokenizer.decode(generated_tokens)}")
    return tokenizer.decode(generated_tokens)


# check performance and evaluate performance
wandb.init(
    project="Testing",
    
    config={
    "model": model,
    "prompt": test("text")
    }
)

def check_performance(pair):
  filenumbers = []
  outputs_greedy = []
  outputs_beam = []
  print(len(pair))
  for i, elem in enumerate(pair):
    print(i)
    print(elem[1])
    text = elem[0]
    filenumber = elem[1]
    filenumbers.append(filenumber)
    print(text)
    wandb.config.n = len(pair)
    output_beam = beam_search_test(text)
    output_greedy = greedy_test(text)
    outputs_beam.append(output_beam)
    outputs_greedy.append(output_greedy)
    print(f"Filenumber Annotated: {filenumber}")

  # log metrics to wandb
  acc_greedy, true_g, total_g = accuracy(outputs_greedy, filenumbers)
  acc_beam, true_b, total_b = accuracy(outputs_beam, filenumbers)
  similarity_acc_g, true_similiarity_g, total_similarity_g = similarity(outputs_greedy, filenumbers)
  similarity_acc_b, true_similiarity_b, total_similarity_b = similarity(outputs_beam, filenumbers)
  wandb.log({"greedy_acc": acc_greedy})
  wandb.log({"beam_acc": acc_beam})
  wandb.log({"greedy_similarity": similarity_acc_g})
  wandb.log({"beam_similarity": similarity_acc_b})
  print(f"Greedy:{acc_greedy}, {true_g}/{total_g}")
  print(f"Beam:{acc_beam}, {true_b}/{total_b}")
  print(f"Greedy Similarity Acc:{similarity_acc_g}, {true_similiarity_g}/{total_similarity_g}")
  print(f"Beam Similarity Acc:{similarity_acc_b}, {true_similiarity_b}/{total_similarity_b}")


# Data loading
# load real data
def load_real_data():
    csv_file_name = 'real_data_edited.csv'

    # Initialize an empty list to store the nested list
    nested_list = []

    # Read the CSV file
    with open(csv_file_name, mode='r', encoding='utf-8') as file:
        # Create a CSV reader object with semicolon as delimiter
        csv_reader = csv.reader(file, delimiter=';')
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Append the current row as a list to the nested list
            nested_list.append([row[0], row[1]])
    return nested_list

# load synthetic data
def load_synthetic_data():
    synthetic_list = []
    for i in range(100):
        text, filenumber = generate_random_email()
        synthetic_list.append([text, filenumber])
    return synthetic_list

# Texts
real = True
if real == False:
  pair = load_synthetic_data()
  wandb.config.text = "fake"
else:
  pair = load_real_data()
  wandb.config.text = "real"

check_performance(pair)
