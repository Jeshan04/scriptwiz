from transformers import TFT5ForConditionalGeneration, RobertaTokenizer
from datasets import load_dataset
import random

def run_predict(args, text):
    # load saved finetuned model
    model = TFT5ForConditionalGeneration.from_pretrained(args.save_dir)
    # load saved tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.save_dir) 
    
    # encode texts by prepending the task for input sequence and appending the test sequence
    query = args.prefix + text 
    encoded_text = tokenizer(query, return_tensors='tf', padding='max_length', truncation=True, max_length=args.max_input_length)
    
    # inference
    generated_code = model.generate(
        encoded_text["input_ids"], attention_mask=encoded_text["attention_mask"], 
        max_length=args.max_target_length, top_p=0.95, top_k=50, num_return_sequences=1
    )
    
    # decode generated tokens
    decoded_code = tokenizer.decode(generated_code.numpy()[0], skip_special_tokens=True)
    return decoded_code

def predict_from_dataset(args):
    # load using hf datasets
    dataset = load_dataset('json', data_files='https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl') 
    # train test split
    dataset = dataset['train'].train_test_split(0.1, shuffle=False) 
    test_dataset = dataset['test']
    
    # randomly select an index from the validation dataset
    index = random.randint(0, len(test_dataset))
    text = test_dataset[index]['text']
    code = test_dataset[index]['code']
    
    # run-predict on text
    decoded_code = run_predict(args, text)
    
    print("#" * 25); print("QUERY: ", text); 
    print()
    print('#' * 25); print("ORIGINAL: "); print("\n", code);
    print()
    print('#' * 25); print("GENERATED: "); print("\n", decoded_code);
    
def predict_from_text(args, text):
    # run-predict on text
    decoded_code = run_predict(args, text)
    print("#" * 25); print("QUERY: ", text); 
    print()
    print('#' * 25); print("GENERATED: "); print("\n", decoded_code);
    return decoded_code

# set the path to the saved model directory
model_dir = './'

# set the arguments
class Args:
    save_dir = model_dir
    prefix = ""
    max_input_length = 512
    max_target_length = 64

# call the functions
def foo(STR):
    predict_from_dataset(Args)
    prediction_foo = predict_from_text(Args, STR)
    return prediction_foo