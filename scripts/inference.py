import torch
from transformers import AutoTokenizer
from models.losdf import LOSDF
from config import Config
import argparse
import logging

# Setup log
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def inference(config, model, tokenizer, context, question, speaker_ids, device):
    """
    Q&A using the LOSDF model.

    Args.
        config: Configuration object.
        model: LOSDF model.
        tokenizer: The tokenizer.
        context: Conversation context, a list of strings.
        question: question, a string.
        speaker_ids: Speaker ids for each utterance, a list.
        device: device (cpu or cuda).

    returns: a string representing the predicted answer.
        A string representing the predicted answer.
    """
    # Coding of inputs
    
    
    speaker_separator = " [SEP] "
    context_str = ""
    
    for i, turn in enumerate(context):
        context_str += turn["speaker"] + ": " + turn["utterance"] + speaker_separator
    
    encoded_inputs = tokenizer(
        context_str,
        question,
        truncation=True,
        max_length=config.max_seq_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    
    
    
    token_type_ids = []
    for i in range(len(encoded_inputs["input_ids"][0])):
        if encoded_inputs["sequence_ids"][0][i] == 0:
            token_type_ids.append(0)
        else:
            token_type_ids.append(1)
    
    
    # If bert_baseline, need to enter token_type_ids
    if config.model_name == "bert_baseline":
      inputs = {
          "input_ids": encoded_inputs["input_ids"].to(device),
          "attention_mask": encoded_inputs["attention_mask"].to(device),
          "token_type_ids": torch.tensor(token_type_ids).unsqueeze(0).to(device),
          "speaker_ids": torch.tensor(speaker_ids).unsqueeze(0).to(device),
          "question_input_ids": encoded_inputs["input_ids"].to(device),
          "question_attention_mask": encoded_inputs["attention_mask"].to(device),
      }
    else:
      inputs = {
          "input_ids": encoded_inputs["input_ids"].to(device),
          "attention_mask": encoded_inputs["attention_mask"].to(device),
          "speaker_ids": torch.tensor(speaker_ids).unsqueeze(0).to(device),
          "question_input_ids": encoded_inputs["input_ids"].to(device),
          "question_attention_mask": encoded_inputs["attention_mask"].to(device),
      }
    
    # Reasoning
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs[1]
    end_logits = outputs[2]

    # Get answers to predictions
    start_index = torch.argmax(start_logits).item()
    end_index = torch.argmax(end_logits).item()
    
    
    if start_index >= len(encoded_inputs["input_ids"][0]):
      start_index = 0
    if end_index >= len(encoded_inputs["input_ids"][0]):
      end_index = 0

    answer_tokens = encoded_inputs["input_ids"][0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the trained model.")
    parser.add_argument("--context_file", default=None, type=str,
                        help="File containing the dialog context.")
    parser.add_argument("--question", default=None, type=str,
                        help="The question to ask.")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode.")
    args = parser.parse_args()

    config = Config()


    device = torch.device(config.device)

 
    if config.model_name == "losdf":
      tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    elif config.model_name == "electra_baseline":
      tokenizer = AutoTokenizer.from_pretrained(config.electra_model, do_lower_case=config.do_lower_case)
    elif config.model_name == "bert_baseline":
      tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    
    # Loading Models
    if config.model_name == "losdf":
        model = LOSDF(config).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin")))
    else:
        model = LOSDF(config).to(device)
        model.qa_outputs.load_state_dict(AutoModelForQuestionAnswering.from_pretrained(args.model_path).state_dict())

    model.eval()

    if args.interactive:
        # interactive mode
        print("Entering interactive mode. Type 'exit' to quit.")
        while True:
            context_str = input("Enter the dialog context (or 'load' to load from file): ")
            if context_str.lower() == "exit":
                break
            
            if context_str.lower() == "load":
                context_file = input("Enter the context file path: ")
                try:
                    with open(context_file, "r") as f:
                        context = json.load(f)
                    
               
                    if not isinstance(context, list) or not all(isinstance(turn, dict) for turn in context):
                        print("Error: Invalid context format. Context should be a list of dictionaries.")
                        continue
                    
                 
                    if not all("speaker" in turn and "utterance" in turn for turn in context):
                        print("Error: Invalid context format. Each turn should have 'speaker' and 'utterance' keys.")
                        continue
                    
                   
                    speaker_ids = [idx % 2 for idx, turn in enumerate(context) for _ in range(len(tokenizer.encode(turn["utterance"], add_special_tokens=False)) + 2)]

                except FileNotFoundError:
                    print(f"Error: File not found: {context_file}")
                    continue
            else:
                context = [{"speaker": "User", "utterance": context_str}] # 临时方案, 因为是交互式, 所以不知道说话人
                speaker_ids = [0] * (len(tokenizer.encode(context_str, add_special_tokens=False)) + 2) # 2 for the speaker and the separator

            question = input("Enter your question: ")
            if question.lower() == "exit":
                break
            
            answer = inference(config, model, tokenizer, context, question, speaker_ids, device)
            print(f"Answer: {answer}")

    else:
        # non-interactive mode
        if args.context_file is None or args.question is None:
            print("Error: Please provide both --context_file and --question in non-interactive mode.")
            return

        try:
            with open(args.context_file, "r") as f:
                context = json.load(f)
                
                # Ensure that context is a list of dict
                if not isinstance(context, list) or not all(isinstance(turn, dict) for turn in context):
                    print("Error: Invalid context format. Context should be a list of dictionaries.")
                    return

                # Make sure that each dictionary contains the keys “speaker” and “utterance”.
                if not all("speaker" in turn and "utterance" in turn for turn in context):
                    print("Error: Invalid context format. Each turn should have 'speaker' and 'utterance' keys.")
                    return
                
                # 提取speaker_ids
                speaker_ids = [idx % 2 for idx, turn in enumerate(context) for _ in range(len(tokenizer.encode(turn["utterance"], add_special_tokens=False)) + 2)]
        except FileNotFoundError:
            print(f"Error: File not found: {args.context_file}")
            return

        answer = inference(config, model, tokenizer, context, args.question, speaker_ids, device)
        print(f"Context: {context}")
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()