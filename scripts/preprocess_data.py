import json
import os
import argparse
from tqdm import tqdm
from config import Config

def preprocess_molweni(data_dir, output_dir, config):
    """
    Preprocessing the Molweni dataset.

    Args.
        data_dir: path to the original dataset.
        output_dir: path to the output of the preprocessed dataset.
        config: Configuration object.
    """

    for split in ["train", "dev", "test"]:
        input_file = os.path.join(data_dir, f"{split}.json")
        output_file = os.path.join(output_dir, f"{split}.json")
        processed_data = []

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for dialog_id, dialog_data in tqdm(data.items(), desc=f"Processing Molweni {split}"):
            context = []
            for turn in dialog_data["log"]:
                speaker = turn["speaker"].replace("participant_", "")
                utterance = turn["text"]
                context.append({"speaker": speaker, "utterance": utterance})
            
            for qa in dialog_data["questions"]:
                question = qa["question"]
                
                # Make sure the list of answers exists and there is at least one answer
                if "answers" not in qa or not qa["answers"]:
                    # Handle cases where "answers" key is missing or the list is empty
                    # For example, skip this Q&A pair, or assign a default answer.
                    print(f"Warning: Skipping question in dialog {dialog_id} due to missing or empty answers.")
                    continue

                answer = qa["answers"][0]["text"]  # Take the first answer as the target answer
                processed_data.append({
                    "dialog_id": dialog_id,
                    "context": context,
                    "question": question,
                    "answer": answer
                })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        print(f"Processed Molweni {split} data saved to {output_file}")

def preprocess_friendsqa(data_dir, output_dir, config):
    """
    Preprocessing the FriendsQA dataset.

    Args.
        data_dir: path to the original dataset.
        output_dir: path to the output of the preprocessed dataset.
        config: Configuration object.
    """

    for split in ["train", "dev", "test"]:
        input_file = os.path.join(data_dir, f"{split}.json")
        output_file = os.path.join(output_dir, f"{split}.json")
        processed_data = []

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for dialog_data in tqdm(data, desc=f"Processing FriendsQA {split}"):
            dialog_id = dialog_data["id"]
            
            
            context = []
            for turn in dialog_data["turns"]:
                speaker = turn["speaker"]
                utterance = turn["utterance"]
                context.append({"speaker": speaker, "utterance": utterance})
            
            for qa in dialog_data["questions"]:
                question = qa["question"]
                
            
                if "answers" not in qa or not qa["answers"]:
                    # Handle cases where "answers" key is missing or the list is empty
                    # For example, skip this Q&A pair, or assign a default answer.
                    print(f"Warning: Skipping question in dialog {dialog_id} due to missing or empty answers.")
                    continue
                    
                answer = qa["answer"]
                processed_data.append({
                    "dialog_id": dialog_id,
                    "context": context,
                    "question": question,
                    "answer": answer
                })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        print(f"Processed FriendsQA {split} data saved to {output_file}")

def preprocess_dream(data_dir, output_dir, config):
    """
    Preprocesses the DREAM dataset.

    Args.
        data_dir: path to the original dataset.
        output_dir: path to the output of the preprocessed dataset.
        config: Configuration object.
    """
    for split in ["train", "dev", "test"]:
        input_file = os.path.join(data_dir, f"{split}.json")
        output_file = os.path.join(output_dir, f"{split}.json")
        processed_data = []

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for dialog_data in tqdm(data, desc=f"Processing DREAM {split}"):
            dialog_id = dialog_data["dialog_id"]
            
            
            context = []
            for turn in dialog_data["dialogue"]:
                speaker = turn["speaker"]
                utterance = turn["text"]
                context.append({"speaker": speaker, "utterance": utterance})
            
            for qa in dialog_data["qas"]:
                question = qa["question"]
                
           
                if "answers" not in qa or not qa["answers"]:
                    # Handle cases where "answers" key is missing or the list is empty
                    # For example, skip this Q&A pair, or assign a default answer.
                    print(f"Warning: Skipping question in dialog {dialog_id} due to missing or empty answers.")
                    continue
                    
                answer = qa["answer"]
                processed_data.append({
                    "dialog_id": dialog_id,
                    "context": context,
                    "question": question,
                    "answer": answer
                })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        print(f"Processed DREAM {split} data saved to {output_file}")

def preprocess_dailydialog(data_dir, output_dir, config):
    """
    Preprocessing the DailyDialog dataset.

    Args.
        data_dir: path to the original dataset.
        output_dir: path to the output of the preprocessed dataset.
        config: Configuration object.
    """
    for split in ["train", "dev", "test"]:
        input_file = os.path.join(data_dir, f"{split}.json")
        output_file = os.path.join(output_dir, f"{split}.json")
        processed_data = []

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for dialog_data in tqdm(data, desc=f"Processing DailyDialog {split}"):
            dialog_id = dialog_data["dialog_id"]
            
            
            context = []
            for turn in dialog_data["dialogue"]:
                speaker = turn["speaker"]
                utterance = turn["text"]
                context.append({"speaker": speaker, "utterance": utterance})
            
            for qa in dialog_data["qas"]:
                question = qa["question"]
                
                # Make sure the list of answers exists and there is at least one answer
                if "answers" not in qa or not qa["answers"]:
                    # Handle cases where "answers" key is missing or the list is empty
                    # For example, skip this Q&A pair, or assign a default answer.
                    print(f"Warning: Skipping question in dialog {dialog_id} due to missing or empty answers.")
                    continue
                    
                answer = qa["answer"]
                processed_data.append({
                    "dialog_id": dialog_id,
                    "context": context,
                    "question": question,
                    "answer": answer
                })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        print(f"Processed DailyDialog {split} data saved to {output_file}")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", type=str, required=True,
                        help="Which dataset to preprocess. Choose from: 'molweni', 'friendsqa', 'dream', 'dailydialog', or 'all'.")
    parser.add_argument("--data_dir", default="data", type=str,
                        help="The directory where the raw data is stored.")
    parser.add_argument("--output_dir", default="data", type=str,
                        help="The directory where the processed data will be saved.")
    args = parser.parse_args()
    
    config = Config()

    if args.dataset == "all":
        datasets = config.dataset_choices
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        print(f"Preprocessing {dataset}...")
        
        raw_data_dir = os.path.join(args.data_dir, dataset)
        processed_data_dir = os.path.join(args.output_dir, dataset)
        
        # Make sure the output directory exists
        os.makedirs(processed_data_dir, exist_ok=True)

        if dataset == "molweni":
            preprocess_molweni(raw_data_dir, processed_data_dir, config)
        elif dataset == "friendsqa":
            preprocess_friendsqa(raw_data_dir, processed_data_dir, config)
        elif dataset == "dream":
            preprocess_dream(raw_data_dir, processed_data_dir, config)
        elif dataset == "dailydialog":
            preprocess_dailydialog(raw_data_dir, processed_data_dir, config)
        else:
            print(f"Unknown dataset: {dataset}")

if __name__ == "__main__":
    main()