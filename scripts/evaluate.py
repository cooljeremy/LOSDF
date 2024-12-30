import os
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer
from tqdm import tqdm
from models.losdf import LOSDF
from config import Config
from scripts.train import evaluate, load_and_cache_examples  # 复用 train.py 中的函数
import argparse
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the trained model.")
    parser.add_argument("--dataset_name", default="molweni", type=str,
                        help="The name of the dataset to evaluate on.")
    parser.add_argument("--split", default="test", type=str,
                        help="The split to evaluate on: 'dev' or 'test'.")
    parser.add_argument("--output_file", default=None, type=str,
                        help="File to store predictions.")
    args = parser.parse_args()

    config = Config()
   
    device = torch.device(config.device)

    if config.model_name == "losdf":
      tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    elif config.model_name == "electra_baseline":
      tokenizer = AutoTokenizer.from_pretrained(config.electra_model, do_lower_case=config.do_lower_case)
    elif config.model_name == "bert_baseline":
      tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    
    dataset = load_and_cache_examples(config, tokenizer, args.dataset_name, args.split)

    if config.model_name == "losdf":
        model = LOSDF(config).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin")))
    else:
        model = LOSDF(config).to(device)
        model.qa_outputs.load_state_dict(AutoModelForQuestionAnswering.from_pretrained(args.model_path).state_dict())

    results = evaluate(config, model, dataset, device)

    logger.info("***** Eval results *****")
    for key, value in results.items():
        logger.info("  %s = %s", key, value)


    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()