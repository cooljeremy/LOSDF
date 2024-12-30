import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from models.losdf import LOSDF
from models.utils import calculate_em_f1
from config import Config
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import logging

# Setup Log
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train(config, model, train_dataset, dev_dataset, device):
    """
    Training the LOSDF model.

    Args.
        config: Configuration object.
        model: LOSDF model.
        train_dataset: Training dataset.
        dev_dataset: Validation dataset.
        device: Training device (cpu or cuda).
    """
    
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)

    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total
    )

    # 训练
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", config.batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        config.batch_size
        * config.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = 0
    best_dev_f1 = 0.0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(config.num_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2] if config.model_name in ["bert_baseline", "losdf"] else None,
                "speaker_ids": batch[3],
                "question_input_ids": batch[4],
                "question_attention_mask": batch[5],
                "start_positions": batch[6],
                "end_positions": batch[7],
            }

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            
            tr_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                # Print training information
                tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / config.logging_steps, global_step)
                
                if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                    # Evaluate the model every config.logging_steps step
                    results = evaluate(config, model, dev_dataset, device)
                    
                    # Print the current result
                    logger.info("***** Eval results *****")
                    for key, value in results.items():
                        logger.info("  %s = %s", key, value)
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    
                    
                    # Save the best current model
                    if results["f1"] > best_dev_f1:
                        best_dev_f1 = results["f1"]
                        if config.save_model:
                            output_dir = os.path.join(config.save_path, "best_model")
                            os.makedirs(output_dir, exist_ok=True)
                            logger.info("Saving best model to %s", output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            
                            if config.model_name in ["losdf"]:
                              torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                            else:
                              model_to_save.save_pretrained(output_dir)

                            torch.save(config, os.path.join(output_dir, "training_config.bin"))
                            
    
    tb_writer.close()
    
    return global_step, tr_loss / global_step
    

def evaluate(config, model, eval_dataset, device):
    """
    Evaluation of LOSDF models.

    Args.
        config: Configuration object.
        model: LOSDF model.
        eval_dataset: Evaluation dataset.
        device: Training device (cpu or cuda).

    returns: A dictionary containing the evaluation metrics (EM).
        A dictionary containing the evaluation metrics (EM, F1).
    """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", config.batch_size)

    all_results = []
    start_time = time.time()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2] if config.model_name in ["bert_baseline", "losdf"] else None,
                "speaker_ids": batch[3],
                "question_input_ids": batch[4],
                "question_attention_mask": batch[5],
            }

            outputs = model(**inputs)

        start_logits = outputs[1]
        end_logits = outputs[2]
        
        # Convert logits to predictions
        start_preds = torch.argmax(start_logits, dim=1).cpu().numpy().tolist()
        end_preds = torch.argmax(end_logits, dim=1).cpu().numpy().tolist()

        # Get the true start and end positions
        start_positions = batch[6].cpu().numpy().tolist()
        end_positions = batch[7].cpu().numpy().tolist()
        
        #Here, only the first 20 tokens were counted as possible answers.
        for i in range(len(start_preds)):
          if start_preds[i] >= 20:
            start_preds[i] = 0
          if end_preds[i] >= 20:
            end_preds[i] = 0
            

        # Calculate EM and F1 scores 
        for i in range(len(start_preds)):
          
            prediction = (start_preds[i], end_preds[i])
            ground_truth = (start_positions[i], end_positions[i])
            all_results.append({"prediction": prediction, "ground_truth": ground_truth})

    # Calculation of overall assessment indicators
    predictions = [result["prediction"] for result in all_results]
    ground_truths = [result["ground_truth"] for result in all_results]
    
    preds = []
    labels = []
    
    for (s_pred, e_pred), (s_label, e_label) in zip(predictions, ground_truths):
        pred = list(range(s_pred, e_pred + 1))
        label = list(range(s_label, e_label + 1))
        preds.append(pred)
        labels.append(label)
    
    em_f1 = calculate_em_f1(preds, labels)
    results = {
        "em": em_f1["em"],
        "f1": em_f1["f1"],
    }

    return results

def load_and_cache_examples(config, tokenizer, dataset_name, split):
    """
    Load the dataset and cache it.
    """
    # Load data features from cache or dataset file
    
    
    # Reading data using huggingface's datasets library
    data_dir = os.path.join(config.data_dir, dataset_name)
    
    dataset = load_dataset('json', data_files={"train": [os.path.join(data_dir, "train.json")],
                                              "dev": [os.path.join(data_dir, "dev.json")],
                                              "test": [os.path.join(data_dir, "test.json")]})
    
    def prepare_features(examples):
        
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        
        speaker_separator = " [SEP] "
        
        contexts = []
        token_type_ids_list = []
        speaker_ids_list = []
        
        for dialog in examples["context"]:
            context = ""
            speaker_id_per_dialog = []
            # Add speaker information to each utterance
            for i, turn in enumerate(dialog):
                
                context += turn["speaker"] + ": " + turn["utterance"] + speaker_separator
                speaker_id_per_dialog.extend([i%2] * (len(tokenizer.encode(turn["utterance"], add_special_tokens=False)) + 2)) # 2 for the speaker and the separator
            
            contexts.append(context)
            speaker_ids_list.append(speaker_id_per_dialog)
        
        tokenized_examples = tokenizer(
            contexts,
            examples["question"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["speaker_ids"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            
            answer = examples["answer"][sample_index]
            
            # Start/end character index of the answer in the text.
            start_char = examples["context"][sample_index][answer[0]]["utterance"].find(answer[1])
            end_char = start_char + len(answer[1])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 0:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 0:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
            
            # Add the corresponding speaker_id to each feature
            tokenized_examples["speaker_ids"].append(speaker_ids_list[sample_index])
            
            
            token_type_ids = []
            for i in range(len(tokenized_examples["input_ids"][0])):
              if tokenized_examples["sequence_ids"][0][i] == 0:
                token_type_ids.append(0)
              else:
                token_type_ids.append(1)
            
            tokenized_examples["token_type_ids"] = token_type_ids
            
        
        return tokenized_examples

    
    if split == "train":
        features = dataset["train"].map(
            prepare_features,
            batched=True,
            num_proc=4,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )
    elif split == "dev":
        features = dataset["dev"].map(
            prepare_features,
            batched=True,
            num_proc=4,
            remove_columns=dataset["dev"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dev dataset",
        )
    elif split == "test":
        features = dataset["test"].map(
            prepare_features,
            batched=True,
            num_proc=4,
            remove_columns=dataset["test"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on test dataset",
        )
        
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    all_speaker_ids = torch.tensor([f["speaker_ids"] for f in features], dtype=torch.long)
    all_question_input_ids = torch.tensor([f["question_input_ids"] for f in features], dtype=torch.long)
    all_question_attention_mask = torch.tensor([f["question_attention_mask"] for f in features], dtype=torch.long)
    
    if split != "test":
      all_start_positions = torch.tensor([f["start_positions"] for f in features], dtype=torch.long)
      all_end_positions = torch.tensor([f["end_positions"] for f in features], dtype=torch.long)
      
      dataset = TensorDataset(
          all_input_ids,
          all_attention_mask,
          all_token_type_ids,
          all_speaker_ids,
          all_question_input_ids,
          all_question_attention_mask,
          all_start_positions,
          all_end_positions,
      )
    else:
      dataset = TensorDataset(
          all_input_ids,
          all_attention_mask,
          all_token_type_ids,
          all_speaker_ids,
          all_question_input_ids,
          all_question_attention_mask,
      )

    return dataset

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="molweni", type=str, help="The name of the dataset to train on.")
    parser.add_argument("--model_type", default="losdf", type=str, help="Model type: losdf, bert, electra")
    args = parser.parse_args()
    
    # Getting configuration parameters
    config = Config()
    
    # Setting up the device
    device = torch.device(config.device)

    #  tokenizer
    if config.model_name == "losdf":
      tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    elif config.model_name == "electra_baseline":
      tokenizer = AutoTokenizer.from_pretrained(config.electra_model, do_lower_case=config.do_lower_case)
    elif config.model_name == "bert_baseline":
      tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    # Load Dataset
    train_dataset = load_and_cache_examples(config, tokenizer, args.dataset_name, "train")
    dev_dataset = load_and_cache_examples(config, tokenizer, args.dataset_name, "dev")

    # Loading Models
    model = LOSDF(config).to(device)

    # training model
    train(config, model, train_dataset, dev_dataset, device)

if __name__ == "__main__":
    main()