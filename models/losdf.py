import torch
import torch.nn as nn
from models.components import RewritingModule, MultiPartyAttention, InformationDecouplingModule
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from config import Config

class LOSDF(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

         # Initialize the components
        self.rewriting_module = RewritingModule(config)
        self.multi_party_attention = MultiPartyAttention(config)
        self.information_decoupling_module = InformationDecouplingModule(config)
        
        # Use a separate encoder to encode the question (also consider sharing with MultiPartyAttention)
        if config.model_name in ["losdf", "electra_baseline"]:
            self.question_encoder = AutoModel.from_pretrained(self.config.electra_model)
        elif config.model_name == "bert_baseline":
            self.question_encoder = AutoModel.from_pretrained(self.config.bert_model)
        
        # Fine-tuning the pre-trained model used
        if config.model_name == "losdf":
          self.qa_outputs = nn.Linear(self.question_encoder.config.hidden_size, 2)
        elif config.model_name == "electra_baseline":
          self.qa_outputs = AutoModelForQuestionAnswering.from_pretrained(config.electra_model)
        elif config.model_name == "bert_baseline":
          self.qa_outputs = AutoModelForQuestionAnswering.from_pretrained(config.bert_model)

    def forward(self, input_ids, attention_mask, speaker_ids, question_input_ids, question_attention_mask, token_type_ids=None, start_positions=None, end_positions=None):
        """
        Forward propagation process for LOSDF models.

        Args.
            input_ids: input token ID, shape: (batch_size, seq_len)
            attention_mask: attention mask, shape: (batch_size, seq_len)
            speaker_ids: speaker ID, shape: (batch_size, )
            question_input_ids: question token ID, shape: (batch_size, seq_len)
            question_attention_mask: Attention mask for the question, shape: (batch_size, seq_len)
            token_type_ids: optional, to distinguish sentences when the input is a sentence pair, shape: (batch_size, seq_len)
            start_positions: start positions of answers, shape: (batch_size, )
            end_positions: end positions of the answer, shape: (batch_size, )
        Returns.
            Returns the loss if start_positions and end_positions are provided;
            Otherwise, return the predicted start and end logits.
        """
        
        # 1. Semantic rewriting
        # Due to the fact that the input to MultiPartyAttention is encoded, it is necessary here to rewrite it according to the semantics of
        # Each utterance in the dialog is rewritten.
        batch_size = input_ids.shape[0]
        rewritten_dialogs = []
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model)
        for i in range(batch_size):
          dialog_utterances = []
          for j in range(len(input_ids[i])):
            if input_ids[i][j].item() == tokenizer.sep_token_id:
                break
            utterance = tokenizer.decode(input_ids[i][j], skip_special_tokens=True)
            dialog_utterances.append({"speaker":"", "utterance": utterance}) # 这里speaker可以不填
          
          rewritten_dialog = self.rewriting_module(dialog_utterances)
          rewritten_dialogs.append(rewritten_dialog)

        
        # 2. Use Multi-Party Attention to get contextual representation
        # Recoding using rewritten text
        
        rewritten_input_ids = []
        rewritten_attention_mask = []
        for i, rewritten_dialog in enumerate(rewritten_dialogs):
          encoded_inputs = tokenizer(
              rewritten_dialog,
              padding="max_length",
              truncation=True,
              max_length=self.config.max_seq_length,
              return_tensors="pt",
          )
          rewritten_input_ids.append(encoded_inputs["input_ids"])
          rewritten_attention_mask.append(encoded_inputs["attention_mask"])
        
        rewritten_input_ids = torch.cat(rewritten_input_ids, dim=0)
        rewritten_attention_mask = torch.cat(rewritten_attention_mask, dim=0)

        
        context_vector = self.multi_party_attention(rewritten_input_ids, rewritten_attention_mask, speaker_ids)
        
        # 3. Use another encoder to encode the problem and get the problem representation
        if self.config.model_name in ["losdf", "electra_baseline"]:
          question_outputs = self.question_encoder(question_input_ids, attention_mask=question_attention_mask)
        elif self.config.model_name == "bert_baseline":
          question_outputs = self.question_encoder(question_input_ids, attention_mask=question_attention_mask, token_type_ids=question_token_type_ids)
        
        question_vector = question_outputs.last_hidden_state[:, 0, :]

        # 4. decoupled information
        logits = self.information_decoupling_module(context_vector, question_vector)

        # 5. Answer prediction
        start_logits, end_logits = self.qa_outputs(torch.cat((context_vector, question_vector), dim=1)).split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Calculate loss 
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits