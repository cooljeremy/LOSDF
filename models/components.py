import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering
from config import Config
import openai

class RewritingModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # If you are calling the API, you don't need to load the model.
        if self.config.llm_model == "gpt-3.5-turbo":
          self.tokenizer = None
          self.llm = None
        else:
          self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
          self.llm = AutoModel.from_pretrained(self.config.llm_model)

    def forward(self, dialog_history):
        """
        Use LLM to rewrite every utterance in the dialog history.

        Args.
            dialog_history: a list of dictionaries, each representing an utterance.
                            contains the keys “speaker” and “utterance”.

        Returns.
            A list of rewritten utterances.
        """
        rewritten_history = []
        for turn in dialog_history:
            speaker = turn["speaker"]
            utterance = turn["utterance"]

            # Rewriting is done in different ways depending on the LLM selected
            if self.config.llm_model == "gpt-3.5-turbo":
              rewritten_utterance = self.rewrite_with_api(utterance)
            else:
              prompt = f"Please rewrite the following utterance to make it more formal and remove colloquial expressions, while preserving the original meaning and speaker's style:\n\nSpeaker: {speaker}\nUtterance: {utterance}\n\nRewritten Utterance:"
              inputs = self.tokenizer(prompt, return_tensors="pt")
              
              with torch.no_grad():
                outputs = self.llm.generate(**inputs, max_length=256, num_return_sequences=1)
              
              rewritten_utterance = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            rewritten_history.append(rewritten_utterance)
        return rewritten_history
    
    def rewrite_with_api(self, utterance):
        """
        Rewrite using the OpenAI API.
        """
        openai.api_key = self.config.llm_api_key

        prompt = (f"Please rewrite the following utterance to make it more formal and remove colloquial expressions, "
                  f"while preserving the original meaning and speaker's style:\n\n"
                  f"Utterance: {utterance}\n\n"
                  f"Rewritten Utterance:")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rewrites utterances."},
                {"role": "user", "content": prompt}
            ]
        )

        rewritten_utterance = response.choices[0].message['content'].strip()
        return rewritten_utterance

class MultiPartyAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        if config.model_name in ["losdf", "electra_baseline"]:
          self.encoder = AutoModel.from_pretrained(self.config.electra_model)
        elif config.model_name == "bert_baseline":
          self.encoder = AutoModel.from_pretrained(self.config.bert_model)
          
        self.W_q = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.W_k = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.W_v = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)

    def forward(self, input_ids, attention_mask, speaker_ids, token_type_ids=None):
        """
        Enforcement of the Multiparty Attention Mechanism.
        Args.
            input_ids: Input token ID.
            attention_mask: Attention mask.
            speaker_ids: Speaker ID.
            token_type_ids: optional, when the input is a pair of sentences, to distinguish sentences.
        Returns.
            A tensor representing the context representation after multiple attention mechanisms.
        """
        if self.config.model_name in ["losdf", "electra_baseline"]:
          outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        elif self.config.model_name == "bert_baseline":
          outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs.last_hidden_state # (batch_size, seq_len, hidden_size)
        
        # Get the representation of each utterance (the representation of the first token)
        utterance_representations = sequence_output[:, 0, :] # (batch_size, hidden_size)

        # 计算 Q, K, V
        Q = self.W_q(utterance_representations) # (batch_size, hidden_size)
        K = self.W_k(utterance_representations) # (batch_size, hidden_size)
        V = self.W_v(utterance_representations) # (batch_size, hidden_size)

        # Calculating Attention Scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.encoder.config.hidden_size ** 0.5) # (batch_size, batch_size)

        # Generate masks based on speaker_ids, ignore interactions between different speakers
        num_utterances = input_ids.size(0)
        speaker_mask = torch.zeros((num_utterances, num_utterances), device=input_ids.device)
        for i in range(num_utterances):
            for j in range(num_utterances):
                if speaker_ids[i] == speaker_ids[j]:
                    speaker_mask[i][j] = 1
                # Limiting the scope of interaction
                if abs(i - j) > self.config.max_interaction_range:
                    speaker_mask[i][j] = 0

        attention_scores = attention_scores.masked_fill(speaker_mask == 0, -1e9)

        # Softmax the attention scores
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # Pruning based on attention scores
        attention_probs = attention_probs.masked_fill(attention_probs < self.config.attention_threshold, 0)

        # Perform a weighted summation of V
        context_vector = torch.matmul(attention_probs, V) # (batch_size, hidden_size)
        
        return context_vector

class InformationDecouplingModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        hidden_size = 0
        if config.model_name in ["losdf", "electra_baseline"]:
          hidden_size = config.hidden_size
        elif config.model_name == "bert_baseline":
          hidden_size = config.hidden_size

        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_size, 2)  

    def forward(self, context_vector, question_vector):
        """
        Execution information decoupling.

        Args.
            context_vector: A representation of the context after a multi-party attention mechanism.
            question_vector: A representation of the question.

        Returns.
            A tensor representing the probability that each utterance is an answer.
        """
        # Splice contextual and problem representations
        combined_vector = torch.cat((context_vector, question_vector), dim=1) # (batch_size, 2 * hidden_size)

        # Feature extraction using a fully connected layer and activation function
        pooled_output = self.activation(self.fc(combined_vector)) # (batch_size, hidden_size)

        # Use a classifier to determine if each utterance is an answer
        logits = self.classifier(pooled_output) # (batch_size, 2)

        return logits