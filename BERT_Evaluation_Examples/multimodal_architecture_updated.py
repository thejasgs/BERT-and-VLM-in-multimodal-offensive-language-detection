import torch
import torch.nn as nn
from transformers import *
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel, PeftConfig

logging.set_verbosity_error()

# Maps model identifiers to (display_name, pooling_strategy, fusion_layer_sizes).
# Models with hidden_size < 512 start their fusion layers at 256.
MODEL_CONFIGS = {
    'google-bert/bert-base-uncased':              ('BaseBERT',       'mean', [512, 256, 128, 64]),
    'GroNLP/hateBERT':                            ('HateBERT',       'mean', [512, 256, 128, 64]),
    'FacebookAI/roberta-base':                    ('RoBERTa',        'mean', [512, 256, 128, 64]),
    'answerdotai/ModernBERT-base':                ('ModernBERT',     'mean', [512, 256, 128, 64]),
    'distilbert-base-uncased':                    ('DistilBERT',     'mean', [512, 256, 128, 64]),
    'ibm-granite/granite-embedding-30m-english':  ('IBM-Granite',    'mean', [256, 128, 64]),
    'sentence-transformers/all-MiniLM-L6-v2':     ('All-MiniLM-L6', 'mean', [256, 128, 64]),
}


class MultimodalClassifier(nn.Module):
    """
    Multimodal hate speech classifier combining a BERT-variant text encoder
    with pre-extracted image features.

    Text and image representations are each projected to a shared
    `projection_size`-dimensional space, concatenated, then passed through
    a configurable MLP fusion head.

    Args:
        num_labels:         Number of output classes.
        image_feature_size: Dimensionality of the incoming image feature vectors.
        model_name:         HuggingFace model ID; must be a key in MODEL_CONFIGS.
        projection_size:    Shared projection dimension for both text and image.
        fusion_layers:      Hidden layer sizes for the MLP head. Defaults to the
                            model's entry in MODEL_CONFIGS.
        dropout:            Dropout probability applied between fusion layers.
        class_weights:      Optional tensor of per-class weights for CrossEntropyLoss,
                            useful for imbalanced datasets.
    """

    def __init__(
        self,
        num_labels=2,
        image_feature_size=1280,
        model_name=None,
        projection_size=512,
        fusion_layers=None,
        dropout=0.1,
        class_weights=None,
    ):        
        super(MultimodalClassifier, self).__init__()

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: '{model_name}'.\n"
                f"Supported models: {list(MODEL_CONFIGS.keys())}"
            )

        display_name, self.pooling_strategy, default_fusion_layers = MODEL_CONFIGS[model_name]
        print(f"Loaded {display_name} — using '{self.pooling_strategy}' pooling, fusion layers: {default_fusion_layers}")

        self.model = AutoModel.from_pretrained(model_name)
        text_hidden_size = self.model.config.hidden_size

        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_size, projection_size),
            nn.LayerNorm(projection_size),
            nn.GELU()
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, projection_size),
            nn.LayerNorm(projection_size),
            nn.GELU()
        )

        fusion_layers = fusion_layers if fusion_layers is not None else default_fusion_layers
        self.fusion = self._build_fusion(
            in_size=projection_size * 2,
            hidden_sizes=fusion_layers,
            out_size=num_labels,
            dropout=dropout,
        )

        self.class_weights = class_weights

    def _build_fusion(self, in_size, hidden_sizes, out_size, dropout):
        """Builds the MLP fusion head: Linear -> GELU -> Dropout for each hidden layer."""
        layers = []
        current_size = in_size
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(current_size, hidden_size), nn.GELU(), nn.Dropout(dropout)]
            current_size = hidden_size
        layers.append(nn.Linear(current_size, out_size))
        return nn.Sequential(*layers)

    def _pool(self, last_hidden_state, attention_mask):
        """
        Pools the token-level hidden states into a single sentence embedding.

        Strategies:
            'cls':  Returns the [CLS] token embedding (index 0).
            'max':  Returns the element-wise max across non-padding tokens.
            'mean': Returns the attention-mask-weighted mean (default).
        """
        if self.pooling_strategy == 'cls':
            return last_hidden_state[:, 0, :]

        elif self.pooling_strategy == 'max':
            mask = attention_mask.unsqueeze(-1).float()
            masked = last_hidden_state * mask + (1 - mask) * -1e9
            return masked.max(dim=1).values

        else:
            mask = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (last_hidden_state * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, image_features, labels=None):
        """
        Forward pass.

        Args:
            input_ids:       Tokenized text input, shape (B, L).
            attention_mask:  Padding mask, shape (B, L).
            image_features:  Pre-extracted image embeddings, shape (B, image_feature_size).
            labels:          Optional ground-truth class indices, shape (B,).

        Returns:
            dict with 'logits' (and 'loss' if labels were provided).
        """
        text_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self._pool(text_outputs.last_hidden_state, attention_mask)

        text_features = self.text_projection(text_features)
        image_features = self.image_projection(image_features)

        combined_features = torch.cat([text_features, image_features], dim=1)
        logits = self.fusion(combined_features)

        loss = None
        if labels is not None:
            weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


class LoRAMultiModalClassifier(nn.Module):
    """
    Multimodal classifier identical in architecture to MultimodalClassifier,
    but with Low-Rank Adaptation (LoRA) applied to the text encoder's linear layers.

    LoRA freezes the base model weights and injects trainable low-rank matrices,
    significantly reducing the number of parameters updated during fine-tuning.

    Args:
        num_labels:         Number of output classes.
        projection_size:    Shared projection dimension for both modalities.
        image_feature_size: Dimensionality of incoming image feature vectors.
        fusion_layers:      Hidden layer sizes for the MLP fusion head.
        class_weights:      Optional per-class loss weights.
        lora_r:             LoRA rank (controls capacity of adapter matrices).
        lora_alpha:         LoRA scaling factor.
        lora_dropout:       Dropout applied inside LoRA adapters and fusion layers.
        model_used:         HuggingFace model ID; must be a key in MODEL_CONFIGS.
    """

    def __init__(
        self,
        num_labels=2,
        projection_size=512,
        image_feature_size=1280,
        fusion_layers=None,
        class_weights=None,
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        model_used=None,
    ):
        super(LoRAMultiModalClassifier, self).__init__()

        self.model = AutoModel.from_pretrained(model_used)

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
        )
        self.model = get_peft_model(self.model, peft_config)
        text_hidden_size = self.model.config.hidden_size

        if model_used not in MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: '{model_used}'.\n"
                f"Supported models: {list(MODEL_CONFIGS.keys())}"
            )

        display_name, self.pooling_strategy, default_fusion_layers = MODEL_CONFIGS[model_used]
        print(f"Loaded {display_name} — using '{self.pooling_strategy}' pooling, fusion layers: {default_fusion_layers}")

        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_size, projection_size),
            nn.LayerNorm(projection_size),
            nn.GELU()
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, projection_size),
            nn.LayerNorm(projection_size),
            nn.GELU()
        )

        fusion_layers = fusion_layers if fusion_layers is not None else default_fusion_layers
        self.fusion = self._build_fusion(
            in_size=projection_size * 2,
            hidden_sizes=fusion_layers,
            out_size=num_labels,
            dropout=lora_dropout,
        )

        self.class_weights = class_weights

    def _build_fusion(self, in_size, hidden_sizes, out_size, dropout):
        """Builds the MLP fusion head: Linear -> GELU -> Dropout for each hidden layer."""
        layers = []
        current_size = in_size
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(current_size, hidden_size), nn.GELU(), nn.Dropout(dropout)]
            current_size = hidden_size
        layers.append(nn.Linear(current_size, out_size))
        return nn.Sequential(*layers)

    def _pool(self, last_hidden_state, attention_mask):
        """
        Pools token-level hidden states into a sentence embedding.
        See MultimodalClassifier._pool for strategy descriptions.
        """
        if self.pooling_strategy == 'cls':
            return last_hidden_state[:, 0, :]
        elif self.pooling_strategy == 'max':
            mask = attention_mask.unsqueeze(-1).float()
            masked = last_hidden_state * mask + (1 - mask) * -1e9
            return masked.max(dim=1).values
        else:
            mask = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (last_hidden_state * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, image_features, labels=None):
        """
        Forward pass. Identical signature and return format to MultimodalClassifier.forward.
        """
        text_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self._pool(text_outputs.last_hidden_state, attention_mask)

        text_features = self.text_projection(text_features)
        image_features = self.image_projection(image_features)

        combined_features = torch.cat([text_features, image_features], dim=1)
        logits = self.fusion(combined_features)

        loss = None
        if labels is not None:
            weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


class MultimodalDataCollator:
    """
    Data collator for multimodal batches containing tokenized text and image features.

    Pads text sequences to the longest sequence in the batch and stacks
    image features and labels into tensors.

    Args:
        tokenizer: HuggingFace tokenizer used for padding.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        image_features = [np.array(f['image_features'], dtype=np.float32) for f in features]
        labels = [f['label'] for f in features]

        batch = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            padding=True,
            return_tensors='pt'
        )

        batch['image_features'] = torch.from_numpy(np.stack(image_features))
        batch['labels'] = torch.tensor(labels, dtype=torch.long)

        return batch


class MultimodalTrainer(Trainer):
    """
    HuggingFace Trainer subclass that routes multimodal batch fields
    to the model's forward method.

    Overrides compute_loss to explicitly pass image_features alongside
    the standard text inputs, since Trainer's default loop does not
    handle non-standard batch keys.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_features=inputs["image_features"],
            labels=labels
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


class BestEpochCallback(TrainerCallback):
    """
    Tracks the best and most recent evaluation metrics across training epochs.

    'Best' is defined as the epoch with the highest F1 score.

    Attributes:
        best_f1, best_loss, best_epoch, best_metrics: Metrics at the best epoch.
        last_f1, last_loss, last_epoch:               Metrics at the final epoch.
    """

    def __init__(self):
        self.best_loss = float("inf")
        self.best_f1 = float("-inf")
        self.best_epoch = None
        self.best_metrics = None
        self.last_f1 = float("-inf")
        self.last_loss = float("inf")
        self.last_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_f1" in metrics and "eval_loss" in metrics:
            current_f1 = metrics.get('eval_f1', 0.0)
            current_loss = metrics.get('eval_loss', float('inf'))

            if isinstance(current_loss, torch.Tensor):
                current_loss = current_loss.item()

            self.last_loss = current_loss
            self.last_f1 = current_f1
            self.last_epoch = state.epoch

            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_loss = current_loss
                self.best_epoch = state.epoch
                self.best_metrics = metrics.copy()

    def on_train_end(self, args, state, control, metrics=None, **kwargs):
        pass