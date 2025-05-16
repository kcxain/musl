from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="")
    max_length: int = field(default=2048, metadata={"help": "The maximum total input sequence length after tokenization."})


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_dir: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    max_samples: int = field(
        default=None, metadata={"help": "If set, will limit the number of training examples."}
    )
    source: str = field(
        default="cpp", metadata={"help": "Source language"}
    )
    target: str = field(
        default="cuda", metadata={"help": "Target language"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch")
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: float = field(
        default=0.2,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_steps: float = field(
        default=0.2,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    per_device_train_batch_size: int = field(
        default=2, 
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    learning_rate: float = field(default=1.0e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    logging_steps: float = field(
        default=10,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    optim: str = field(
        default="adamw_8bit",
        metadata={"help": "The optimizer to use."},
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )