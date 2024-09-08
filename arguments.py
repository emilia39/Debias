import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import HfArgumentParser, TrainingArguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class OtherArguments:
    debias_type: str = field(
        default="gender",
        metadata={"help": "Bias type.",
        "choices": ["gender"]}
    )
    log_dir: str = field(
        default="log",
        metadata={"help": "The log directory where to save log file."}
    )
    model_type: str = field(
        default="bert",
        metadata={"help": "Choose from ['bert','roberta','albert']"}
    )
    alpha:float =field(
        default=0.3,
        metadata={"help": "Weight of Loss Function 1"}
    )
    beta:float =field(
        default=0.7,
        metadata={"help": "Weight of Loss Function 2"}
    )
    block_size: int = field(
        default=128, 
        metadata={"help": "Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens)."}
    )
    loss_target: str = field(
        default="token",
        metadata={"help": "Debias at token-level or sentence-level."}
    )
    debias_layer: str = field(
        default="all",
        metadata={"help": "Debias the first, last or all layers of a PLM.",
        "choices": ['all', 'first', 'last']}
    )

def get_args():
    #Modify the default value of output_dir in TrainingArguments
    parser = HfArgumentParser((ModelArguments, TrainingArguments, OtherArguments))
    model_args, training_args, other_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(training_args), **vars(other_args))
    args.model_name_or_path = model_args.model_name_or_path
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    return model_args, args

if __name__ == "__main__":
    model_args, args=get_args()
    # print(args.model_type)
    # print(args.alpha,args.beta)