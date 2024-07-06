import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from software.qabba import QABBA
import os
import pickle
from sklearn import preprocessing
from sklearn.metrics import matthews_corrcoef, accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, f1_score
from src.preprocessing import encoders, vector_embed
import torch
import math
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
import torch.nn.functional as F

import evaluate
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import warnings

from utils.data_loader import load_from_tsfile_to_dataframe
from utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from utils.tools import create_directory
from utils.transformer_tools import fit_transformer

torch.cuda.empty_cache()


# Python program to convert a list to string
def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele + ' '
    # return string
    return str1

class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(label_weights, device=model.device, dtype=logits.dtype))
        #         loss = loss_fct(logits, logits).mean()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def model_preprocessing_function(examples):
    return model_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)


class MSELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute MSE loss
        #         loss = F.mse_loss(logits.squeeze(), labels.float())
        loss = F.mse_loss(logits.squeeze(), labels.float()) / batch_size
        #         total_loss += loss.item()

        return (loss, outputs) if return_outputs else loss



def main(opts):

    warnings.filterwarnings("ignore")

    global label_weights
    global num_classes
    global model_tokenizer
    global MAX_LENGTH
    global batch_size

    batch_size = opts.batch_size
    MAX_LENGTH = opts.MAX_LENGTH
    num_classes = 1

    ##  Quantization Coonfig
    quantization_config = BitsAndBytesConfig(
        # Load the model with 4-bit quantization
        load_in_4bit=True,
        # Use double quantization
        bnb_4bit_use_double_quant=True,
        # Use 4-bit Normal Float for storing the base model weights in GPU memory
        bnb_4bit_quant_type="nf4",
        # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # lora config
    if opts.model_name == "roberta-large":

        model_checkpoint = "roberta-large"
        model_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                        padding_side="left",
                                                        add_eos_token=True,
                                                        add_prefix_space=True
                                                        )
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        model_tokenizer.pad_token = model_tokenizer.eos_token

        model_input = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            num_labels=num_classes,
            device_map="auto"
        )

        # Data collator for padding a batch of examples to the maximum length seen in the batch
        model_data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer)

        model_input.config.pad_token_id = model_input.config.eos_token_id

        model_input = prepare_model_for_kbit_training(model_input)

        roberta_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=opts.lora_r,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )
        model_input = get_peft_model(model_input, roberta_peft_config)
    elif opts.model_name == "mistral-7B":

        model_checkpoint = 'mistralai/Mistral-7B-Instruct-v0.1'
        model_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                        padding_side="left",
                                                        add_eos_token=True,
                                                        add_prefix_space=True
                                                        )
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        model_tokenizer.pad_token = model_tokenizer.eos_token

        model_input = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            num_labels=num_classes,
            device_map="auto"
        )

        # Data collator for padding a batch of examples to the maximum length seen in the batch
        model_data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer)

        model_input.config.pad_token_id = model_input.config.eos_token_id
        model_input = prepare_model_for_kbit_training(model_input)

        mistral_lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=opts.lora_r,  # the dimension of the low-rank matrices
            lora_alpha=16,  # scaling factor for LoRA activations vs pre-trained weight activations
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            lora_dropout=0.05,  # dropout probability of the LoRA layers
            bias='none',  # wether to train bias weights, set to 'none' for attention layers
        )
        model_input = get_peft_model(model_input, mistral_lora_config)
    elif opts.model_name == "llama2-7B":

        model_checkpoint = "starmpcc/Asclepius-Llama2-7B"
        model_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                        padding_side="left",
                                                        add_eos_token=True,
                                                        add_prefix_space=True
                                                        )
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        model_tokenizer.pad_token = model_tokenizer.eos_token

        model_input = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            num_labels=num_classes,
            device_map="auto"
        )

        # Data collator for padding a batch of examples to the maximum length seen in the batch
        model_data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer)

        model_input.config.pad_token_id = model_input.config.eos_token_id
        model_input = prepare_model_for_kbit_training(model_input)
        
        llama_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=opts.lora_r,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        )
        model_input = get_peft_model(model_input, llama_peft_config)

    else:
        print("Please input correct models!")

    model_input.print_trainable_parameters()
    model_input = model_input.cuda()

    ###############   Monash Regression data   ###############

    # data_name = ['AppliancesEnergy', 'HouseholdPowerConsumption1', 'HouseholdPowerConsumption2', 'BenzeneConcentration',
    #              'BeijingPM25Quality', 'BeijingPM10Quality', 'LiveFuelMoistureContent', 'FloodModeling1',
    #              'FloodModeling2',
    #              'FloodModeling3', 'AustraliaRainfall', 'PPGDalia', 'IEEEPPG', 'BIDMCRR', 'BIDMCHR', 'BIDMCSpO2',
    #              'NewsHeadlineSentiment',
    #              'NewsTitleSentiment', 'Covid3Month']

    data_folder = 'data/monash-regression/'
    train_file = data_folder + opts.data_name + "_TRAIN.ts"
    test_file = data_folder + opts.data_name + "_TEST.ts"

    X_train, y_train = load_from_tsfile_to_dataframe(train_file)
    X_test, y_test = load_from_tsfile_to_dataframe(test_file)
    norm = "minmax"  # none, standard, minmax

    train_test_split = [X_train.shape[0], X_test.shape[0]]
    data_all = pd.concat([X_train, X_test])
    target_scaled = np.concatenate([y_train, y_test])

    min_len = np.inf
    for i in range(len(data_all)):
        x = data_all.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    # print("[{}] Minimum length: {}".format(module, min_len))

    data_scaled = process_data(data_all, normalise=norm, min_len=min_len)

    qabba = QABBA(tol=opts.ABBA_tol, init=opts.ABBA_init,
                  alpha=opts.ABBA_alpha, bits_for_len=opts.bits_for_len,
                  bits_for_inc=opts.bits_for_inc, scl=opts.ABBA_scl, verbose=1)

    symbols = qabba.fit_transform(data_scaled)
    # reconstruction = jabba.inverse_transform(symbols)
    print('##############################################################')
    print("[{}] Task: {}".format(opts.model_name, opts.data_name))
    print("The length of used symbols is:" + str(qabba.parameters.centers.shape[0]))


    symbols_convert = []
    for i_data in range(len(symbols)):
        symbols_convert.append(listToString(list(symbols[i_data])))

    train_data_symbolic, test_data_symbolic, train_target_symbolic, test_target_symbolic = \
        symbols_convert[:train_test_split[0]], symbols_convert[train_test_split[0]:], \
        target_scaled[:train_test_split[0]], target_scaled[train_test_split[0]:]

    data_TS = DatasetDict({
        'train': Dataset.from_dict({'label': train_target_symbolic, 'text': train_data_symbolic}),
        'test': Dataset.from_dict({'label': test_target_symbolic, 'text': test_data_symbolic})
    })

    model_tokenized_datasets = data_TS.map(model_preprocessing_function, batched=True)
    model_tokenized_datasets.set_format("torch")

    model_data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer)



    project = "ts-finetune-" + opts.data_name
    # b-instruct-v0.1-h
    base_model_name = "llama2"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=20,
        optim="paged_adamw_8bit",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=True,
        gradient_checkpointing=True,
        save_total_limit=2,
    )

    trainer_abba = MSELossTrainer(
        model=model_input,
        args=training_args,
        train_dataset=model_tokenized_datasets['train'],
        eval_dataset=model_tokenized_datasets["test"],
        data_collator=model_data_collator,
        #     compute_metrics=compute_metrics,
    )
    trainer_abba.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Times Series Regression')

    parser.add_argument('--data_name', type=str, default='None')
    parser.add_argument('--model_name', type=str, default='llama2-7B')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--ABBA_tol', type=float, default=0.05)
    parser.add_argument('--ABBA_alpha', type=float, default=0.05)
    parser.add_argument('--ABBA_init', type=str, default='agg')
    parser.add_argument('--ABBA_k', type=int, default=1000)
    parser.add_argument('--ABBA_scl', type=int, default=3)
    parser.add_argument('--bits_for_len', type=int, default=16)
    parser.add_argument('--bits_for_inc', type=int, default=16)
    parser.add_argument('--UCR_data_num', type=int, default=1)
    parser.add_argument('--MAX_LENGTH', type=int, default=2048)

    args = parser.parse_args()
    # print('***********************************************************************************')
    main(args)
    # print('***********************************************************************************')




