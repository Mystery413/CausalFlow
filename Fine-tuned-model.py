import os
import torch
import time
from datasets import load_dataset
from tabulate import tabulate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载数据集
train_dataset = load_dataset("csv", data_files="train_whole.csv", split="train")
dev_dataset = load_dataset("csv", data_files="dev_whole.csv", split="train")

# 2. 加载分词器和基础模型
base_model_path = "../deepseek-7b"  # 请自行替换为实际模型路径
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 如果模型没有 pad_token，就设置为 eos_token（避免后续 padding 出错）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# 4. 数据预处理
def preprocess_function(examples):
    combined_texts = []
    for inp, tgt in zip(examples["input"], examples["gold"]):
        text = f"{inp}\nAnswer:\n{tgt}"
        combined_texts.append(text)

    tokenized = tokenizer(
        combined_texts,
        truncation=True,
        max_length=1024,  # 可根据需要调小，例如 512 或 768
        add_special_tokens=True
    )
    return tokenized

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

dev_dataset = dev_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dev_dataset.column_names
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="aaai/deepseek-20w-ckpt",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    num_train_epochs=3,
    logging_steps=2000,
    learning_rate=2e-5,
    fp16=True,
    report_to=[],
)

# 时间记录回调
class TimeLoggingCallback(TrainerCallback):
    def __init__(self, start_time):
        super().__init__()
        self.start_time = start_time
        self.total_eval_time = 0

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_start_time = time.time()

    def on_evaluate_end(self, args, state, control, **kwargs):
        self.total_eval_time += time.time() - self.eval_start_time
        
    def on_save(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time - self.total_eval_time  # 排除评估时间
        with open('checkpoint_times.log', 'a') as f:
            f.write(f"Checkpoint saved at step {state.global_step}. Effective elapsed time (excluding evaluation): {elapsed_time:.2f} seconds.\n")

# 6. 自定义 Trainer，优化日志格式
class CustomTrainer(Trainer):
    def log(self, logs, *args, **kwargs):  
        if "eval_loss" in logs or "loss" in logs:
            table_data = [[
                logs.get("step", "-"),
                logs.get("loss", "-"),
                logs.get("eval_loss", "-"),
                logs.get("grad_norm", "-"),
                logs.get("learning_rate", "-"),
                logs.get("epoch", "-")
            ]]
            print(tabulate(
                table_data,
                headers=["Step", "Training Loss", "Validation Loss", "Grad Norm", "Learning Rate", "Epoch"],
                tablefmt="grid"
            ))

# 记录训练开始时间
start_time = time.time()

# **使用 CustomTrainer**
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    callbacks=[TimeLoggingCallback(start_time)]   # 添加时间记录回调
)

trainer.train()

# 8. 保存模型和分词器
model.save_pretrained("aaai/deepseek-20w-model")
tokenizer.save_pretrained("aaai/deepseek-20w-model")

