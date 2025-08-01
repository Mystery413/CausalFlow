import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import pandas as pd
import os
import re
import transformers
from sklearn.metrics import f1_score

# 关闭 transformers 的日志输出
transformers.logging.set_verbosity_error()

def load_model(locationBaseModel, locationLoRAModel):
    """加载基础模型和LoRA微调模型"""
    tokenizer = AutoTokenizer.from_pretrained(locationBaseModel)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        locationBaseModel,
        device_map="auto",
        torch_dtype=torch.float16
    )

    config = PeftConfig.from_pretrained(locationLoRAModel)
    model = PeftModel.from_pretrained(base_model, locationLoRAModel)
    
    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device

def generate_predictions(model, tokenizer, device, df, max_new_tokens=10):
    """对输入的DataFrame执行推理，返回预测结果"""
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(df.shape[0]):
            prompt = str(df["prompt"].values[i]) + "\nAnswer"
            inputs = tokenizer([prompt], return_tensors="pt").to(device)

            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                temperature=0
            )
            decoded = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            predictions.append(decoded[0])

            if i % 10 == 0:
                print("Processed row", i)
    
    return predictions

def extract_first_word(text):
    """从模型输出的文本中提取'Answer: Yes' 或 'Answer: No'"""
    if pd.isna(text):
        return None
    match = re.search(r"Answer:[\s\S]*?(Yes|No)", text)
    return match.group(1) if match else None

def process_and_evaluate(predictions, gold_file):
    """处理模型输出，提取回答，并计算 matching_ratio 和 F1 Score"""
    
    # 读取gold标准答案
    gold_data = pd.read_csv(gold_file)

    # 创建DataFrame存储模型预测
    df = pd.DataFrame({'pred': predictions})

    # 提取 Yes/No 答案
    df['answer'] = df['pred'].apply(extract_first_word)

    # 计算 matching_ratio
    total_rows = len(df)
    matching_rows = (df['answer'] == gold_data['label']).sum()
    matching_ratio = matching_rows / total_rows if total_rows > 0 else 0

    # 计算 F1 Score
    df['answer_binary'] = df['answer'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    gold_data['gold_binary'] = gold_data['label'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    
    f1 = f1_score(gold_data['gold_binary'], df['answer_binary'])
    macro_f1 = f1_score(gold_data['gold_binary'], df['answer_binary'], average="macro")

    return matching_ratio, f1, macro_f1

def main(locationBaseModel, locationLoRAModels, outputFileName, inputFileName, goldFileName):
    """测试多个LoRA微调模型，计算matching_ratio和F1 Score，并实时保存到CSV"""
    
    # 读取输入数据
    df = pd.read_csv(inputFileName)
    print(f"Total rows in {inputFileName} = {df.shape[0]}")

    # 如果文件不存在，写入表头
    if not os.path.exists(outputFileName):
        pd.DataFrame(columns=['model', 'matching_ratio', 'f1_score', 'macro_f1_score']).to_csv(outputFileName, index=False)

    # 逐个加载LoRA模型进行推理
    for locationLoRAModel in locationLoRAModels:
        print(f"Testing model: {locationLoRAModel}")

        try:
            # 加载模型
            tokenizer, model, device = load_model(locationBaseModel, locationLoRAModel)
            
            # 生成预测
            predictions = generate_predictions(model, tokenizer, device, df)
            
            # 处理预测结果并计算 matching_ratio 和 F1 Score
            matching_ratio, f1, macro_f1 = process_and_evaluate(predictions, goldFileName)

            # 存储模型匹配率和F1 Score
            model_name = os.path.basename(locationLoRAModel)
            result_df = pd.DataFrame([{'model': model_name, 'matching_ratio': matching_ratio, 'f1_score': f1, 'macro_f1_score' : macro_f1}])

            # 追加保存结果
            result_df.to_csv(outputFileName, mode='a', header=False, index=False)
            print(f"Results saved for {model_name}: Matching Ratio = {matching_ratio:.4f}, F1 Score = {f1:.4f}, macro f1 score = {macro_f1_score:.4f}")

        except Exception as e:
            print(f"Error testing model {locationLoRAModel}: {e}")

        finally:
            # 释放 GPU 显存
            del model
            torch.cuda.empty_cache()
            print(f"Released GPU memory after testing {locationLoRAModel}")

if __name__ == "__main__":
    locationBaseModel = "../../deepseek-7b"  # 预训练模型路径
    locationLoRAModels = [
    "deepseek-20w-ckpt/checkpoint-2000",
    "deepseek-20w-ckpt/checkpoint-4000",
    "deepseek-20w-ckpt/checkpoint-6000",
    "deepseek-20w-ckpt/checkpoint-8000",
    "deepseek-20w-ckpt/checkpoint-10000",
    "deepseek-20w-ckpt/checkpoint-12000",
    "deepseek-20w-ckpt/checkpoint-14000",
    "deepseek-20w-ckpt/checkpoint-16000",
    "deepseek-20w-ckpt/checkpoint-18000",
    "deepseek-20w-ckpt/checkpoint-20000",
    "deepseek-20w-ckpt/checkpoint-22000",
    "deepseek-20w-ckpt/checkpoint-24000",
    "deepseek-20w-ckpt/checkpoint-26000",
    "deepseek-20w-ckpt/checkpoint-28000",
    "deepseek-20w-ckpt/checkpoint-30000",
    "deepseek-20w-ckpt/checkpoint-32000",
    "deepseek-20w-ckpt/checkpoint-34000",
    "deepseek-20w-ckpt/checkpoint-36000"

]
    outputFileName = "79_test_a.csv"  # 结果输出文件
    inputFileName = "../imbalanced_given_test.csv"  # 测试集
    goldFileName = "../imbalanced_given_test.csv"  # 标准答案文件

    main(locationBaseModel, locationLoRAModels, outputFileName, inputFileName, goldFileName)
