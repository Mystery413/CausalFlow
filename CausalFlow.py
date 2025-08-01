import pandas as pd
import numpy as np
import random, string, math, json, ast, re, textwrap, torch
from pgmpy.estimators import PC
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import bnlearn as bn

# ===== 参数 =====
ALPHA          = 0.01
DATA_CSV       = "asia_50000.csv"      # 数据集
BIF_FILE       = "asia.bif"            # 真值网络
MODEL_DIR      = "../deepseek-7b"     # 本地模型目录
SEED_LIST      = range(1, 31)           # 30 个随机种子
NUM_ROWS_LIST  = [250,500,1000,5000,10000]
NUM_ROWS_LIST  = [10000]
OUTPUT_CSV     = "1111_sachs_llm_results.csv"

# ===== LLM 加载 =====
device   = "cuda" if torch.cuda.is_available() else "cpu"
dtype    = torch.float16 if device == "cuda" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None
)

# ===== 真值图 =====
true_edges = set(bn.import_DAG(BIF_FILE)['model'].edges())

# ===== 辅助函数 =====
EDGE_RE = re.compile(
    r"processed.\nAnswer:.*?(\[\s*(?:\(\s*['\"][^'\"]+['\"]\s*,\s*['\"][^'\"]+['\"]\s*\)\s*,?\s*)+\])",
    re.S
)

def safe_parse(txt):
    m = EDGE_RE.search(txt)
    if not m:
        return []
    raw = m.group(1).replace("“", '"').replace("”", '"')
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(raw)
        except:
            pass
    return []

def cond_entropy(ct_df):
    p = ct_df / ct_df.values.sum()
    h = 0.0
    for i in p.index:
        px = p.loc[i].sum()
        if px == 0:
            continue
        for pij in p.loc[i]:
            if pij:
                h -= pij * math.log2(pij / px)
    return h

def random_alias(cols):
    available_letters = [ch for ch in string.ascii_uppercase if ch not in {"A", "B"}]
    letters = random.sample(available_letters, len(cols))
    amap = dict(zip(cols, letters))
    return amap, {v: k for k, v in amap.items()}

def chat(prompt, max_new=20):
    cfg = GenerationConfig(temperature=0, do_sample=False, max_new_tokens=max_new)
    # full = (
    #     "<|system|>\nYou are an expert in causal discovery.\n<|user|>\n"
    #     + prompt
    #     + "\n<|assistant|>\nEDGE_LIST = "
    # )
    full = "You are an expert in causal discovery.\n" + prompt + "\nAnswer:"
    inputs = tokenizer(full, return_tensors="pt").to(device)
    out = model.generate(**inputs, generation_config=cfg)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def shd(pred_edges, true_edges):
    return len(pred_edges - true_edges) + len(true_edges - pred_edges)

# ===== 主流程 =====
df_raw = pd.read_csv(DATA_CSV)
all_results = []

reverse_prompt_order_count = 0
total_prompt_calls = 0

for N in NUM_ROWS_LIST:
    shd_vals = []
    print(f"\n====== NUM_ROWS = {N} ======")
    for seed in SEED_LIST:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        df_sample = df_raw.sample(n=N, random_state=seed).reset_index(drop=True)

        alias_map, rev_map = random_alias(df_sample.columns)
        df = df_sample.rename(columns=alias_map)

        skeleton, _ = PC(df).build_skeleton(significance_level=ALPHA, test='chi_square')
        final_edges = set()

        for a, b in skeleton.edges():
            joint_ct = pd.crosstab(df[a], df[b])
            h_b_a = cond_entropy(joint_ct)      # H(b|a)
            h_a_b = cond_entropy(joint_ct.T)    # H(a|b)

            jt_lines = [
                f"{a}={i}, {b}={j}: {int(joint_ct.at[i, j])}"
                for i in joint_ct.index
                for j in joint_ct.columns
            ]
            joint_txt = "\n".join(jt_lines)

            prompt = textwrap.dedent(f"""
            You are given two anonymized variables: {a} and {b}.
   
            Joint frequency counts of {a} and {b}:
            {joint_txt}

            Conditional entropies:
            H({a}|{b}) = {h_a_b:.4f}
            H({b}|{a}) = {h_b_a:.4f}

            Decide the most likely causal direction.

            Example:
            Variables: A and B. A causes B.
            Answer in this template: EDGE_LIST = [("A", "B")]

            You must choose one direction or your answer can not be processed.
            """).strip()

            resp = chat(prompt)
            print(resp)
            parsed = safe_parse(resp)

            total_prompt_calls += 1
            if len(parsed) == 1 and len(parsed[0]) == 2:
                src, tgt = parsed[0]
                # 只在 src 和 tgt 都在 rev_map 中才处理
                if src in rev_map and tgt in rev_map:
                    final_edges.add((rev_map[src], rev_map[tgt]))
                    if src == b and tgt == a:
                        reverse_prompt_order_count += 1

        shd_vals.append(shd(final_edges, true_edges))

    mu, sigma = np.mean(shd_vals), np.std(shd_vals)
    all_results.append(
        {
            "n_rows": N,
            "mean_shd": round(mu, 2),
            "std_shd": round(sigma, 2),
            "formatted": f"{mu:.2f} ± {sigma:.2f}",
        }
    )

df_out = pd.DataFrame(all_results)
#df_out.to_csv(OUTPUT_CSV, index=False)
print("\n✅ All experiments finished. Results saved to", OUTPUT_CSV)
print(df_out)
print(f"\n📊 LLM returned (b, a) order {reverse_prompt_order_count} times out of {total_prompt_calls} prompts "
      f"({reverse_prompt_order_count / total_prompt_calls:.2%})")
