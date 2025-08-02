# 🧾 BwETAFv2-130M: Model Card

**Boring’s Experimental Transformer for Autoregression (Flax)**
A 130M parameter autoregressive language model built using a custom Flax pipeline, loads of tuning, and a sprinkle of existential dread.

> *Trained on determination, fueled by suffering, powered by free TPUs.*

---

## 📌 Model Summary

* **Model Name:** BwETAFv2-130M
* **Parameters:** 130,649,088
* **Training Tokens:** 3,014,656,000
* **Training Time:** 19.32 TPUv2-8 Hours
* **Framework:** Flax / JAX
* **Max Context Length:** 4096 tokens
* **Tokenizer:** GPT-2 BPE (50,257 vocab size)

---

## 🧪 Hyperparameters

```json
{
  num_heads=12,
  attention_dim=768,
  vocab_size=50260,
  num_blocks=12,
  ff_dim=2304,
  dropout_rate=0.1,
  max_len = 8192,
  use_fash_attention = False,
  attn_chunks = 1,
  emb_init_range = 0.02,
  use_rope = True,
  emb_scaling_factor = 1,
  res_scale = 1
}
```

---

## 🛠 Optimizer Settings

```json
{
  "peaklr":5e-4,
  "warmup_percent":0.03,
  "min_value":1e-8,
  "training_decay":"cosine",
  "weight_decay": 0.1,
  "min_warmup_value":2e-4,
  "b1": 0.9,
  "b2": 0.95,
  "eps": 1e-7
}
```

---

## 📈 Performance

* **Final Validation Loss:** `3.59`

* **Validation loss Graphs:**
![image/png](https://cdn-uploads.huggingface.co/production/uploads/661e235e08dd378c818654ad/2JNuMflvNviXtF8I4dRep.png)

* **Training loss Graphs:**
![image/png](https://cdn-uploads.huggingface.co/production/uploads/661e235e08dd378c818654ad/chN0GMaekmWcDU6xbTPUK.png)

* For detailed stats, refer to `stats.json` in the model files.

---

## ⚡ Quickstart

```bash
pip install BwETAF==0.5.1
```

```python
import BwETAF

# 🔍 Quick API test
prompt = "The meaning of life is"
output = BwETAF.SetUpAPI(prompt, "WICKED4950/BwETAFv2-130M")
print(output)

# ⬇️ Load from Hugging Face Hub
model = BwETAF.load_hf("WICKED4950/BwETAFv2-130M")

# 📁 Load from local path
BwETAF.load_model("path/to/model")

# 💾 Save to local directory
model.save_model("path/to/save")

# 🔧 Inspect model
params = model.trainable_variables
structure = model.model_struct
```

> ☁️ *Colab support and examples coming soon!*

---

## 🧠 BwETAFv2 Architecture Overview

The **BwETAFv2** architecture introduces several refinements over its predecessors, improving convergence, training stability, and scalability. Below is an architecture-level overview shared across all models in this series.

---

### 🔩 Core Architecture

* **Attention:** Standard multi-head self-attention (no GQA, no MQA, no FlashAttention)
* **FFN:** Uses **Swish-GLU** with FF dimension = `3 × model_dim`
* **Norm Layer:** **RMSNorm**, pre-layer
* **Positional Encoding:** **RoPE** on embeddings and K/Q vectors
* **Precision:**

  * Weights & forward pass: `bfloat16 (bf16)`
  * Optimizer states: `float32`
* **Tokenization:** GPT-2 BPE (`vocab_size = 50257`)
* **Bias Terms:** No biases in K/Q/V dense projections

---

## ⚙️ Training Setup

* **Optimizer:** AdamW
* **Batch Size:** 32 per step
* **Learning Schedule:** Cosine decay with linear warmup
* **Chinchilla Scaling Law:** Followed (20× tokens per parameter)
* **Context Window during Training:**

  * BwETAFv2-53M: 2048
  * BwETAFv2-130M: 4096

---

## 📊 Model Comparison

| Model Name    | Params | Tokens Seen | TPUv2-8 Hours | Val Loss | Context Length |
| ------------- | ------ | ----------- | ------------- | -------- | -------------- |
| BwETAFv2-53M  | 53M    | 1.34B       | 3.21          | 3.77     | 2048           |
| BwETAFv2-130M | 130M   | 3.01B       | 19.32         | 3.59     | 4096           |

---

## 📬 Contact Me

* 📸 Instagram: [boring.\_.wicked](https://www.instagram.com/boring._.wicked/)
* 💬 Discord: `fused_computation.1` *(if you spot me lurking in any AI-related servers)*

---
