from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 2.2GB, real safetensors

print("loading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_id)

print("loading model to MPS...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="mps",  # uses your M5 GPU
    attn_implementation="eager"
)

prompt = "Jane is a girl. Jane"
inputs = tok(prompt, return_tensors="pt").to("mps")
print("inputs:", inputs)
with torch.no_grad():
    out = model(**inputs, output_attentions=True)

# last layer, head 0 — change head number to explore

#print("outputs:", out)

# attn = out.attentions[-1][0, 3].cpu().numpy()

# tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])
# print("tokens:", tokens)

# layer = 21 # last layer
# for h in range(4): # TinyLlama uses GQA, heads 0-3 are distinct groups
#     a = out.attentions[layer][0, h].cpu().numpy()
#     girl_idx = tokens.index('▁girl')
#     jane_idx = tokens.index('▁Jane')
#     print(f"head {h}: girl→Jane = {a[girl_idx, jane_idx]:.3f}")

tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])
j1, girl, j2 = 1, 4, 6

best = (0,0,0)
for layer in range(15, 22): # top third
    for h in range(32):
        a = out.attentions[layer][0, h].cpu().numpy()
        score = a[j2, j1] + a[j2, girl] # second Jane looks back
        if score > best[0]:
            best = (score, layer, h)

print(f"best = {best[0]:.3f} at layer {best[1]} head {best[2]}")

# plot it

a = out.attentions[best[1]][0,best[2]].cpu().numpy()
j2 = 6 # second Jane
print("--------------------")
print(list(zip(tokens, a.round(2))))
print("--------------------")
print()
a = out.attentions[best[1]][0, best[2]].cpu().numpy()
import matplotlib.pyplot as plt
plt.imshow(a, vmin=0, vmax=0.3); plt.xticks(range(len(tokens)), tokens, rotation=45); plt.yticks(range(len(tokens)), tokens); plt.title(f"L{best[1]} H{best[2]}"); plt.show()


