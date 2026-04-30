token_pos = 2 # third token (0-indexed)
batch_idx = 0

def inspect_token(model, idx):
    model.eval()
    with torch.no_grad():
        B, T = idx.shape
        tok_emb = model.token_embedding_table(idx) # (1, T, 4)
        pos_emb = model.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        print(f"INPUT token id={idx[0,token_pos].item()} → embedding ={tok_emb[0,token_pos].cpu().numpy()}")
        print(f"+ positional embedding [{token_pos}]={pos_emb[token_pos].cpu().numpy()}")
        print(f"= Sum (X) = {x[0,token_pos].cpu().numpy()}\n")

        # go through first block only
        block = model.blocks[0]
        x = block.ln1(x)
        print(f"After layer norm 1: {x[0,token_pos].cpu().numpy()}")

        # inspect each head
        for i, head in enumerate(block.sa.heads):
            q = head.query(x) # (1,T,1) (size head size is 1)
            k = head.key(x)
            v = head.value(x)
            wei = (q @ k.transpose(-2,-1)) * (1.0) # head_size=1 so scale=1.   
            wei = wei.masked_fill(block.sa.heads[0].tril[:T,:T]==0, float('-inf'))
            wei = F.softmax(wei, dim=-1)

            print(f"\n-- Head {i} --")
            print(f"q[{token_pos}] = {q[0,token_pos,0].item():.4f}")
            print(f"k = {k[0,:,0].cpu().numpy()}")
            print("wei = ", wei)
            print(f"wei[{token_pos}] (to all previous) = {wei[0,token_pos].cpu().numpy()}")
            print(f"v = {v[0,:,0].cpu().numpy()}")
            mul = wei @ v
            print(" mul ", mul)
            out = (wei @ v)[0,token_pos,0].item()
            print(f"head output = {out:.4f}")
