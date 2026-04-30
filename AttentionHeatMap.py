print('We are testing the Jane is a girl part now')
test = "Jane is a girl. She"
idx = torch.tensor([encode(test)], device=device)

with torch.no_grad():
    _ = model(idx)

attn = model.blocks[3].sa.heads[3].last_wei[0].cpu().numpy()

import matplotlib.pyplot as plt
plt.imshow(attn, cmap='Blues')
plt.xticks(range(len(test)), list(test), rotation=90)
plt.yticks(range(len(test)), list(test))
plt.show()