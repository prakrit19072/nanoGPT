My beginnings to understand the world of LLMs and trying to build, pretrain and finetune some of my own as well. 
Based on https://www.youtube.com/watch?v=kCc8FmEb1nY



Trying to walk through the whole lifecycle of a token (Multiple Attention heads working in parallel) 
<img width="1002" height="105" alt="image" src="https://github.com/user-attachments/assets/11e79091-cebb-4416-86d6-c0dbae20699a" />


Weights matrix (After k.q) for the first token of a batch is always 1 (after normalization) because there is no token behind it and it can only attent to itself (in decoder infra)

<img width="1188" height="163" alt="image" src="https://github.com/user-attachments/assets/ec4d07ad-6be8-45db-9219-2912e20f1413" />



Mapping the attention Heads initially on a small model, I saw very less attention values being given to other tokens
<img width="556" height="446" alt="Screenshot 2026-04-30 at 11 42 32 AM" src="https://github.com/user-attachments/assets/eac336d0-d597-414f-acb0-c2e507795b2d" />

Increasing the Embedding layers, number of iterations and the context size (block size), the heat matrix started getting better . I see things like Jane Attentding to girl and older values

<img width="450" height="428" alt="Screenshot 2026-04-30 at 12 11 52 PM" src="https://github.com/user-attachments/assets/3b0ff023-57a8-475b-b562-260df860332f" />


