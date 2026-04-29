My beginnings to understand the world of LLMs and trying to build, pretrain and finetune some of my own as well. 
Based on https://www.youtube.com/watch?v=kCc8FmEb1nY



Trying to walk through the whole lifecycle of a token (Multiple Attention heads working in parallel) 
<img width="1002" height="105" alt="image" src="https://github.com/user-attachments/assets/11e79091-cebb-4416-86d6-c0dbae20699a" />


Weights matrix (After k.q) for the first token of a batch is always 1 (after normalization) because there is no token behind it and it can only attent to itself (in decoder infra)

<img width="1188" height="163" alt="image" src="https://github.com/user-attachments/assets/ec4d07ad-6be8-45db-9219-2912e20f1413" />

