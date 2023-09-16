# LlamaEngineer.py
A script that I was able to use in order to run Phind's CodeLlama.

This script was made to see if I could run CodeLlama on my computer with my single GPU the GeForce 1660.
The CodeLlama I chose was Phind/Phind-CodeLlama-34B-v2 from HuggingFace, and the way that I was able to accomplish running this was
that I used 4bit quantization, and some slight code editing.

Because my GPU isn't all that powerful it does take a bit to load it but I'm looking into some ways to increase inference speed, so if I come up with anything
worthwhile I may end up adding to it.

The modeling_llama.py is located in the transformers\models\llama and is called modeling_llama.py. There's only one slight change so you can either copy and paste this or just do the change yourself. The function is def apply_rotary_pos_emb(q, k, cos, sin, position_ids): and I changed this part: cos = cos.squeeze(1).squeeze(0).to("cuda")  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0).to("cuda")  # [seq_len, dim]
That's all it took to run it.
