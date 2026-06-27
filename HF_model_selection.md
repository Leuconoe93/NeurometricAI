**On finding models on HuggingFace yourself**
Here is exactly where to look:

**Step 1 — Go to huggingface.co/models**


Filter by:

* Task: *Image-Text-to-Text* (this is the correct task for VLMs)
* Library: *Transformers*
* Sort by: *Most Downloads* or *Trending*
This gives you the most-used VLMs with native transformers support.
* 

**Step 2 — On a model page, check three things:**

1. The *pipeline\_tag* in the model card header — should say image-text-to-text for VLMs
2. The *library\_name* in the metadata — should say *transformers* not *custom* or *llama.cpp*
3. The README loading code — if it uses *AutoModelForImageTextToText* or *AutoModel* without *trust\_remote\_code=True*, it will work cleanly with our pipeline. If it requires *trust\_remote\_code=True*, it uses custom code and may have the *all\_tied\_weights\_keys* problem.
4. 

**Step 3 — Check the config.json**
Click "Files and versions" on any model page, open *config.json*. Look for "model\_type". If the model type appears in the transformers *AutoModelForImageTextToText* supported list, it will load without issues. Common safe types: qwen2\_vl, smolvlm, mllama, gemma3, idefics3, internvl (only in recent transformers).



**Step 4 — For quantized models**
Search for the base model name plus AWQ or GPTQ. Check who uploaded it — official org uploads (e.g. Qwen/, google/) are more reliable than community uploads. Look for *quantization\_config* in *config.json* — if it says "quant\_type": "awq" you need autoawq, if it says "quant\_type": "gptq" you need auto-gptq.

