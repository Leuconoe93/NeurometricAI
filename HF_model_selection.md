**On finding models on HuggingFace yourself**
This is a great skill to build. Here is exactly where to look:

**Step 1 — Go to huggingface.co/models**
Filter by:
- Task: *Image-Text-to-Text* (this is the correct task for VLMs)
- Library: *Transformers*
- Sort by: *Most Downloads* or *Trending*
This gives you the most-used VLMs with native transformers support.

**Step 2 — On a model page, check three things:**
1. The *pipeline_tag* in the model card header — should say image-text-to-text for VLMs
2. The *library_name* in the metadata — should say *transformers* not *custom* or *llama.cpp*
3. The README loading code — if it uses *AutoModelForImageTextToText* or *AutoModel* without *trust_remote_code=True*, it will work cleanly with our pipeline. If it requires *trust_remote_code=True*, it uses custom code and may have the *all_tied_weights_keys* problem.

**Step 3 — Check the config.json**
Click "Files and versions" on any model page, open *config.json*. Look for "model_type". If the model type appears in the transformers *AutoModelForImageTextToText* supported list, it will load without issues. Common safe types: qwen2_vl, smolvlm, mllama, gemma3, idefics3, internvl (only in recent transformers).

**Step 4 — For quantized models**
Search for the base model name plus AWQ or GPTQ. Check who uploaded it — official org uploads (e.g. Qwen/, google/) are more reliable than community uploads. Look for *quantization_config* in *config.json* — if it says "quant_type": "awq" you need autoawq, if it says "quant_type": "gptq" you need auto-gptq.
