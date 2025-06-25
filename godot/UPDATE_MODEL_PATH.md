# Update Model Path for Phi-3

To update the example to use the Phi-3 Mini model:

1. **In Godot Editor:**
   - Open `examples/simple/simple.tscn`
   - Select the `LlamaContext` node
   - In the Inspector, click the "Model" property dropdown
   - Navigate to `res://models/Phi-3-mini-4k-instruct-q4.gguf`
   - Assign the new model

2. **Model Details:**
   - File: `Phi-3-mini-4k-instruct-q4.gguf` (2.3GB)
   - Format: Phi-3 chat template (already updated in simple.gd)
   - Context: 4k tokens

3. **Chat Format:**
   - Updated from "llama3" to "phi3" in simple.gd:18
   - Uses `<|user|>`, `<|assistant|>`, `<|system|>` format

Save the scene after updating the model path.