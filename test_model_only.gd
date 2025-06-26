extends Node

func _ready():
	print("=== MODEL-ONLY GPU TEST ===")
	
	# Wait a moment for extension to fully initialize
	await get_tree().process_frame
	
	var llama_context = LlamaContext.new()
	add_child(llama_context)
	
	# Load model manually without UI dependencies
	if llama_context.model == null:
		print("Creating LlamaModel...")
		var llama_model = LlamaModel.new()
		llama_model.set_path("res://models/Phi-3-mini-4k-instruct-q4.gguf")
		
		print("Loading model...")
		llama_model.load_model()
		
		if llama_model.is_loaded():
			print("✅ Model loaded successfully")
			print("GPU layers:", llama_model.get_n_gpu_layers())
			
			llama_context.model = llama_model
			llama_context.initialize_context()
			print("✅ Context initialized")
			
			# Test a simple completion
			print("Requesting completion...")
			llama_context.request_completion("Hello")
		else:
			print("❌ Model failed to load")
	
	# Quit after a delay to see output
	await get_tree().create_timer(2.0).timeout
	get_tree().quit()