extends Node

func _ready():
	print("🔍 Quick GPU test starting...")
	
	var model = LlamaModel.new()
	model.set_path("godot/models/Phi-3-mini-4k-instruct-q4.gguf")
	
	print("Loading model...")
	model.load_model()
	
	if model.is_loaded():
		print("✅ Model loaded successfully")
		print("GPU layers:", model.get_n_gpu_layers())
		
		var context = LlamaContext.new()
		context.set_model(model)
		context.set_n_ctx(512)
		context.set_temperature(0.8)
		
		print("Initializing context...")
		context.initialize_context()
		
		print("Requesting completion...")
		context.request_completion("Hello")
		
		# Wait for completion
		await context.completion_generated
		
		print("✅ GPU test completed")
	else:
		print("❌ Model failed to load")
	
	get_tree().quit()