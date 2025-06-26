extends SceneTree

func _initialize():
	print("=== MINIMAL TEST START ===")
	
	# Simple model loading test
	var model = LlamaModel.new()
	print("Model created")
	
	model.set_path("godot/models/Phi-3-mini-4k-instruct-q4.gguf")
	print("Path set")
	
	print("About to call load_model()...")
	model.load_model()
	print("load_model() completed")
	
	if model.is_loaded():
		print("✅ Model loaded successfully")
		print("GPU layers:", model.get_n_gpu_layers())
	else:
		print("❌ Model failed to load")
	
	print("=== MINIMAL TEST END ===")
	quit()