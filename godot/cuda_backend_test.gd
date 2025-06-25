extends Node

func _ready():
	print("🧪 CUDA Backend Detection Test")
	print("===================================")
	
	# Wait a moment for full initialization
	await get_tree().create_timer(1.0).timeout
	
	print("Creating LlamaContext to trigger backend detection...")
	var context = LlamaContext.new()
	
	# Wait another moment
	await get_tree().create_timer(2.0).timeout
	
	print("✅ Test complete")
	print("Expected output above:")
	print("- 'GGML_USE_CUDA is defined - checking CUDA availability...'")
	print("- 'register_backend: registered backend CUDA'")
	print("- 'Backend registrations available: 2'")
	
	get_tree().quit()