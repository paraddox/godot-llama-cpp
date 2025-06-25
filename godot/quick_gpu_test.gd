extends Node

func _ready():
	print("🧪 Quick GPU Backend Test")
	
	# This will trigger the backend initialization logs
	var context = LlamaContext.new()
	
	print("✅ Test complete - check console output above for backend detection")
	
	# Exit after test
	get_tree().quit()