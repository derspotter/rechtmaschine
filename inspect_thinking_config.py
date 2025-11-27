from google.genai import types
import pydantic

try:
    print("ThinkingConfig fields:")
    for field_name, field_info in types.ThinkingConfig.model_fields.items():
        print(f"- {field_name}: {field_info.annotation}")
except Exception as e:
    print(f"Error inspecting ThinkingConfig: {e}")
    # Try dir() if it's not a Pydantic model (though the error suggests it is)
    print("Dir:", dir(types.ThinkingConfig))
