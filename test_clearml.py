import os
from clearml import Task

# Check if ClearML config file is being used
print("CLEARML_CONFIG_FILE:", os.environ.get("CLEARML_CONFIG_FILE"))

# Try initializing a dummy ClearML task
try:
    task = Task.init(project_name="Test Project", task_name="Test Task")
    print("ClearML Task Initialized Successfully!")
except Exception as e:
    print("Error:", e)
