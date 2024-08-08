import os
import bucho

print(f"bucho package location: {os.path.dirname(bucho.__file__)}")
print(f"Current working directory: {os.getcwd()}")

spinners_path = os.path.join(os.path.dirname(bucho.__file__), "spinners.json")
print(f"Expected spinners.json path: {spinners_path}")
print(f"spinners.json exists: {os.path.exists(spinners_path)}")

gui_textual_path = os.path.join(os.path.dirname(bucho.__file__), "gui_textual.py")
print(f"gui_textual.py path: {gui_textual_path}")
print(f"gui_textual.py exists: {os.path.exists(gui_textual_path)}")

# List contents of the bucho package directory
print("\nContents of bucho package directory:")
for item in os.listdir(os.path.dirname(bucho.__file__)):
    print(item)
