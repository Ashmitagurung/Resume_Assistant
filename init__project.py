# Directory structure initialization
# Create all necessary directories and empty __init__.py files

import os

# Define the directory structure
directories = [
    "src",
    "src/document_processor",
    "src/embeddings",
    "src/retrieval",
    "src/utils",
    "src/interface",
    "data",
    "data/temp",
    ".streamlit"
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)

    # Create __init__.py files in Python module directories
    if directory.startswith("src"):
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                # Add module docstring
                module_name = directory.replace("/", ".").split(".", 1)[-1] if "." in directory.replace("/",
                                                                                                        ".") else directory
                f.write(f'"""{module_name} module for Resume Assistant"""\n')

# Create .streamlit/secrets.toml file with example content
secrets_file = os.path.join(".streamlit", "secrets.toml")
if not os.path.exists(secrets_file):
    with open(secrets_file, "w") as f:
        f.write("""# Secrets configuration file
# Replace with your actual API keys

[groq]
api_key = "your_groq_api_key_here"
""")

print("Project structure initialized successfully!")
print("Remember to add your actual Groq API key to .streamlit/secrets.toml")