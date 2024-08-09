import os
import appdirs

def get_model_directory():
    # Use appdirs to get the appropriate user data directory
    user_data_dir = appdirs.user_data_dir("lemonpepper", "UnidatumIntegratedProductsLLC")
    model_dir = os.path.join(user_data_dir, "whisper_models")
    
    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    return model_dir