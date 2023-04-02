import requests
import os
import time
import pickle
from tqdm import tqdm

def downmodel(model_name, model_id, model_dir):
    model_path = os.path.join(model_dir, f'{model_name}')
    if os.path.isfile(model_path):
        print(f'Model {model_name} is already downloaded')
        return

    url = f'https://civitai.com/api/download/models/{model_id}'
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    print(f'Downloading model {model_name}...')
    with open(model_path, 'wb') as f:
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        start_time = time.time()
        prev_size = 0

        for data in response.iter_content(block_size):
            # Update progress bar and write data to file
            progress_bar.update(len(data))
            f.write(data)

            # Check if size has changed in the last 10 seconds
            curr_size = os.path.getsize(model_path)
            if curr_size != prev_size:
                start_time = time.time()
                prev_size = curr_size
            elif time.time() - start_time > 3:
                print(f'Download of {model_name} has stalled. Restarting...')
                f.close()
                os.remove(model_path)
                downmodel(model_name, model_id, model_dir)
                return

        progress_bar.close()

    # Determine the file format based on whether it can be opened as pickle
    model_format = 'pickle' if is_pickle(model_path) else 'safetensor'

    # Add the appropriate file extension based on the file format
    if model_format == 'pickle':
        model_path += '.pickle'
    elif model_format == 'safetensor':
        model_path += '.safetensors'

    # Rename the file with the appropriate extension
    os.rename(os.path.join(model_dir, model_name), model_path)

    print(f'Model {model_name} downloaded successfully')

def is_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            pickle.load(f)
        return True
    except (pickle.UnpicklingError, EOFError):
        return False
model_dir = r'/md126p3/stable-diffusion-webui/models/Stable-diffusion/'  #WAY TO MODELS EXAMPLE
url = 'https://civitai.com/api/v1/models'
params = {
    'limit': 100,
    'sort': 'Highest Rated'
}

response = requests.get(url, params=params, timeout=60*60*24*7)  # set timeout to 1 week

if response.ok:
    models = response.json().get('items')
    for model in models:
        model_name = model.get('name')
        model_id = model.get('id')
        files = model.get('files')
        if model_name:
            print(model_name)
            print(model_id)
            try:
                downmodel(model_name, model_id, model_dir)
            except Exception as e:
                print(f'Error downloading model {model_name}: {e}')
                print('Attempting to download next model...')
                continue
else:
    print(f'Request failed with status code {response.status_code}')
