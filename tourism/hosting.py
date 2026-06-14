
from huggingface_hub import HfApi
import os

api = HfApi()
api.upload_folder(
    folder_path="/content/tourism",     # the local folder containing your files
    repo_id="Shalmali85/TourismPrediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="/tourism/",                          # optional: subfolder path inside the repo
)
