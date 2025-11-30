from huggingface_hub import login, HfApi

login(token="")

api = HfApi()

api.upload_folder(
    folder_path="my_checkpoints",
    repo_id="TuanVu219/Vit_Checkpoint_New",
    repo_type="model"
)

print("Upload nhiều file thành công!")
