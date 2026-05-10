# UnHateMemeDL

Mitigate hateful content on image meme using open source Vision-Language Models (VLMs) and Diffusion Models.

## Build the Docker image

Install Docker on your local machine if you haven't already. Then, go in the docker/ directory. If you want to add dependecies, you can modify the `requirements.txt` file and rebuild the image.

Once done, you can build the image using the following command:

```bash
docker build --platform linux/amd64 . --tag registry.rcp.epfl.ch/ee-559-elboudir/my-toolbox:v{VERSION}
```

Then, push the image to the registry:

```bash
docker push registry.rcp.epfl.ch/ee-559-elboudir/my-toolbox:v{VERSION}
```


## Connect to the cluster and run scripts

To connect to the cluster, you can use the following command:

```bash
ssh {EPFL_USERNAME}@jumphost.rcp.epfl.ch:/mnt/course-ee-559/rcp-caas-ee-559-g09/scratch-g09
```

Once connected, go to the group directory:

```bash
cd /mnt/course-ee-559/rcp-caas-ee-559-g09/scratch-g09/UnHateMemeDL
```

Then, you to be able to run the scripts, lets request a node with GPU:

- For interactive session:

```bash
runai submit \
  --name unhatememe \
  --run-as-uid {uid} \
  --image registry.rcp.epfl.ch/ee-559-elboudir/my-toolbox:v{VERSION} \
  --gpu 1 \
  --node-pools default \
  --existing-pvc claimname=course-ee-559-scratch-g09,path=/scratch-g09 \
  --existing-pvc claimname=home,path=/home/{EPFL_USERNAME} \
  --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
  --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
  --interactive --attach
```

Note: replace `{uid}` with your user id, which you can get using the `id -u` command. {VERSION} should be replaced with the version of the image you built and pushed to the registry (if you didn't build and push the image, you can use the `1.0` tag). {EPFL_USERNAME} should be replaced with your EPFL username.

- For running a script inside the container, example for inference:

```bash
python3 UnHateMemeDL/src/unhate_pipeline/main.py \
  --vlm_name {VLM_NAME} \
  --data_path {DATA_PATH} \
  --cache_dir {HF_CACHE_DIR}
```

Note: replace `{VLM_NAME}` with the name of the Vision-Language Model you want to use (e.g., "Qwen/Qwen3.6-27B") should be correspond to the model name in the Hugging Face Hub. `{DATA_PATH}` should be replaced with the path to the dataset you want to use for inference. `{HF_CACHE_DIR}` should be replaced with the path to the Hugging Face cache directory where the models will be downloaded and stored.
