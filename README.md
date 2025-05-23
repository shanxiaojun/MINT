# MINT
## Quick Start
We fine-tuning Qwen2-VL with LLaMA-Factory. 
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train config/qwen_vl_Reduandancy.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train config/qwen_vl_Synergy.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train config/qwen_vl_Uniqueness.yaml
```

Configure your environment based on the instruction below.
## Provided Datasets



Some datasets require confirmation before using them, so we recommend logging in with your Hugging Face account using these commands.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## Requirement

| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.8     | 3.11      |
| torch        | 1.13.1  | 2.4.0     |
| transformers | 4.41.2  | 4.43.4    |
| datasets     | 2.16.0  | 2.20.0    |
| accelerate   | 0.30.1  | 0.32.0    |
| peft         | 0.11.1  | 0.12.0    |
| trl          | 0.8.6   | 0.9.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.14.0    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.5.0     |
| flash-attn   | 2.3.0   | 2.6.3     |

### Hardware Requirement

\* *estimated*

| Method            | Bits |   7B  |  13B  |  30B  |   70B  |  110B  |  8x7B |  8x22B |
| ----------------- | ---- | ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| Full              | AMP  | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| Full              |  16  |  60GB | 120GB | 300GB |  600GB |  900GB | 400GB | 1200GB |
| Freeze            |  16  |  20GB |  40GB |  80GB |  200GB |  360GB | 160GB |  400GB |
| LoRA/GaLore/BAdam |  16  |  16GB |  32GB |  64GB |  160GB |  240GB | 120GB |  320GB |
| QLoRA             |   8  |  10GB |  20GB |  40GB |   80GB |  140GB |  60GB |  160GB |
| QLoRA             |   4  |   6GB |  12GB |  24GB |   48GB |   72GB |  30GB |   96GB |
| QLoRA             |   2  |   4GB |   8GB |  16GB |   24GB |   48GB |  18GB |   48GB |

## Getting Started

### Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
cd MINT
pip install -e ".[torch,metrics]"
```

Extra dependencies available: torch, torch-npu, metrics, deepspeed, liger-kernel, bitsandbytes, hqq, eetq, gptq, awq, aqlm, vllm, galore, badam, adam-mini, qwen, modelscope, openmind, quality

> [!TIP]
> Use `pip install --no-deps -e .` to resolve package conflicts.

<details><summary>For Windows users</summary>

If you want to enable the quantized LoRA (QLoRA) on the Windows platform, you need to install a pre-built version of `bitsandbytes` library, which supports CUDA 11.1 to 12.2, please select the appropriate [release version](https://github.com/jllllll/bitsandbytes-windows-webui/releases/tag/wheels) based on your CUDA version.

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```

To enable FlashAttention-2 on the Windows platform, you need to install the precompiled `flash-attn` library, which supports CUDA 12.1 to 12.2. Please download the corresponding version from [flash-attention](https://github.com/bdashore3/flash-attention/releases) based on your requirements.

</details>

<details><summary>For Ascend NPU users</summary>

To install LLaMA Factory on Ascend NPU devices, please specify extra dependencies: `pip install -e ".[torch-npu,metrics]"`. Additionally, you need to install the **[Ascend CANN Toolkit and Kernels](https://www.hiascend.com/developer/download/community/result?module=cann)**. Please follow the [installation tutorial](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/softwareinstall/instg/atlasdeploy_03_0031.html) or use the following commands:

```bash
# replace the url according to your CANN version and devices
# install CANN Toolkit
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC701/Ascend-cann-toolkit_8.0.RC1.alpha001_linux-"$(uname -i)".run
bash Ascend-cann-toolkit_8.0.RC1.alpha001_linux-"$(uname -i)".run --install

# install CANN Kernels
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC701/Ascend-cann-kernels-910b_8.0.RC1.alpha001_linux.run
bash Ascend-cann-kernels-910b_8.0.RC1.alpha001_linux.run --install

# set env variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

| Requirement  | Minimum | Recommend   |
| ------------ | ------- | ----------- |
| CANN         | 8.0.RC1 | 8.0.RC1     |
| torch        | 2.1.0   | 2.1.0       |
| torch-npu    | 2.1.0   | 2.1.0.post3 |
| deepspeed    | 0.13.2  | 0.13.2      |

Remember to use `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES` to specify the device to use.

If you cannot infer model on NPU devices, try setting `do_sample: false` in the configurations.

Download the pre-built Docker images: [32GB](http://mirrors.cn-central-221.ovaijisuan.com/detail/130.html) | [64GB](http://mirrors.cn-central-221.ovaijisuan.com/detail/131.html)

</details>

### Data Preparation

Please refer to [data/README.md](data/README.md) for checking the details about the format of dataset files. You can either use datasets on HuggingFace / ModelScope / Modelers hub or load the dataset in local disk.

> [!NOTE]
> Please update `data/dataset_info.json` to use your custom dataset.


> [!TIP]
> Use `llamafactory-cli help` to show help information.

### Fine-Tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

```bash
llamafactory-cli webui
```

### Build Docker

For CUDA users:

```bash
cd docker/docker-cuda/
docker compose up -d
docker compose exec llamafactory bash
```

For Ascend NPU users:

```bash
cd docker/docker-npu/
docker compose up -d
docker compose exec llamafactory bash
```

For AMD ROCm users:

```bash
cd docker/docker-rocm/
docker compose up -d
docker compose exec llamafactory bash
```

<details><summary>Build without Docker Compose</summary>

For CUDA users:

```bash
docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg INSTALL_BNB=false \
    --build-arg INSTALL_VLLM=false \
    --build-arg INSTALL_DEEPSPEED=false \
    --build-arg INSTALL_FLASHATTN=false \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    -t llamafactory:latest .

docker run -dit --gpus=all \
    -v ./hf_cache:/root/.cache/huggingface \
    -v ./ms_cache:/root/.cache/modelscope \
    -v ./om_cache:/root/.cache/openmind \
    -v ./data:/app/data \
    -v ./output:/app/output \
    -p 7860:7860 \
    -p 8000:8000 \
    --shm-size 16G \
    --name llamafactory \
    llamafactory:latest

docker exec -it llamafactory bash
```

For Ascend NPU users:

```bash
# Choose docker image upon your environment
docker build -f ./docker/docker-npu/Dockerfile \
    --build-arg INSTALL_DEEPSPEED=false \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    -t llamafactory:latest .

# Change `device` upon your resources
docker run -dit \
    -v ./hf_cache:/root/.cache/huggingface \
    -v ./ms_cache:/root/.cache/modelscope \
    -v ./om_cache:/root/.cache/openmind \
    -v ./data:/app/data \
    -v ./output:/app/output \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -p 7860:7860 \
    -p 8000:8000 \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    --shm-size 16G \
    --name llamafactory \
    llamafactory:latest

docker exec -it llamafactory bash
```

For AMD ROCm users:

```bash
docker build -f ./docker/docker-rocm/Dockerfile \
    --build-arg INSTALL_BNB=false \
    --build-arg INSTALL_VLLM=false \
    --build-arg INSTALL_DEEPSPEED=false \
    --build-arg INSTALL_FLASHATTN=false \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    -t llamafactory:latest .

docker run -dit \
    -v ./hf_cache:/root/.cache/huggingface \
    -v ./ms_cache:/root/.cache/modelscope \
    -v ./om_cache:/root/.cache/openmind \
    -v ./data:/app/data \
    -v ./output:/app/output \
    -v ./saves:/app/saves \
    -p 7860:7860 \
    -p 8000:8000 \
    --device /dev/kfd \
    --device /dev/dri \
    --shm-size 16G \
    --name llamafactory \
    llamafactory:latest

docker exec -it llamafactory bash
```

</details>

<details><summary>Details about volume</summary>

- `hf_cache`: Utilize Hugging Face cache on the host machine. Reassignable if a cache already exists in a different directory.
- `ms_cache`: Similar to Hugging Face cache but for ModelScope users.
- `om_cache`: Similar to Hugging Face cache but for Modelers users.
- `data`: Place datasets on this dir of the host machine so that they can be selected on LLaMA Board GUI.
- `output`: Set export dir to this location so that the merged result can be accessed directly on the host machine.

</details>

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml
```

> [!TIP]
> Visit [this page](https://platform.openai.com/docs/api-reference/chat/create) for API document.
>
> Examples: [Image understanding](scripts/test_image.py) | [Function calling](scripts/test_toolcall.py)

### Download from ModelScope Hub

If you have trouble with downloading models and datasets from Hugging Face, you can use ModelScope.

```bash
export USE_MODELSCOPE_HUB=1 # `set USE_MODELSCOPE_HUB=1` for Windows
```

Train the model by specifying a model ID of the ModelScope Hub as the `model_name_or_path`. You can find a full list of model IDs at [ModelScope Hub](https://modelscope.cn/models), e.g., `LLM-Research/Meta-Llama-3-8B-Instruct`.

### Download from Modelers Hub

You can also use Modelers Hub to download models and datasets.

```bash
export USE_OPENMIND_HUB=1 # `set USE_OPENMIND_HUB=1` for Windows
```

Train the model by specifying a model ID of the Modelers Hub as the `model_name_or_path`. You can find a full list of model IDs at [Modelers Hub](https://modelers.cn/models), e.g., `TeleAI/TeleChat-7B-pt`.

### Use W&B Logger

To use [Weights & Biases](https://wandb.ai) for logging experimental results, you need to add the following arguments to yaml files.

```yaml
report_to: wandb
run_name: test_run # optional
```

Set `WANDB_API_KEY` to [your key](https://wandb.ai/authorize) when launching training tasks to log in with your W&B account.



</details>

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

Please follow the model licenses to use the corresponding model weights: [Qwen](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT) 

## Acknowledgement

This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct). Thanks for their wonderful works.
