# Document setup the i4 repo

### `uv` is the required package management

* Need to go to `dockerrun` and setup inside docker

```
$ dockerrun
$ pip install uv==0.7.3
$ python -m pip install --break-system-packages "uv==0.7.3"
```

* Make uv a true system-wise

```
$ python -m pip install --break-system-packages "uv==0.7.3"
```

* Here need to exit docker environment and re-enter to let `uv` command visible

```
$ exit
$ dockerrun
$ uv --version # check if everything OK
```

### [Now Solution] Clone a docker python environment with uv

* Why do this, because we want to customize install the package, not to the system (docker) python environment, but to our replicated .venv
* To clone a uv venv from docker default path

```
uv venv --python /usr/bin/python --system-site-packages .venv
uv pip install -e . --no-deps
```

### [Earlier Solution] Create a uv .venv from pyproject.toml

* For debugging with `vscode`, one need to build a `.venv/bin/python` (Linux/macOS)
* This is doable via `uv sync` inside the project repo with `pyproject.toml`
* As the package install with `uv` mostly about the packages in `./packages`
* The `uv` also create a `uv.lock` to make enviornment stable.

```
$ cd ~/Project/imaginaire4
$ uv sync
```

* The above `uv sync` only create a default `.venv/bin/python` environment.
* Although we sometimes may need multiple environments by name specification, but this is supported well for `uv`
* To do this, one can:
* But `uv.lock` is shared, meaning `uv` is project centric, so one shall not make different dependency in one project.

```
$ cd ~/Project/imaginaire4
$ uv venv .venv-t2ipipe
$ uv sync --python .venv-t2ipipe/bing/python --lockfile uv-t2ipipe.lock
```

* Remove `uv.lock` then `uv sync` tells `uv` to reinstall/relink packages based on `pyproject.toml`
* By default `uv` has a cache package install so reinstall like `flash-attn` is not happening, it is just relink..


### other packages


* After `uv` setting up the correct `.venv/bin/python`, there are some other package on demand.

```
# For T2I inference pipeline
uv pip install hydra-core
uv pip install fvcore
uv pip install wandb
uv pip install git+https://github.com/NVIDIA/Megatron-LM.git   # megatron-core
uv pip install transformers==4.57.3
uv pip install diffusers==0.35.2
uv pip install peft==0.18.0
uv pip install sentencepiece==0.2.1
uv pip install ftfy
uv pip install ffmpegcv
uv pip install transformer-engine[pytorch]
uv pip install ninja cmake
MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16 MAKEFLAGS="-j16" uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install torchcodec==0.9.1
```

* If launcher (the customized job launching package) is needed (not included in docker)

```
uv pip install -e packages/launcher
```

* Maybe encounter the one_logger issue, fix it with wandb==0.24.0

```
uv pip install --upgrade wandb==0.24.0
```

* If need cosmos-internal-data-utils

```
uv pip install -e packages/cosmos-internal-data-utils --no-deps
uv pip install -e packages/cosmos-xenna --no-deps # super slow, suggest circumvent
```

* If need cosmos-s3-utils

```
uv pip install -e packages/cosmos-s3-utils
uv pip install cattrs
```

* When install `cuda12.4/toolkit/12.4.1`, you may need load module 

```
module list
module avail cuda
module load ***
module avail cudnn
module load ***
```

* If need PaddleOCR

```
uv pip install paddleocr==3.4.0
uv pip install paddlepaddle==3.2.2 # CPU version
uv pip install "paddlepaddle-gpu==3.3.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
uv pip install huggingface-hub==0.36.0
```

* To restore huggingfacehub

```
uv pip install huggingface-hub==0.36.0
```

* If need use Nanobanana

```
uv pip install -U google-genai
```

* If need nltk (nltk installed already, but need to download wordnet)
* If need wordfreq, to sample words based an english default behaviour

```
python -m nltk.downloader -d "/home/xingqianx/Project/imaginaire4/.venv/share/nltk_data" wordnet omw-1.4
uv pip install wordfreq
```

* If need gateway model

```
uv pip install openai
```

* If need to test Flux-2
```
uv pip install -U "diffusers>=0.37.0"
uv pip install huggingface-hub==0.36.0
```

* If need to test GLMImage
```
uv pip install -U "diffusers>=0.37.0"
uv pip install transformers==5.3.0
uv pip install huggingface-hub==1.6.0
```

* If need to process pptx or aspose-slides
```
uv pip install pymupdf python-pptx
pip install aspose-slides
```


### SILA Setup

```
uv sync
uv pip install pylance==1.0.4
uv pip install transformers==4.52.4
```


# Ruff format

* Codelint (old):

```
uv pip install ruff
ruff check projects/cosmos3 --fix
ruff format check projects/cosmos3
```

* Codelint:

**First-time setup (once):**
```bash
uv python install 3.10
uv tool install "pre-commit>=4.3.0"
pre-commit install -c .pre-commit-config-base.yaml
```

**Run lint (equivalent to just lint):**
```bash
pre-commit run -a || pre-commit run -a
(runs twice because some hooks auto-fix on first pass)
```

**Just ruff (fastest):**
```bash
uv run ruff check .
uv run ruff format --check .
```

**Auto-fix everything:**
```bash
pre-commit run -a --hook-stage manual ruff-fix
pre-commit run -a --hook-stage manual ruff-noqa
pre-commit run -a --hook-stage manual ruff-format
```

**Type check:**
```bash
uv run pyrefly check --output-format=min-text
```

# Config for Credential

* i4 repo ask the config (at least the pipeline run needs it)

```
$ vim ~/.config/dir/config.yaml

aws:
    s3_profiles:
    # Update from https://nvidia.slack.com/archives/C0620KYQJEB/p1741111206349849
        s3-training-profile:
            user: "---BNY"
            key: <dont know this>
            endpoint: "https://s3.amazonaws.com"
            region: "us-east-1"
        s3-training:
            user: "---BNY"
            key:
            endpoint: "https://s3.amazonaws.com"
            region: "us-east-1"
        team-cosmos-benchmark:
            user: "team-cosmos-benchmark"
            key:
            endpoint: "https://pdx.s8k.io"
    team-dir:
            user: "team-dir"
            key: <dont know this>
postgres:
    profiles:
        benchmarking_editor_dev:
            user: "benchmarking_editor_dev"
            password: "7QF7jh8cAHC39rcc9qOhK3I2TcAIVCdn"
        benchmarking_editor_prd:
            user: "benchmarking_editor_prd"
            password: "oI1xxsH8XZRZQtDpxRgynGmG3VXsCkVh"
```


### Setup Claude for GCP (the aarch64)

Regularly

```
curl -fsSL https://claude.ai/install.sh | bash
```

Need to first install nvm (without the permission of root)

```
# Install nvm (if you don't have it)
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
# load nvm in the current shell (pick the one that exists)
source ~/.bashrc 2>/dev/null || true
source ~/.zshrc 2>/dev/null || true
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# install and use Node 20 LTS (safe choice)
nvm install 20
nvm use 20
nvm alias default 20

node -v   # should be v20.x
```

NPM install the claude

```
# install cc
npm install -g @anthropic-ai/claude-code
# check version
npm list -g
# update
npm upgrade -g

which claude # check claude

```

You may also need to remove the ~/.vscode-server/extensions/anthropic.claude-code-2.1.34-linux-arm64/resources/native-binary/claude 

```
rm ~/.vscode-server/extensions/anthropic.claude-code-<version-may-change>-linux-arm64/resources/native-binary/claude
ln -s /home/xingqianx/.nvm/versions/node/v20.20.0/bin/claude .
```
and make a fast link to the npm installed claude
