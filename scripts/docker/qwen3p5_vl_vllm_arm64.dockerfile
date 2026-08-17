# Nightly vllm image to support Qwen3_5ForConditionalGeneration architecture.
FROM vllm/vllm-openai:nightly

# Install s5cmd. This is needed to sync code.
# Detect architecture and download appropriate binary for both ARM and x86.
RUN mkdir -p /s5cmd_bin && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
    ARCH_SUFFIX="arm64"; \
    else \
    ARCH_SUFFIX="64bit"; \
    fi && \
    curl -k -L https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-${ARCH_SUFFIX}.tar.gz -o /tmp/s5cmd.tar.gz && \
    tar -xzf /tmp/s5cmd.tar.gz -C /tmp && \
    mv /tmp/s5cmd /s5cmd_bin/s5cmd && \
    chmod +x /s5cmd_bin/s5cmd && \
    rm /tmp/s5cmd.tar.gz

# Install uv. We use this to install packages more quickly than pip.
# We use a specific sha so that this doesn't break our cache.
COPY --from=ghcr.io/astral-sh/uv@sha256:4de5495181a281bc744845b9579acf7b221d6791f99bcc211b9ec13f417c2853 /uv /uvx /bin/
# Needed for qwen-vl-utils preprocessing and robust JSON repair in image captioning.
RUN uv pip install --break-system-packages --system --no-cache-dir qwen-vl-utils==0.0.14 json-repair
# `docker/package_requirements_no_torch.txt` is a file which is updated by the `yotta` cli. It contains all of the
# publically available packages which are needed to run yotta.
COPY docker/package_requirements_no_torch.txt docker/package_requirements.txt
RUN grep -v "^decord" docker/package_requirements.txt > docker/package_requirements_filtered.txt
RUN uv pip install --break-system-packages --system --no-cache-dir -r docker/package_requirements_filtered.txt

# Xenna and internal data utils dependencies are distributed internally. We do this as a seperate command
# so that we can avoid relying on Nvidia's internal package repo as much as possible. It has a tendency to break/go slowly.
COPY docker/internal_package_requirements.txt docker/internal_package_requirements.txt
RUN uv pip install --break-system-packages --system --no-cache-dir -r docker/internal_package_requirements.txt --index-url=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

# This is needed so that we can import the internal data utils and sila packages.
ENV PYTHONPATH=/yotta/code/packages/cosmos-internal-data-utils:/yotta/code/packages/cosmos-sila:/yotta/code/packages/cosmos-data
# This is needed for the vllm image to work. `python` is not available in the vllm image, but `python3` is.
RUN ln -s /usr/bin/python3 /usr/bin/python
# Raise PIL decompression bomb limit so vLLM can decode very large images before resizing.
RUN python3 -c "import PIL.Image; PIL.Image.MAX_IMAGE_PIXELS = 933120000; print(f'Set MAX_IMAGE_PIXELS to {PIL.Image.MAX_IMAGE_PIXELS}')"
RUN SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    echo "import PIL.Image; PIL.Image.MAX_IMAGE_PIXELS = 933120000" >> "$SITE_DIR/sitecustomize.py"
# Reset the entrypoint from the base image so we can run arbitrary commands
# Upgrade transformers last so it isn't overwritten by package_requirements.txt
RUN uv pip install --break-system-packages --system --no-cache-dir jupyterlab

EXPOSE 8888

ENTRYPOINT []
CMD ["/bin/bash"]
