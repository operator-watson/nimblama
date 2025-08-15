# Solo LLM Framework

A personal, homebrew C++ project exploring LLM frameworks using llama.cpp, CUDA, Docker, and CMake. Built as a solo learning project—not intended for public use.

## Overview
This repository is an experimental effort to understand and implement a lightweight LLM framework in C++. Key components:

- llama.cpp – minimal LLaMA model inference
- CUDA – GPU acceleration
- Docker – reproducible development environments
- CMake – build automation

Goal: personal learning, experimenting with model loading, inference pipelines, and optimization.

## Features
- Load and run LLaMA models via C++
- GPU support with CUDA
- Modular CMake build
- Dockerized environment for reproducibility

## Installation
Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```
Build using CMake

```bash
mkdir build
cd build
cmake -S .. -B . -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FORCE_CUBLAS=ON
ninja -v
```
Optional: run with Docker

```bash
docker build -t nimblama .
docker run --gpus all -it nimblama
```
## Usage
Intended as a personal experimentation sandbox. Use the code to load models, run inference, and test learning approaches. No public support or guarantees.

## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).  
See the [full license text](https://www.gnu.org/licenses/agpl-3.0.en.html) for details.
