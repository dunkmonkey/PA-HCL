"""
PA-HCL 软件包的安装脚本。
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pa-hcl",
    version="0.1.0",
    author="PA-HCL Team",
    description="心音生理感知分层对比学习",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dunkmonkey/PA-HCL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "librosa>=0.10.1",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "all": [
            "mamba-ssm>=1.2.0",
            "accelerate>=0.25.0",
            "wandb>=0.15.0",
            "tensorboard>=2.14.0",
        ],
    },
)
