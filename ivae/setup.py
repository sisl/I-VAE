from setuptools import setup, find_packages

setup(
    name="ivae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch", "tqdm"],
    author="Robert Moss",
    author_email="mossr@cs.stanford.edu",
    description="Inversion variational autoencoder (I-VAE) PyTorch implementation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sisl/I-VAE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
