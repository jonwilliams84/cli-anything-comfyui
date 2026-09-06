from setuptools import setup, find_namespace_packages

with open("cli_anything/comfyui/README.md") as f:
    long_description = f.read()

setup(
    name="cli-anything-comfyui",
    version="0.1.0",
    description="Drive a running ComfyUI from the shell: workflows, nodes, queue, renders, outputs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jon Williams",
    url="https://github.com/jonwilliams84/cli-anything-comfyui",
    license="MIT",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    install_requires=[
        "click>=8.0.0",
        "prompt-toolkit>=3.0.0",
    ],
    extras_require={"test": ["pytest>=7.0"]},
    entry_points={
        "console_scripts": [
            "cli-anything-comfyui=cli_anything.comfyui.comfyui_cli:main",
        ],
    },
    package_data={
        "cli_anything.comfyui": ["skills/*.md", "data/*.json"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Environment :: Console",
    ],
)
