from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements file if it exists, otherwise use an empty list
requirements = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = fh.read().splitlines()

# Add appdirs to the requirements
requirements.append('appdirs')

setup(
    name="lemonpepper",
    version="0.1.7",
    author="Unidatum Integrated Products LLC",
    author_email="alex.chang@unidatum.com",
    description="A real-time audio transcription and AI interaction tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexoms/lemonpepper",
    packages=find_packages(include=['lemonpepper', 'lemonpepper.*']),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "lemonpepper=lemonpepper.gui_textual:main",
        ],
    },
    include_package_data=True,
    package_data={
        "lemonpepper": ["spinners.json", "gui_textual.tcss"],
    },
)