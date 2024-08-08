from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="bucho",
    version="0.1.0",
    author="Unidatum Integrated Products LLC",
    author_email="alex.chang@unidatum.com",
    description="A real-time audio transcription and AI interaction tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexoms/bucho",
    packages=find_packages(include=['bucho', 'bucho.*']),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "bucho=bucho.gui_textual:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bucho": ["spinners.json"],
    },
)