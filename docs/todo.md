Introduce whisper.cpp real-time streaming into the application
./stream -m ./models/ggml-base.en.bin -t 8 --step 1000 --length 10000

[OPTIONAL] It is recommended to utilize a Python version management system, such as Miniconda for this step:
To create an environment, use: conda create -n py310-whisper python=3.10 -y
To activate the environment, use: conda activate py310-whisper

#for whisper.cpp build of the dylib libraries under arm64
# using Makefile
make clean
WHISPER_COREML=1 make -j

# using CMake
cmake -B build -DWHISPER_COREML=1 -D CMAKE_OSX_ARCHITECTURES="arm64"
cmake --build build -j --config Release