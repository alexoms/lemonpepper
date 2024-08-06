Introduce whisper.cpp real-time streaming into the application
./stream -m ./models/ggml-base.en.bin -t 8 --step 1000 --length 10000


#for whisper.cpp build of the dylib libraries under arm64
# using Makefile
make clean
WHISPER_COREML=1 make -j

# using CMake
cmake -B build -DWHISPER_COREML=1 -D CMAKE_OSX_ARCHITECTURES="arm64"
cmake --build build -j --config Release