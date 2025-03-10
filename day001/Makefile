# CUDA Makefile for Day 1 Vector Addition

# CUDA compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -O3 -arch=sm_50

# For Jetson Nano, use sm_53 architecture
# Uncomment the line below when compiling on Jetson Nano
# NVCC_FLAGS := -O3 -arch=sm_53

# Target executable
TARGET := vector_add

# Source files
SRC := vector_add.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
