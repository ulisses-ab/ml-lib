# Name of the final executable
TARGET = a.out

# Find all .cu and .cpp files inside the src directory
SRCS = $(shell find -name "*.cu" -o -name "*.cpp")

# The compiler
CC = nvcc

# C++23 Flag
# --std=c++23 sets the standard for both device and host code
CFLAGS = -arch=sm_89 -std=c++17


# If you get "cannot find -lbacktrace", you may need to install it:
# sudo apt install libbacktrace-dev

# Default rule
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

# Clean up the executable
clean:
	rm -f $(TARGET)