# Compiler
CC = gcc

# Compiler flags from CMakeLists.txt
# -Ofast: Aggressive optimization
# -flto: Link-Time Optimization
# -mavx2 -mfma -mpopcnt: Required instruction sets
# -Wall: Enable all warnings
CFLAGS = -Ofast -flto -mavx2 -mfma -mpopcnt -Wall

# Linker flags
# -lpthread: Link with the POSIX threads library
LDFLAGS = -lpthread

# Compile-time definitions from CMakeLists.txt options
# USE_POPCNT_PATH is ON by default
# PIN_THREADS is OFF by default
DEFS = -DUSE_POPCNT_PATH

# Executable name
TARGET = pi.out

# Source file
# Note: CMakeLists.txt says "pi.c", but the directory contains "pi_pthread.c"
SRCS = pi.c

# Object file derived from source
OBJS = $(SRCS:.c=.o)

# Default rule: build the executable
all: $(TARGET)

# Rule to link the executable
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Rule to compile C source files into object files
%.o: %.c
	$(CC) $(CFLAGS) $(DEFS) -c $< -o $@

# Phony target for cleaning up build files
.PHONY: all clean

# Rule to clean the directory
clean:
	rm -f $(OBJS) $(TARGET)

