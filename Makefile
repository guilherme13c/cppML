# Makefile for building the program and library

# Directories
BIN_DIR := bin
INCLUDE_DIR := inc
SRC_DIR := src

# Compiler settings
CC := clang++
CFLAGS := -Wall -Wextra -pedantic -std=c++20 -fcolor-diagnostics
INC_FLAGS := -I$(INCLUDE_DIR)

# Source file and output executable
SOURCE := $(SRC_DIR)/main.cpp
OUTPUT := $(BIN_DIR)/main

# Operating system check
ifeq ($(OS),Windows_NT)
	EXECUTABLE := $(OUTPUT).exe
else
	EXECUTABLE := $(OUTPUT)
endif

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCE)
	$(CC) $(CFLAGS) $(INC_FLAGS) $(SOURCE) -o $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
