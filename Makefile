CC = g++
CFLAGS = -O3 -Wall -Wno-sign-compare -std=c++20 -I/usr/include/linalg -L/usr/lib
LIBRARIES = -llinalg

SOURCES := src/test.cpp
OBJECTS := $(SOURCES:src/%.cpp=obj/%.o)

TEST_EXECUTABLE = bin/test

$(TEST_EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TEST_EXECUTABLE) $(LIBRARIES)

obj/%.o: src/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_EXECUTABLE)
	$(TEST_EXECUTABLE)

clean:
	rm -rf $(OBJECTS) $(TEST_EXECUTABLE)
