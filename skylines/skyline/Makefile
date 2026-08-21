CC ?= gcc
CFLAGS ?= -O3 -fPIC -std=c11 -Wall -Wextra
LDFLAGS ?= -shared
TARGET = libcskyline.so
SOURCES = c_kernels.c c_kernels_bbs.c
HEADERS = c_kernels.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SOURCES)

clean:
	rm -f $(TARGET)
