INCLUDE = $(shell pkg-config --cflags opencv) -I /usr/include/eigen3/
LIBS = $(shell pkg-config --libs opencv)
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = camera_calibration

$(TARGET):$(OBJECTS)
	g++ -o $(TARGET) $(OBJECTS) $(INCLUDE) $(LIBS)
$(OBJECTS):$(SOURCES)
	g++ -c $(SOURCES)

clean:
	rm $(OBJECTS) $(TARGET)