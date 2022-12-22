# indicate how the object files are to be created
CC         := gcc 
CFLAGS     := -g -O3
# CFLAGS     := -g -O3 -axCORE-AVX2  -qopenmp 

OBJECT_FILES := driver.o helper.o gemy.o
saxpy: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) -o saxpy
clean:
	rm *.o 