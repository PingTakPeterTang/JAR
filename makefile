CC=gcc
CFLAGS=-I.
DEPS = jar_sim.h jar_type.h jar_utils.h 
OBJ = demo.o jar_utils.o jar_sim.o 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

demo: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

