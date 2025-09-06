# git update-index --skip-worktree Makefile
# git update-index --no-skip-worktree Makefile

ASANCOPT = -fsanitize=address -fno-omit-frame-pointer
ASANLDOPT = -fsanitize=address -Wl,-Bdynamic -lasan

# default settings
OPT=-g #$(ASANCOPT)

# settings from 'configure', may overwrite defaults
-include ../Makefile.config

EXAMPLES = min vsum vsum2 spmv spmv2 \
    jac1d jac1d-ser jac2d jac2d-ser jac2d-lb jac3d jac3d-lb test \
    lb markov-ser markov markov2 md md-lb md-ser md3d md3d-ser \
    propagation1d propagation2d \
    resize vsum3 \
    ping_pong \
    README-example

LDFLAGS = $(OPT) #$(ASANLDOPT)
CFLAGS = $(OPT) $(WARN) $(DEFS) -std=gnu99 -I$(SDIR)../include
LAIKLIB = $(abspath ../liblaik.so)

all: $(EXAMPLES)

%.o: $(SDIR)%.c
	$(CC) -c $(CFLAGS) -c $< -o $@

min: min.o $(LAIKLIB)

vsum: vsum.o $(LAIKLIB)

vsum2: vsum2.o $(LAIKLIB)

spmv: spmv.o $(LAIKLIB)

spmv2: $(SDIR)spmv2.c $(LAIKLIB)
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< $(LAIKLIB) -o $@

jac1d: jac1d.o $(LAIKLIB)

jac1d-ser: jac1d-ser.o $(LAIKLIB)

jac2d: jac2d.o $(LAIKLIB)

jac2d-lb: jac2d-lb.o -lm $(LAIKLIB)

jac3d: jac3d.o $(LAIKLIB)

jac3d-lb: jac3d-lb.o -lm $(LAIKLIB)

lb: lb.o -lm $(LAIKLIB)

test: test.o -lm $(LAIKLIB)

markov: markov.o $(LAIKLIB)

markov2: markov2.o $(LAIKLIB)

md: md.o -lm $(LAIKLIB)

md-lb: md-lb.o -lm $(LAIKLIB)

md-ser: md-ser.o -lm $(LAIKLIB)

md3d: md3d.o -lm $(LAIKLIB)

md3d-ser: md3d-ser.o -lm $(LAIKLIB)

propagation1d: $(SDIR)propagation1d.c $(LAIKLIB)
	$(CC) $(CFLAGS) $< $(LAIKLIB) -o $@ -lm

propagation2d: $(SDIR)propagation2d.c $(LAIKLIB)
	$(CC) $(CFLAGS) $< $(LAIKLIB) -o $@ -lm

resize: resize.o $(LAIKLIB)

vsum3: vsum3.o $(LAIKLIB)

README-example: README-example.o $(LAIKLIB)

ping_pong: ping_pong.o $(LAIKLIB)

clean:
	rm -f *.o *~ *.ppm $(EXAMPLES)
