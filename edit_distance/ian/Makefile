######################
# Makefile for edist #
######################

APP1 = edist
SRC1 = edist.c

CFLAGS = -O3 -march=native
WFLAGS = -Wall -Werror

###########
# Targets #
###########

$(APP1): $(SRC1) $(LIBS)
	gcc $(CFLAGS) $(WFLAGS) -o $(APP1) $(SRC1) -lm -lpthread

clean:
	rm -f *.o $(APP1)

