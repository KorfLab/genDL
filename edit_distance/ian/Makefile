######################
# Makefile for edist #
######################

APP1 = edist
SRC1 = edist.c

CFLAGS = -O3 -march=native -Wall -Werror -march=native

###########
# Targets #
###########

$(APP1): $(SRC1) $(LIBS)
	gcc $(CFLAGS) -o $(APP1) $(SRC1) -lm -lpthread

clean:
	rm -f *.o $(APP1)

