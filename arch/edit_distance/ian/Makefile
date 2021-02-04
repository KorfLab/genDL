######################
# Makefile for edist #
######################

APP1 = edist
SRC1 = edist.c

APP2 = cmp2
SRC2 = cmp2.c

CFLAGS = -O3 -march=native
WFLAGS = -Wall -Werror

###########
# Targets #
###########

default:
	make $(APP1)
	make $(APP2)

$(APP1): $(SRC1) $(LIBS)
	gcc $(CFLAGS) $(WFLAGS) -o $(APP1) $(SRC1) -lm -lpthread

$(APP2): $(SRC2) $(LIBS)
	gcc $(CFLAGS) $(WFLAGS) -o $(APP2) $(SRC2) -lm -lpthread

clean:
	rm -f *.o $(APP1) $(APP2)

