##############################
# Makefile for uniqueometer  #
##############################

APP1 = edist
SRC1 = edist.c

CFLAGS = -O2 -Wall -Werror

###########
# Targets #
###########

$(APP1): $(SRC1) $(LIBS)
	gcc -Wall -Werror -o $(APP1) $(SRC1) $(IK) -lm -O2

clean:
	rm -f *.o $(APP1)

