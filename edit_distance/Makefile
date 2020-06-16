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
	gcc -O2 -Wall -Werror -o $(APP1) $(SRC1) $(IK) -lm

clean:
	rm -f *.o $(APP1)

