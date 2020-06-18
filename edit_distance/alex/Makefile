##############################
# Makefile for uniqueometer  #
##############################

APP1 = edist
SRC1 = edist.c

###########
# Targets #
###########

$(APP1): $(SRC1) $(LIBS)
	gcc -Wall -Werror -o $(APP1) $(SRC1) $(IK) -lm -O3 -fopenmp -funroll-loops -march=native

clean:
	rm -f *.o $(APP1)

