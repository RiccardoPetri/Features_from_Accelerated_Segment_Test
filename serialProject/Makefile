#CC and CFLAGS are special constants that communicate to make how we want to compile the file
# the compiler: 

CC=g++ #for C++ is g++

# compiler flags:
CFLAGS=-g -Wall -std=c++14 #  -g     - this flag adds debugging information to the executable file
 			   #  -Wall  - this flag is used to turn on most compiler warnings
			   #  -std   - standard used for C++, it includes features for the language core and extends the standard library
SRCS=\
main.cpp\
circleHandler.cpp\
cornerDetection.cpp

OBJS=$(SRCS:.cpp=.o) #places the object files .o in an OBJS subdirectory within the src directory
DEPS=$(SRCS:.cpp=.d) #.d files included in the DEPS macro 
#(.d -> For C/C++ source files they determine what #include files are required and automatically generate that information into the .d file)  

#The -I. is included so that g++ will look in the current directory
INCS=-I. `pkg-config --cflags opencv`#pkg-config is a computer program that defines and supports a unified interface for querying installed 
				     #libraries for the purpose of compiling software that depends on them.
LIBS=`pkg-config --libs opencv` # It also has a macro defined for the library

RM=rm -rf #variable that contain the first part for cleaning up

#It turns out these are special constants that communicate to make how we want to compile the file cornerDetection.cpp
cornerDetection: $(OBJS)
	$(CC) $(OBJS) -o $@ $(CFLAGS) $(INCS) $(LIBS)

#The rule says that the .o file depends upon the .cpp version of the file
#The rule then says that to generate the .o file, make needs to compile the .cpp file using the compiler defined in the CC macro. 
#The -c flag says to generate the object file, the -o $@ says to put the output of the compilation in the file named on the left side of the :, the $< is the first item in the dependencies list, and the CFLAGS macro is defined as above.
%.o: %.cpp
	$(CC) $< -o $@ $(CFLAGS) -c $(INCS)

#rule for cleaning up your source and object directories if you type make clean.
clean:
	$(RM) cornerDetection $(OBJS) $(DEPS) #The second target label ‘clean’ removes all the files, with $(RM) for example we have rm -rf
