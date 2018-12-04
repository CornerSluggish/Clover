CC := g++ # Ver. >= 7.0
FLAG := -std=c++17 -mavx2 -mfma -mrdrnd -fopenmp -g -O3

SHCOMP :=  -pthread -B /opt/conda/compiler_compat -Wl,--sysroot=/ \
	-DNDEBUG -fwrapv -fPIC -I/opt/conda/include/python3.6m

SHLINK :=  -pthread -shared -B /opt/conda/compiler_compat \
	-L/opt/conda/lib -Wl,-rpath=/opt/conda/lib -Wl,--no-as-needed -Wl,--sysroot=/

MKLROOT := /workspace/intel/compilers_and_libraries/linux/mkl
IPPROOT := /workspace/intel/compilers_and_libraries/linux/ipp

INC := -I/workspace/Clover/include/ \
       -I${IPPROOT}/include/ \
       -I${MKLROOT}/include/
LIB := -L${IPPROOT}/lib/intel64/ \
       -L${MKLROOT}/lib/intel64/ \
       -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm \
       -lippi -lipps -lippcore -lippvm
#       ${IPPROOT}/lib/intel64/libippi.a ${IPPROOT}/lib/intel64/libipps.a ${IPPROOT}/lib/intel64/libippcore.a ${IPPROOT}/lib/intel64/libippvm.a


fully_con.so: fully_con.o
	mkdir -p fully_con
	touch fully_con/__init__.py
	$(CC) $(LIB) $(SHLINK) fully_con.o -o fully_con/fully_con.so
	rm fully_con.o

fully_con.o: clover_fc.cpp
	$(CC) -c $(FLAG) $(SHCOMP) $(INC) $< -o $@

.PHONY: clean
clean:
	rm exe*
