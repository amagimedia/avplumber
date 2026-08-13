# Standalone NVOF dense-CUDA link/init probe.
# Run on a box with the CUDA toolkit + NVIDIA driver (e.g. the T4 dev host).
#
#   make            # build ./nvof_probe
#   make run        # build + run (needs an NVIDIA GPU with an OF engine)

CUDA_HOME ?= /usr/local/cuda
CXX       ?= g++
CXXFLAGS  += -std=c++17 -O2 -I../NvOFInterface -I$(CUDA_HOME)/include
LDFLAGS   += -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs
LDLIBS    += -lcuda -lcudart -lnvidia-opticalflow

nvof_probe: nvof_probe.cpp ../NvOFInterface/nvOpticalFlowCuda.h ../NvOFInterface/nvOpticalFlowCommon.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: run clean
run: nvof_probe
	./nvof_probe

clean:
	rm -f nvof_probe
