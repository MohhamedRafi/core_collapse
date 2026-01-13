SHELL := /bin/bash

# ---------------- Compiler ----------------
CXX      ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS  ?=

# ---------------- Conda / HDF5 ----------------
CONDA_PREFIX ?= $(shell python -c "import os; print(os.environ.get('CONDA_PREFIX',''))")

INCLUDES := -I$(CONDA_PREFIX)/include
LIBS     := -L$(CONDA_PREFIX)/lib -lhdf5
RPATH    := -Wl,-rpath,$(CONDA_PREFIX)/lib   # macOS runtime fix

# ---------------- Python ----------------
PY      ?= python3
VIZ_PY  := viz_h5.py
VIZ3D_PY := viz_h5_3d.py
GIFDIR  := gifs

# Diagnostics flags for viz_h5.py
GAMMA      ?= 1.4
DIAG       ?= 1
DIAG_TIMES ?= 25
DIAG_NBINS ?= 128
SHOCK_Q    ?= 0.995

VIZ_DIAG_FLAGS :=
ifeq ($(DIAG),1)
VIZ_DIAG_FLAGS += --diag --gamma $(GAMMA) --diag_times $(DIAG_TIMES) --diag_nbins $(DIAG_NBINS) --shock_q $(SHOCK_Q)
endif

# ---------------- Files ----------------
SPH_CPP  := spherical_euler_h5.cpp
SPH_BIN  := spherical_euler_h5
SPH_H5   := out.h5

E2D_CPP  := euler2d_h5.cpp
E2D_HDR  := gravity_enclosed.h
E2D_BIN  := euler2d_h5
E2D_H5   := out2d.h5

E2D_HLLC_CPP := euler2d_gravity_hllc.cpp
E2D_HLLC_BIN := euler2d_gravity_hllc
E2D_HLLC_H5  := out2d_hllc.h5

# 2D Navier–Stokes (HLLC + viscosity + Poisson gravity)
NS2D_CPP := navier_stokes_2d_hllc_poisson.cpp
NS2D_BIN := navier_stokes_2d_hllc_poisson
NS2D_H5  := out_ns2d_poisson.h5

# 2D Resistive MHD (HLLD + CT + Poisson gravity)
MHD2D_CPP := mhd2d_resistive_hlld_ct_gravity.cpp
MHD2D_BIN := mhd2d_resistive_hlld_ct_gravity
MHD2D_H5  := out2d_mhd.h5

# 3D Resistive MHD (HLLD + CT)
MHD3D_CPP := mhd3d_resistive_hlld_ct.cpp
MHD3D_BIN := mhd3d_resistive_hlld_ct
MHD3D_H5  := out3d_mhd.h5

# ---------------- Phony ----------------
.PHONY: all clean info dirs \
        build_sph run_sph viz_sph \
        build_2d run_2d viz_2d \
        build_2d_hllc run_2d_hllc viz_2d_hllc \
        build_ns2d run_ns2d viz_ns2d \
        build_mhd2d run_mhd2d viz_mhd2d \
        build_mhd3d run_mhd3d viz_mhd3d \
        viz_all

# Default: build + run + visualize all
all: viz_all

viz_all: viz_sph viz_2d viz_2d_hllc viz_ns2d viz_mhd2d viz_mhd3d

# ---------------- Info ----------------
info:
	@echo "CONDA_PREFIX=$(CONDA_PREFIX)"
	@$(PY) -c "import h5py, numpy, matplotlib; print('python deps OK')"

dirs:
	@mkdir -p $(GIFDIR)

# ============================
# 1D Spherical Euler
# ============================
build_sph: $(SPH_BIN)

$(SPH_BIN): $(SPH_CPP) Makefile
	$(CXX) $(CXXFLAGS) $(SPH_CPP) -o $(SPH_BIN) \
		$(INCLUDES) $(LIBS) $(RPATH) $(LDFLAGS)

run_sph: build_sph
	./$(SPH_BIN)

viz_sph: run_sph dirs
	@mkdir -p $(GIFDIR)/spherical
	$(PY) $(VIZ_PY) $(SPH_H5) \
		--outdir $(GIFDIR)/spherical \
		--fields /rho,/P,/Mach \
		$(VIZ_DIAG_FLAGS)

# ============================
# 2D Euler (with gravity)
# ============================
build_2d: $(E2D_BIN)

$(E2D_BIN): $(E2D_CPP) $(E2D_HDR) Makefile
	$(CXX) $(CXXFLAGS) $(E2D_CPP) -o $(E2D_BIN) \
		$(INCLUDES) $(LIBS) $(RPATH) $(LDFLAGS)

run_2d: build_2d
	./$(E2D_BIN)

viz_2d: run_2d dirs
	@mkdir -p $(GIFDIR)/euler2d
	$(PY) $(VIZ_PY) $(E2D_H5) \
		--outdir $(GIFDIR)/euler2d \
		--fields /rho,/P,/Mach \
		$(VIZ_DIAG_FLAGS)

# ============================
# 2D Euler (HLLC + gravity)
# ============================
build_2d_hllc: $(E2D_HLLC_BIN)

$(E2D_HLLC_BIN): $(E2D_HLLC_CPP) $(E2D_HDR) Makefile
	$(CXX) $(CXXFLAGS) $(E2D_HLLC_CPP) -o $(E2D_HLLC_BIN) \
		$(INCLUDES) $(LIBS) $(RPATH) $(LDFLAGS)

run_2d_hllc: build_2d_hllc
	./$(E2D_HLLC_BIN)

viz_2d_hllc: run_2d_hllc dirs
	@mkdir -p $(GIFDIR)/euler2d_hllc
	$(PY) $(VIZ_PY) $(E2D_HLLC_H5) \
		--outdir $(GIFDIR)/euler2d_hllc \
		--fields /rho,/P,/Mach \
		$(VIZ_DIAG_FLAGS)

# ============================
# 2D Navier–Stokes (HLLC + viscosity + Poisson gravity)
# ============================
build_ns2d: $(NS2D_BIN)

$(NS2D_BIN): $(NS2D_CPP) Makefile
	$(CXX) $(CXXFLAGS) $(NS2D_CPP) -o $(NS2D_BIN) \
		$(INCLUDES) $(LIBS) $(RPATH) $(LDFLAGS)

run_ns2d: build_ns2d
	./$(NS2D_BIN)

viz_ns2d: run_ns2d dirs
	@mkdir -p $(GIFDIR)/ns2d_poisson
	$(PY) $(VIZ_PY) $(NS2D_H5) \
		--outdir $(GIFDIR)/ns2d_poisson \
		--fields /rho,/P,/Mach \
		$(VIZ_DIAG_FLAGS)

# ============================
# 2D Resistive MHD (HLLD + CT + Poisson gravity)
# ============================
build_mhd2d: $(MHD2D_BIN)

$(MHD2D_BIN): $(MHD2D_CPP) Makefile
	$(CXX) $(CXXFLAGS) $(MHD2D_CPP) -o $(MHD2D_BIN) \
		$(INCLUDES) $(LIBS) $(RPATH) $(LDFLAGS)

run_mhd2d: build_mhd2d
	./$(MHD2D_BIN)

viz_mhd2d: run_mhd2d dirs
	@mkdir -p $(GIFDIR)/mhd2d
	$(PY) $(VIZ_PY) $(MHD2D_H5) \
		--outdir $(GIFDIR)/mhd2d \
		--fields /rho,/P,/Mach,/Bx,/By,/Bz,/divB \
		$(VIZ_DIAG_FLAGS)

# ============================
# 3D Resistive MHD (HLLD + CT)
# ============================
build_mhd3d: $(MHD3D_BIN)

$(MHD3D_BIN): $(MHD3D_CPP) Makefile
	$(CXX) $(CXXFLAGS) $(MHD3D_CPP) -o $(MHD3D_BIN) \
		$(INCLUDES) $(LIBS) $(RPATH) $(LDFLAGS)

run_mhd3d: build_mhd3d
	./$(MHD3D_BIN)

viz_mhd3d: run_mhd3d dirs
	@mkdir -p $(GIFDIR)/mhd3d
	$(PY) $(VIZ3D_PY) $(MHD3D_H5) \
		--outdir $(GIFDIR)/mhd3d \
		--fields /P,/Bmag --gif --log

# ---------------- Clean ----------------
clean:
	rm -f $(SPH_BIN) $(E2D_BIN) $(E2D_HLLC_BIN) $(NS2D_BIN) $(MHD2D_BIN) $(MHD3D_BIN) \
	      $(SPH_H5) $(E2D_H5) $(E2D_HLLC_H5) $(NS2D_H5) $(MHD2D_H5) $(MHD3D_H5)
	rm -rf $(GIFDIR)
