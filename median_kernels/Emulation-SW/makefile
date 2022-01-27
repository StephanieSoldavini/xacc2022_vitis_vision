#
# this file was created by a computer. trust it.
#

# compiler tools
XILINX_VITIS ?= /opt/Xilinx/Vitis/2021.1
XILINX_XRT ?= /opt/xilinx/xrt
XILINX_VIVADO ?= /opt/Xilinx/Vivado/2021.1
XILINX_VIVADO_HLS ?= $(XILINX_VITIS)/Vivado_HLS

VPP ?= ${XILINX_VITIS}/bin/v++
RM = rm -f
RMDIR = rm -rf

VITIS_PLATFORM = xilinx_aws-vu9p-f1_shell-v04261818_201920_2
VITIS_PLATFORM_DIR = /home/centos/aws-fpga/Vitis/aws_platform/xilinx_aws-vu9p-f1_shell-v04261818_201920_2
VITIS_PLATFORM_PATH = $(VITIS_PLATFORM_DIR)/xilinx_aws-vu9p-f1_shell-v04261818_201920_2.xpfm

TARGET = sw_emu
VPP_OPTS = --target $(TARGET)

#
# Hw kernel files
#


BUILD_SUBDIRS += build
HW_KERNEL_OBJS += build/medianblur_accel.xo
SRC_KERNEL_OBJS += build/medianblur_accel.xo
BUILD_SUBDIRS += build
HW_KERNEL_OBJS += build/sobel_accel.xo
SRC_KERNEL_OBJS += build/sobel_accel.xo

ALL_MESSAGE_FILES = $(subst .xo,.o,.mdb,$(SRC_KERNEL_OBJS))

#
# primary build targets
#

.PHONY: all clean
all: $(HW_KERNEL_OBJS)


clean:
	-$(RM) $(SRC_KERNEL_OBJS) $(ALL_MESSAGE_FILES) 
	-$(RMDIR) $(BUILD_SUBDIRS)
	-$(RMDIR) .Xil

.PHONY: incremental
incremental: all


nothing:

#
# Hw Kernels compile;
#

build/medianblur_accel.xo: ../src/xf_median_blur_accel.cpp ../src/xf_median_blur_config.h medianblur_accel-compile.cfg
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --config medianblur_accel-compile.cfg -I/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include -I/home/centos/workspace/median_kernels/src/build -o"$@" "$<"

build/sobel_accel.xo: ../src/xf_sobel_accel.cpp ../src/xf_sobel_config.h sobel_accel-compile.cfg
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --config sobel_accel-compile.cfg -I/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include -I/home/centos/workspace/median_kernels/src/build -o"$@" "$<"

