
a
GRunning '/opt/Xilinx/Vitis_HLS/2021.1/bin/unwrapped/lnx64.o/vitis_hls'
*HLSZ200-10h px 
¨
For user 'centos' on host 'ip-172-31-86-173.ec2.internal' (Linux_x86_64 version 3.10.0-1160.31.1.el7.x86_64) on Thu Jan 27 17:19:13 UTC 2022
*HLSZ200-10h px 
G
-On os "CentOS Linux release 7.9.2009 (Core)"
*HLSZ200-10h px 

jIn directory '/home/centos/workspace/median_kernels/Emulation-SW/build/medianblur_accel/medianblur_accel'
*HLSZ200-10h px 
F
+Sourcing Tcl script 'medianblur_accel.tcl'
*HLSZ200-150h px 
e
Running: %s
2001510*hls22
open_project medianblur_accel 2default:defaultZ200-1510h px 
§
Creating and opening project '/home/centos/workspace/median_kernels/Emulation-SW/build/medianblur_accel/medianblur_accel/medianblur_accel'.
*HLSZ200-10h px 
`
Running: %s
2001510*hls2-
set_top medianblur_accel 2default:defaultZ200-1510h px 
Æ
Running: %s
2001510*hls2
ýadd_files /home/centos/workspace/median_kernels/src/xf_median_blur_accel.cpp -cflags  -g -I /home/centos/workspace/median_kernels/src -I /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include -I /home/centos/workspace/median_kernels/src/build  2default:defaultZ200-1510h px 

gAdding design file '/home/centos/workspace/median_kernels/src/xf_median_blur_accel.cpp' to the project
*HLSZ200-10h px 
q
Running: %s
2001510*hls2>
*open_solution -flow_target vitis solution 2default:defaultZ200-1510h px 
±
Creating and opening solution '/home/centos/workspace/median_kernels/Emulation-SW/build/medianblur_accel/medianblur_accel/medianblur_accel/solution'.
*HLSZ200-10h px 
×
Using %sflow_target '%s'
2001505*hls2
 2default:default2
vitis2default:defaultZ200-1505h pxeFor help on HLS 200-1505 see www.xilinx.com/cgi-bin/docs/rdoc?v=2021.1;t=hls+guidance;d=200-1505.html 
°
Setting %s configuration: %s
200435*hls26
"'open_solution -flow_target vitis'2default:default26
"config_interface -m_axi_latency=642default:defaultZ200-435h px 
¼
Setting %s configuration: %s
200435*hls26
"'open_solution -flow_target vitis'2default:default2B
.config_interface -m_axi_alignment_byte_size=642default:defaultZ200-435h px 
¼
Setting %s configuration: %s
200435*hls26
"'open_solution -flow_target vitis'2default:default2B
.config_interface -m_axi_max_widen_bitwidth=5122default:defaultZ200-435h px 
®
Setting %s configuration: %s
200435*hls26
"'open_solution -flow_target vitis'2default:default24
 config_rtl -register_reset_num=32default:defaultZ200-435h px 
d
Running: %s
2001510*hls21
set_part xcvu9p-flgb2104-2-i 2default:defaultZ200-1510h px 
k
Setting target device to '%s'2001611*hls2'
xcvu9p-flgb2104-2-i2default:defaultZ200-1611h px 
x
Running: %s
2001510*hls2E
1create_clock -period 250.000000MHz -name default 2default:defaultZ200-1510h px 
L
1Setting up clock 'default' with a period of 4ns.
*SYNZ201-201h px 
b
Running: %s
2001510*hls2/
config_rtl -kernel_profile 2default:defaultZ200-1510h px 
l
Running: %s
2001510*hls29
%config_dataflow -strict_mode warning 2default:defaultZ200-1510h px 
\
Running: %s
2001510*hls2)
config_debug -enable 2default:defaultZ200-1510h px 
v
Running: %s
2001510*hls2C
/config_export -disable_deadlock_detection=true 2default:defaultZ200-1510h px 
m
Running: %s
2001510*hls2:
&config_rtl -m_axi_conservative_mode=1 2default:defaultZ200-1510h px 
þ
cThe '%s' command is deprecated and will be removed in a future release. Use %s as its replacement.
200483*hls27
#config_rtl -m_axi_conservative_mode2default:default2=
)config_interface -m_axi_conservative_mode2default:defaultZ200-483h px 
f
Running: %s
2001510*hls23
config_interface -m_axi_addr64 2default:defaultZ200-1510h px 
p
Running: %s
2001510*hls2=
)config_interface -m_axi_auto_max_ports=0 2default:defaultZ200-1510h px 

Running: %s
2001510*hls2N
:config_export -format ip_catalog -ipname medianblur_accel 2default:defaultZ200-1510h px 
f
Running: %s
2001510*hls23
csynth_design -synthesis_check 2default:defaultZ200-1510h px 
Ç
«Finished File checks and directory preparation: CPU user time: 0.01 seconds. CPU system time: 0 seconds. Elapsed time: 0.01 seconds; current allocated memory: 108.771 MB.
*HLSZ200-111h px 
z
`Analyzing design file '/home/centos/workspace/median_kernels/src/xf_median_blur_accel.cpp' ... 
*HLSZ200-10h px 

ï'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:379:9
*HLSZ207-5514h px 
­
Ignore interface attribute or pragma which is not used in top function: /home/centos/workspace/median_kernels/src/xf_median_blur_accel.cpp:27:9
*HLSZ207-5528h px 

hunused parameter 'print': /opt/Xilinx/Vitis_HLS/2021.1/common/technology/autopilot/ap_int_base.h:792:16
*HLSZ207-5301h px 

uunused parameter 'src': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:506:30
*HLSZ207-5301h px 

wunused parameter '_data': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:517:30
*HLSZ207-5301h px 

wunused parameter 'index': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:530:14
*HLSZ207-5301h px 

wunused parameter 'index': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:541:20
*HLSZ207-5301h px 

vunused parameter 'dst': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:773:101
*HLSZ207-5301h px 

yunused parameter 'index': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1165:102
*HLSZ207-5301h px 

xunused parameter 'index': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1471:34
*HLSZ207-5301h px 

~unused parameter 'win_size': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:30
*HLSZ207-5301h px 

ï'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:379:9
*HLSZ207-5514h px 
Ë
¯Finished Source Code Analysis and Preprocessing: CPU user time: 6.92 seconds. CPU system time: 0.57 seconds. Elapsed time: 5.75 seconds; current allocated memory: 110.023 MB.
*HLSZ200-111h px 
m
/Using interface defaults for '%s' flow target.
200777*hls2
Vitis2default:defaultZ200-777h px 
Q
6Initial Interval estimation mode is set into default.
*HLSZ214-279h px 
J
/Auto array partition mode is set into default.
*HLSZ214-284h px 
 
In function 'xf::cv::Mat<9, 2160, 3840, 1, 2>::Mat(int, int)', Pragma conflict happens on 'INLINE' and DATAFLOW pragmas: Inline into dataflow region is not suggested (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:658:0)
*HLSZ214-273h px 
 
In function 'xf::cv::Mat<9, 2160, 3840, 1, 2>::Mat(int, int)', Pragma conflict happens on 'INLINE' and DATAFLOW pragmas: Inline into dataflow region is not suggested (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:658:0)
*HLSZ214-273h px 
Õ
¹Inlining function 'hls::stream<ap_uint<24>, 0>::stream()' into 'hls::stream<ap_uint<24>, 2>::stream()' (/opt/Xilinx/Vitis_HLS/2021.1/common/technology/autopilot/hls_stream_39.h:198:43)
*HLSZ214-131h px 
ë
ÏInlining function 'hls::stream<ap_uint<24>, 2>::stream()' into 'xf::cv::Mat<9, 2160, 3840, 1, 2>::Mat(int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:457:5)
*HLSZ214-131h px 
ü
àInlining function 'xf::cv::Mat<9, 2160, 3840, 1, 2>::init(int, int, bool)' into 'xf::cv::Mat<9, 2160, 3840, 1, 2>::Mat(int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:663:2)
*HLSZ214-131h px 
¿
£Inlining function 'hls::stream<ap_uint<32>, 0>::write(ap_uint<32> const&)' into 'xf::cv::MMIterIn<32, 9, 2160, 3840, 1, 2>::Axi2AxiStream(ap_uint<32>*, hls::stream<ap_uint<32>, 0>&, ap_uint<24>&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:957:7)
*HLSZ214-131h px 
Ü
ÀInlining function 'hls::stream<ap_uint<32>, 0>::read(ap_uint<32>&)' into 'hls::stream<ap_uint<32>, 0>::read()' (/opt/Xilinx/Vitis_HLS/2021.1/common/technology/autopilot/hls_stream_39.h:156:9)
*HLSZ214-131h px 
å
ÉInlining function 'hls::stream<ap_uint<24>, 0>::write(ap_uint<24> const&)' into 'void xf::cv::MMIterIn<32, 9, 2160, 3840, 1, 2>::AxiStream2MatStream<2>(hls::stream<ap_uint<32>, 0>&, hls::stream<ap_uint<24>, 2>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1032:46)
*HLSZ214-131h px 
Ò
¶Inlining function 'hls::stream<ap_uint<32>, 0>::read()' into 'void xf::cv::MMIterIn<32, 9, 2160, 3840, 1, 2>::AxiStream2MatStream<2>(hls::stream<ap_uint<32>, 0>&, hls::stream<ap_uint<24>, 2>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1016:27)
*HLSZ214-131h px 
«
Inlining function 'hls::stream<ap_uint<32>, 0>::stream()' into 'xf::cv::MMIterIn<32, 9, 2160, 3840, 1, 2>::Axi2Mat(ap_uint<32>*, hls::stream<ap_uint<24>, 2>&, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1090:35)
*HLSZ214-131h px 
Ü
ÀInlining function 'hls::stream<ap_uint<24>, 0>::read(ap_uint<24>&)' into 'hls::stream<ap_uint<24>, 0>::read()' (/opt/Xilinx/Vitis_HLS/2021.1/common/technology/autopilot/hls_stream_39.h:156:9)
*HLSZ214-131h px 
ÿ
ãInlining function 'hls::stream<ap_uint<24>, 0>::read()' into 'ap_uint<24> xf::cv::Mat<9, 2160, 3840, 1, 2>::read<2, (void*)0>(int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:531:21)
*HLSZ214-131h px 

ýInlining function 'hls::stream<ap_uint<24>, 0>::write(ap_uint<24> const&)' into 'void xf::cv::Mat<9, 2160, 3840, 1, 2>::write<2, (void*)0>(int, ap_uint<24>)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:542:14)
*HLSZ214-131h px 
ò
ÖInlining function 'void xf::cv::ProcessMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, DataType<9, 1>::name (*) [(3840) >> (xfNPixelsPerCycle<1>::datashift)], PixelType<9>::uname (*) [(xfNPixelsPerCycle<1>::nppc) + ((5) - (1))], PixelType<9>::uname*, DataType<9, 1>::name&, unsigned short, unsigned short, unsigned short&, ap_uint<13>*, ap_uint<13>, ap_uint<8>, int&, int&)' into 'void xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, ap_uint<8>, unsigned short, unsigned short)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:431:9)
*HLSZ214-131h px 
é
ÍInlining function 'hls::stream<ap_uint<32>, 0>::write(ap_uint<32> const&)' into 'void xf::cv::MMIterOut<32, 9, 2160, 3840, 1, 1, 2>::MatStream2AxiStream<2>(hls::stream<ap_uint<24>, 2>&, hls::stream<ap_uint<32>, 0>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1282:18)
*HLSZ214-131h px 
é
ÍInlining function 'hls::stream<ap_uint<32>, 0>::write(ap_uint<32> const&)' into 'void xf::cv::MMIterOut<32, 9, 2160, 3840, 1, 1, 2>::MatStream2AxiStream<2>(hls::stream<ap_uint<24>, 2>&, hls::stream<ap_uint<32>, 0>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1272:26)
*HLSZ214-131h px 
Ö
ºInlining function 'hls::stream<ap_uint<24>, 0>::read()' into 'void xf::cv::MMIterOut<32, 9, 2160, 3840, 1, 1, 2>::MatStream2AxiStream<2>(hls::stream<ap_uint<24>, 2>&, hls::stream<ap_uint<32>, 0>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1257:31)
*HLSZ214-131h px 
²
Inlining function 'hls::stream<ap_uint<32>, 0>::read()' into 'xf::cv::MMIterOut<32, 9, 2160, 3840, 1, 1, 2>::AxiStream2Axi(hls::stream<ap_uint<32>, 0>&, ap_uint<32>*, ap_uint<24>&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1325:16)
*HLSZ214-131h px 
¯
Inlining function 'hls::stream<ap_uint<32>, 0>::stream()' into 'xf::cv::MMIterOut<32, 9, 2160, 3840, 1, 1, 2>::Mat2Axi(hls::stream<ap_uint<24>, 2>&, ap_uint<32>*, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1361:42)
*HLSZ214-131h px 
Ã
§Inlining function 'xf::cv::Mat<9, 2160, 3840, 1, 2>::Mat(int, int)' into 'medianblur_accel' (/home/centos/workspace/median_kernels/src/xf_median_blur_accel.cpp:30:45)
*HLSZ214-131h px 
Ã
§Inlining function 'xf::cv::Mat<9, 2160, 3840, 1, 2>::Mat(int, int)' into 'medianblur_accel' (/home/centos/workspace/median_kernels/src/xf_median_blur_accel.cpp:31:48)
*HLSZ214-131h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:153:5) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:158:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:158:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:158:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:158:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:158:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
ã
ÇUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:298:13) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
ã
ÇUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:309:13) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
ã
ÇUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:329:13) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:334:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 4 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:334:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 4 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:334:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 4 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:334:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 4 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:334:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 4 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:437:9) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 4 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
â
ÆUnrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:413:2) in function 'xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>' completely with a factor of 2 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-186h px 
Ì
°Unrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:58:5) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 3 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ð
´Unrolling loop 'xFSortLoop2' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:97:17) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 12 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ð
´Unrolling loop 'xFSortLoop1' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:80:17) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 12 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Í
±Unrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:63:2) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 25 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ð
´Unrolling loop 'xFSortLoop2' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:97:17) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 12 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ð
´Unrolling loop 'xFSortLoop1' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:80:17) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 12 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Í
±Unrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:63:2) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 25 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ð
´Unrolling loop 'xFSortLoop2' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:97:17) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 12 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ð
´Unrolling loop 'xFSortLoop1' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:80:17) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 12 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Í
±Unrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:63:2) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 25 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ô
¸Unrolling loop 'Compute_Grad_Loop' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:44:5) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ì
°Unrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:49:2) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ì
°Unrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:49:2) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ì
°Unrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:49:2) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ì
°Unrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:49:2) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
Ì
°Unrolling loop 'anonymous' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:49:2) in function 'xf::cv::xFMedianProc<3, 1, 9, 5, 25>' completely with a factor of 5 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:31:0)
*HLSZ214-186h px 
ç
ËInlining function 'xf::cv::MMIterIn<32, 9, 2160, 3840, 1, 2>::Array2xfMat(ap_uint<32>*, xf::cv::Mat<9, 2160, 3840, 1, 2>&, int)' into 'void xf::cv::Array2xfMat<32, 9, 2160, 3840, 1>(ap_uint<32>*, xf::cv::Mat<9, 2160, 3840, 1, 2>&, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_utility.hpp:524:0)
*HLSZ214-178h px 

óInlining function 'ap_uint<24> xf::cv::Mat<9, 2160, 3840, 1, 2>::read<2, (void*)0>(int)' into 'void xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, ap_uint<8>, unsigned short, unsigned short)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-178h px 

úInlining function 'void xf::cv::Mat<9, 2160, 3840, 1, 2>::write<2, (void*)0>(int, ap_uint<24>)' into 'void xf::cv::xFMedianNxN<2160, 3840, 3, 9, 1, 0, 3842, 5, 25>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, ap_uint<8>, unsigned short, unsigned short)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_median_blur.hpp:355:0)
*HLSZ214-178h px 
î
ÒInlining function 'xf::cv::MMIterOut<32, 9, 2160, 3840, 1, 1, 2>::xfMat2Array(xf::cv::Mat<9, 2160, 3840, 1, 2>&, ap_uint<32>*, int)' into 'void xf::cv::xfMat2Array<32, 9, 2160, 3840, 1, 1>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, ap_uint<32>*, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_utility.hpp:514:0)
*HLSZ214-178h px 
È
¬Finished Compiling Optimization and Transform: CPU user time: 5.05 seconds. CPU system time: 0.43 seconds. Elapsed time: 5.5 seconds; current allocated memory: 111.358 MB.
*HLSZ200-111h px 
¬
Finished Checking Pragmas: CPU user time: 0 seconds. CPU system time: 0 seconds. Elapsed time: 0 seconds; current allocated memory: 111.359 MB.
*HLSZ200-111h px 
x
SRunning only source code synthesis checks, skipping scheduling and RTL generation.
2001493*hlsZ200-1493h px 
¸
Finished Command csynth_design CPU user time: 11.99 seconds. CPU system time: 1 seconds. Elapsed time: 11.27 seconds; current allocated memory: 111.331 MB.
*HLSZ200-111h px 
6
HLS completed successfully
*HLSZ200-150h px 
«
Total CPU user time: 13.15 seconds. Total CPU system time: 1.34 seconds. Total elapsed time: 12.42 seconds; peak allocated memory: 111.359 MB.
*HLSZ200-112h px 

Exiting %s at %s...
206*common2
	vitis_hls2default:default2,
Thu Jan 27 17:19:25 20222default:defaultZ17-206h px 


End Record