
a
GRunning '/opt/Xilinx/Vitis_HLS/2021.1/bin/unwrapped/lnx64.o/vitis_hls'
*HLSZ200-10h px 
¨
For user 'centos' on host 'ip-172-31-86-173.ec2.internal' (Linux_x86_64 version 3.10.0-1160.31.1.el7.x86_64) on Thu Jan 27 17:41:53 UTC 2022
*HLSZ200-10h px 
G
-On os "CentOS Linux release 7.9.2009 (Core)"
*HLSZ200-10h px 
z
`In directory '/home/centos/workspace/median_kernels/Emulation-SW/build/sobel_accel/sobel_accel'
*HLSZ200-10h px 
A
&Sourcing Tcl script 'sobel_accel.tcl'
*HLSZ200-150h px 
`
Running: %s
2001510*hls2-
open_project sobel_accel 2default:defaultZ200-1510h px 

}Creating and opening project '/home/centos/workspace/median_kernels/Emulation-SW/build/sobel_accel/sobel_accel/sobel_accel'.
*HLSZ200-10h px 
[
Running: %s
2001510*hls2(
set_top sobel_accel 2default:defaultZ200-1510h px 
À
Running: %s
2001510*hls2
÷add_files /home/centos/workspace/median_kernels/src/xf_sobel_accel.cpp -cflags  -g -I /home/centos/workspace/median_kernels/src -I /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include -I /home/centos/workspace/median_kernels/src/build  2default:defaultZ200-1510h px 
{
aAdding design file '/home/centos/workspace/median_kernels/src/xf_sobel_accel.cpp' to the project
*HLSZ200-10h px 
q
Running: %s
2001510*hls2>
*open_solution -flow_target vitis solution 2default:defaultZ200-1510h px 
¢
Creating and opening solution '/home/centos/workspace/median_kernels/Emulation-SW/build/sobel_accel/sobel_accel/sobel_accel/solution'.
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
|
Running: %s
2001510*hls2I
5config_export -format ip_catalog -ipname sobel_accel 2default:defaultZ200-1510h px 
f
Running: %s
2001510*hls23
csynth_design -synthesis_check 2default:defaultZ200-1510h px 
Ç
«Finished File checks and directory preparation: CPU user time: 0.01 seconds. CPU system time: 0 seconds. Elapsed time: 0.01 seconds; current allocated memory: 108.754 MB.
*HLSZ200-111h px 
t
ZAnalyzing design file '/home/centos/workspace/median_kernels/src/xf_sobel_accel.cpp' ... 
*HLSZ200-10h px 

é'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:337:9
*HLSZ207-5514h px 

é'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:341:9
*HLSZ207-5514h px 

é'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:847:9
*HLSZ207-5514h px 

é'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:852:9
*HLSZ207-5514h px 

ê'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:1696:9
*HLSZ207-5514h px 

ê'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:1701:9
*HLSZ207-5514h px 
§
Ignore interface attribute or pragma which is not used in top function: /home/centos/workspace/median_kernels/src/xf_sobel_accel.cpp:35:9
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

runused parameter 't1': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:45:52
*HLSZ207-5301h px 

runused parameter 'm1': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:48:52
*HLSZ207-5301h px 

runused parameter 'b1': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:51:52
*HLSZ207-5301h px 

runused parameter 'm0': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:95:52
*HLSZ207-5301h px 

runused parameter 'm1': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:96:52
*HLSZ207-5301h px 

runused parameter 'm2': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:97:52
*HLSZ207-5301h px 

wunknown HLS pragma ignored: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:336:9
*HLSZ207-5541h px 

é'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:337:9
*HLSZ207-5514h px 

é'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:341:9
*HLSZ207-5514h px 

yunused parameter 'src_buf3': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:566:54
*HLSZ207-5301h px 

é'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:847:9
*HLSZ207-5514h px 

wunknown HLS pragma ignored: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:848:9
*HLSZ207-5541h px 

é'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:852:9
*HLSZ207-5514h px 

zunused parameter 'src_buf4': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:1253:54
*HLSZ207-5301h px 

zunused parameter '_src_mat': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:1544:58
*HLSZ207-5301h px 

|unused parameter 'read_index': /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:1560:26
*HLSZ207-5301h px 

ê'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:1696:9
*HLSZ207-5514h px 

xunknown HLS pragma ignored: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:1697:9
*HLSZ207-5541h px 

ê'Resource pragma' is deprecated, and it will be removed in future release. It is suggested to replace it with 'bind_op/bind_storage pragma'.: /home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:1701:9
*HLSZ207-5514h px 
É
­Finished Source Code Analysis and Preprocessing: CPU user time: 9.53 seconds. CPU system time: 0.6 seconds. Elapsed time: 8.4 seconds; current allocated memory: 110.307 MB.
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
Ä
¨Inlining function 'hls::stream<ap_uint<256>, 0>::write(ap_uint<256> const&)' into 'xf::cv::MMIterIn<256, 9, 2160, 3840, 1, 2>::Axi2AxiStream(ap_uint<256>*, hls::stream<ap_uint<256>, 0>&, ap_uint<21>&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:957:7)
*HLSZ214-131h px 
ß
ÃInlining function 'hls::stream<ap_uint<256>, 0>::read(ap_uint<256>&)' into 'hls::stream<ap_uint<256>, 0>::read()' (/opt/Xilinx/Vitis_HLS/2021.1/common/technology/autopilot/hls_stream_39.h:156:9)
*HLSZ214-131h px 
ç
ËInlining function 'hls::stream<ap_uint<24>, 0>::write(ap_uint<24> const&)' into 'void xf::cv::MMIterIn<256, 9, 2160, 3840, 1, 2>::AxiStream2MatStream<2>(hls::stream<ap_uint<256>, 0>&, hls::stream<ap_uint<24>, 2>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1032:46)
*HLSZ214-131h px 
Õ
¹Inlining function 'hls::stream<ap_uint<256>, 0>::read()' into 'void xf::cv::MMIterIn<256, 9, 2160, 3840, 1, 2>::AxiStream2MatStream<2>(hls::stream<ap_uint<256>, 0>&, hls::stream<ap_uint<24>, 2>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1016:27)
*HLSZ214-131h px 
®
Inlining function 'hls::stream<ap_uint<256>, 0>::stream()' into 'xf::cv::MMIterIn<256, 9, 2160, 3840, 1, 2>::Axi2Mat(ap_uint<256>*, hls::stream<ap_uint<24>, 2>&, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1090:35)
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
¬
Inlining function 'void xf::cv::xfPackPixels<1, 9, 15>(PixelType<15>::name*, StreamType<9>::name&, unsigned short, short, unsigned short&)' into 'void xf::cv::ProcessSobel3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, StreamType<9>::name (*) [(3840) >> (xfNPixelsPerCycle<1>::datashift)], PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, StreamType<9>::name&, StreamType<9>::name&, unsigned short, unsigned short, ap_uint<13>, unsigned short&, unsigned short&, ap_uint<2>, ap_uint<2>, ap_uint<2>, ap_uint<13>, int&, int&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:269:13)
*HLSZ214-131h px 
¬
Inlining function 'void xf::cv::xfPackPixels<1, 9, 15>(PixelType<15>::name*, StreamType<9>::name&, unsigned short, short, unsigned short&)' into 'void xf::cv::ProcessSobel3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, StreamType<9>::name (*) [(3840) >> (xfNPixelsPerCycle<1>::datashift)], PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, StreamType<9>::name&, StreamType<9>::name&, unsigned short, unsigned short, ap_uint<13>, unsigned short&, unsigned short&, ap_uint<2>, ap_uint<2>, ap_uint<2>, ap_uint<13>, int&, int&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:268:13)
*HLSZ214-131h px 
¬
Inlining function 'void xf::cv::xfPackPixels<1, 9, 15>(PixelType<15>::name*, StreamType<9>::name&, unsigned short, short, unsigned short&)' into 'void xf::cv::ProcessSobel3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, StreamType<9>::name (*) [(3840) >> (xfNPixelsPerCycle<1>::datashift)], PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, StreamType<9>::name&, StreamType<9>::name&, unsigned short, unsigned short, ap_uint<13>, unsigned short&, unsigned short&, ap_uint<2>, ap_uint<2>, ap_uint<2>, ap_uint<13>, int&, int&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:258:13)
*HLSZ214-131h px 
¬
Inlining function 'void xf::cv::xfPackPixels<1, 9, 15>(PixelType<15>::name*, StreamType<9>::name&, unsigned short, short, unsigned short&)' into 'void xf::cv::ProcessSobel3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, StreamType<9>::name (*) [(3840) >> (xfNPixelsPerCycle<1>::datashift)], PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, StreamType<9>::name&, StreamType<9>::name&, unsigned short, unsigned short, ap_uint<13>, unsigned short&, unsigned short&, ap_uint<2>, ap_uint<2>, ap_uint<2>, ap_uint<13>, int&, int&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:257:13)
*HLSZ214-131h px 
¬
Inlining function 'void xf::cv::xfPackPixels<1, 9, 15>(PixelType<15>::name*, StreamType<9>::name&, unsigned short, short, unsigned short&)' into 'void xf::cv::ProcessSobel3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, StreamType<9>::name (*) [(3840) >> (xfNPixelsPerCycle<1>::datashift)], PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, StreamType<9>::name&, StreamType<9>::name&, unsigned short, unsigned short, ap_uint<13>, unsigned short&, unsigned short&, ap_uint<2>, ap_uint<2>, ap_uint<2>, ap_uint<13>, int&, int&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:254:13)
*HLSZ214-131h px 
¬
Inlining function 'void xf::cv::xfPackPixels<1, 9, 15>(PixelType<15>::name*, StreamType<9>::name&, unsigned short, short, unsigned short&)' into 'void xf::cv::ProcessSobel3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, StreamType<9>::name (*) [(3840) >> (xfNPixelsPerCycle<1>::datashift)], PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, StreamType<9>::name&, StreamType<9>::name&, unsigned short, unsigned short, ap_uint<13>, unsigned short&, unsigned short&, ap_uint<2>, ap_uint<2>, ap_uint<2>, ap_uint<13>, int&, int&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:253:13)
*HLSZ214-131h px 
æ
ÊInlining function 'void xf::cv::xfPackPixels<1, 9, 15>(PixelType<15>::name*, StreamType<9>::name&, unsigned short, short, unsigned short&)' into 'void xf::cv::xFSobelFilter3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840, false>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, unsigned short, unsigned short)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:440:9)
*HLSZ214-131h px 
æ
ÊInlining function 'void xf::cv::xfPackPixels<1, 9, 15>(PixelType<15>::name*, StreamType<9>::name&, unsigned short, short, unsigned short&)' into 'void xf::cv::xFSobelFilter3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840, false>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, unsigned short, unsigned short)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:439:9)
*HLSZ214-131h px 

ñInlining function 'void xf::cv::ProcessSobel3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, StreamType<9>::name (*) [(3840) >> (xfNPixelsPerCycle<1>::datashift)], PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, PixelType<15>::name*, StreamType<9>::name&, StreamType<9>::name&, unsigned short, unsigned short, ap_uint<13>, unsigned short&, unsigned short&, ap_uint<2>, ap_uint<2>, ap_uint<2>, ap_uint<13>, int&, int&)' into 'void xf::cv::xFSobelFilter3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840, false>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, unsigned short, unsigned short)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:386:9)
*HLSZ214-131h px 
í
ÑInlining function 'hls::stream<ap_uint<256>, 0>::write(ap_uint<256> const&)' into 'void xf::cv::MMIterOut<256, 9, 2160, 3840, 1, 1, 2>::MatStream2AxiStream<2>(hls::stream<ap_uint<24>, 2>&, hls::stream<ap_uint<256>, 0>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1282:18)
*HLSZ214-131h px 
í
ÑInlining function 'hls::stream<ap_uint<256>, 0>::write(ap_uint<256> const&)' into 'void xf::cv::MMIterOut<256, 9, 2160, 3840, 1, 1, 2>::MatStream2AxiStream<2>(hls::stream<ap_uint<24>, 2>&, hls::stream<ap_uint<256>, 0>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1272:26)
*HLSZ214-131h px 
Ø
¼Inlining function 'hls::stream<ap_uint<24>, 0>::read()' into 'void xf::cv::MMIterOut<256, 9, 2160, 3840, 1, 1, 2>::MatStream2AxiStream<2>(hls::stream<ap_uint<24>, 2>&, hls::stream<ap_uint<256>, 0>&, int, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1257:31)
*HLSZ214-131h px 
¶
Inlining function 'hls::stream<ap_uint<256>, 0>::read()' into 'xf::cv::MMIterOut<256, 9, 2160, 3840, 1, 1, 2>::AxiStream2Axi(hls::stream<ap_uint<256>, 0>&, ap_uint<256>*, ap_uint<21>&)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1325:16)
*HLSZ214-131h px 
²
Inlining function 'hls::stream<ap_uint<256>, 0>::stream()' into 'xf::cv::MMIterOut<256, 9, 2160, 3840, 1, 1, 2>::Mat2Axi(hls::stream<ap_uint<24>, 2>&, ap_uint<256>*, int, int, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_structs.hpp:1361:42)
*HLSZ214-131h px 
¸
Inlining function 'xf::cv::Mat<9, 2160, 3840, 1, 2>::Mat(int, int)' into 'sobel_accel' (/home/centos/workspace/median_kernels/src/xf_sobel_accel.cpp:38:45)
*HLSZ214-131h px 
¸
Inlining function 'xf::cv::Mat<9, 2160, 3840, 1, 2>::Mat(int, int)' into 'sobel_accel' (/home/centos/workspace/median_kernels/src/xf_sobel_accel.cpp:43:45)
*HLSZ214-131h px 
¸
Inlining function 'xf::cv::Mat<9, 2160, 3840, 1, 2>::Mat(int, int)' into 'sobel_accel' (/home/centos/workspace/median_kernels/src/xf_sobel_accel.cpp:48:45)
*HLSZ214-131h px 
Æ
ªUnrolling loop 'Compute_Grad_Loop' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:153:5) in function 'xf::cv::xFSobel3x3<3, 1, 15, 15>' completely with a factor of 1 (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:138:0)
*HLSZ214-186h px 
ë
ÏInlining function 'xf::cv::MMIterIn<256, 9, 2160, 3840, 1, 2>::Array2xfMat(ap_uint<256>*, xf::cv::Mat<9, 2160, 3840, 1, 2>&, int)' into 'void xf::cv::Array2xfMat<256, 9, 2160, 3840, 1>(ap_uint<256>*, xf::cv::Mat<9, 2160, 3840, 1, 2>&, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_utility.hpp:524:0)
*HLSZ214-178h px 
³
Inlining function 'ap_uint<24> xf::cv::Mat<9, 2160, 3840, 1, 2>::read<2, (void*)0>(int)' into 'void xf::cv::xFSobelFilter3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840, false>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, unsigned short, unsigned short)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:304:0)
*HLSZ214-178h px 
º
Inlining function 'void xf::cv::Mat<9, 2160, 3840, 1, 2>::write<2, (void*)0>(int, ap_uint<24>)' into 'void xf::cv::xFSobelFilter3x3<9, 9, 2160, 3840, 3, 15, 15, 1, 9, 9, 3840, false>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, xf::cv::Mat<9, 2160, 3840, 1, 2>&, unsigned short, unsigned short)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/imgproc/xf_sobel.hpp:304:0)
*HLSZ214-178h px 
ò
ÖInlining function 'xf::cv::MMIterOut<256, 9, 2160, 3840, 1, 1, 2>::xfMat2Array(xf::cv::Mat<9, 2160, 3840, 1, 2>&, ap_uint<256>*, int)' into 'void xf::cv::xfMat2Array<256, 9, 2160, 3840, 1, 1>(xf::cv::Mat<9, 2160, 3840, 1, 2>&, ap_uint<256>*, int)' (/home/centos/workspace/median_kernels/libs/xf_opencv/L1/include/common/xf_utility.hpp:514:0)
*HLSZ214-178h px 
É
­Finished Compiling Optimization and Transform: CPU user time: 5.79 seconds. CPU system time: 0.41 seconds. Elapsed time: 6.23 seconds; current allocated memory: 111.475 MB.
*HLSZ200-111h px 
¬
Finished Checking Pragmas: CPU user time: 0 seconds. CPU system time: 0 seconds. Elapsed time: 0 seconds; current allocated memory: 111.476 MB.
*HLSZ200-111h px 
x
SRunning only source code synthesis checks, skipping scheduling and RTL generation.
2001493*hlsZ200-1493h px 
»
Finished Command csynth_design CPU user time: 15.34 seconds. CPU system time: 1.01 seconds. Elapsed time: 14.66 seconds; current allocated memory: 111.448 MB.
*HLSZ200-111h px 
6
HLS completed successfully
*HLSZ200-150h px 
ª
Total CPU user time: 16.5 seconds. Total CPU system time: 1.36 seconds. Total elapsed time: 15.81 seconds; peak allocated memory: 111.476 MB.
*HLSZ200-112h px 

Exiting %s at %s...
206*common2
	vitis_hls2default:default2,
Thu Jan 27 17:42:08 20222default:defaultZ17-206h px 


End Record