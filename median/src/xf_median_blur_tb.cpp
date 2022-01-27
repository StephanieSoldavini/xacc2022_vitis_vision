/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/xf_headers.hpp"
#include "xcl2.hpp"
#include "xf_median_blur_config.h"
#include "xf_sobel_config.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <INPUT IMAGE PATH 1>\n", argv[0]);
        return EXIT_FAILURE;
    }

    cv::Mat in_img, out_img, ocv_ref, diff;
    cv::Mat c_grad_x_1, c_grad_y_1, nomed_c_grad_y_1, nomed_c_grad_x_1;
    cv::Mat c_grad_x, c_grad_y;
    cv::Mat hls_grad_x, hls_grad_y;
    cv::Mat diff_grad_x, diff_grad_y;

//  Reading in the image:
#if GRAY
    in_img = cv::imread(argv[1], 0); // reading in the gray image
#else
    in_img = cv::imread(argv[1], 1); // reading in the color image
#endif

    if (in_img.data == NULL) {
        fprintf(stderr, "ERROR: Cannot open image %s\n ", argv[1]);
        return EXIT_FAILURE;
    }
    int scale = 1;
    int delta = 0;
// create memory for output image
#if GRAY
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC1);
    out_img.create(in_img.rows, in_img.cols, CV_8UC1); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC1);
#else
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC3);
    out_img.create(in_img.rows, in_img.cols, CV_8UC3); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC3);
#endif

    int ddepth = CV_8U;
    typedef unsigned char TYPE; // Should be short int when ddepth is CV_16S


#if GRAY
#define PTYPE CV_8UC1 // Should be CV_16S when ddepth is CV_16S
#else
#define PTYPE CV_8UC3 // Should be CV_16S when ddepth is CV_16S
#endif

    hls_grad_x.create(in_img.rows, in_img.cols, PTYPE);
    hls_grad_y.create(in_img.rows, in_img.cols, PTYPE);
    diff_grad_x.create(in_img.rows, in_img.cols, PTYPE);
    diff_grad_y.create(in_img.rows, in_img.cols, PTYPE);

    // OpenCV reference:
    cv::medianBlur(in_img, ocv_ref, WINDOW_SIZE);
    imwrite("ocv_ref.jpg", ocv_ref);
    cv::Sobel(in_img, nomed_c_grad_x_1, ddepth, 1, 0, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);
    cv::Sobel(in_img, nomed_c_grad_y_1, ddepth, 0, 1, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);
    cv::Sobel(ocv_ref, c_grad_x_1, ddepth, 1, 0, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);
    cv::Sobel(ocv_ref, c_grad_y_1, ddepth, 0, 1, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);
    imwrite("out_ocvx.jpg", c_grad_x_1);
    imwrite("out_ocvy.jpg", c_grad_y_1);
    imwrite("nomed_out_ocvx.jpg", nomed_c_grad_x_1);
    imwrite("nomed_out_ocvy.jpg", nomed_c_grad_y_1);
// OpenCL section:
    int height = in_img.rows;
    int width = in_img.cols;
#if GRAY
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 1 * sizeof(unsigned char);
#else
    size_t image_in_size_bytes = in_img.rows * in_img.cols * 3 * sizeof(unsigned char);
#endif
    size_t image_out_size_bytes = image_in_size_bytes;

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Context, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_medianblur");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "medianblur_accel", &err));
    OCL_CHECK(err, cl::Kernel kernel_s(program, "sobel_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY, image_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY, image_out_size_bytes, NULL, &err));
    std::vector<cl::Memory> /*inBufVec, */outBufVec1, outBufVec2;
        //cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width * CH_TYPE));
        cl::Buffer imageFromDevice1(context, CL_MEM_WRITE_ONLY, (height * width * CH_TYPE));
        cl::Buffer imageFromDevice2(context, CL_MEM_WRITE_ONLY, (height * width * CH_TYPE));
    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, buffer_inImage));
    OCL_CHECK(err, err = kernel.setArg(1, in_img.rows));
    OCL_CHECK(err, err = kernel.setArg(2, in_img.cols));
    OCL_CHECK(err, err = kernel.setArg(3, buffer_outImage));

    kernel_s.setArg(0, buffer_outImage);
    kernel_s.setArg(1, imageFromDevice1);
    kernel_s.setArg(2, imageFromDevice2);
    kernel_s.setArg(3, height);
    kernel_s.setArg(4, width);

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err,
              queue.enqueueWriteBuffer(buffer_inImage,      // buffer on the FPGA
                                       CL_TRUE,             // blocking call
                                       0,                   // buffer offset in bytes
                                       image_in_size_bytes, // Size in bytes
                                       in_img.data,         // Pointer to the data to copy
                                       nullptr, &event));

    // Execute the kernel:
    OCL_CHECK(err, err = queue.enqueueTask(kernel));
    OCL_CHECK(err, err = queue.enqueueTask(kernel_s));
    // Copy Result from Device Global Memory to Host Local Memory
    /*
    queue.enqueueReadBuffer(buffer_outImage, // This buffers data will be read
                            CL_TRUE,         // blocking call
                            0,               // offset
                            image_out_size_bytes,
                            out_img.data, // Data will be stored here
                            nullptr, &event);
*/
    queue.enqueueReadBuffer(imageFromDevice1, CL_TRUE, 0, (height * width * CH_TYPE), hls_grad_x.data);
    queue.enqueueReadBuffer(imageFromDevice2, CL_TRUE, 0, (height * width * CH_TYPE), hls_grad_y.data);

    // Clean up:
    queue.finish();
#if (FILTER_WIDTH == 3 | FILTER_WIDTH == 5)
    absdiff(c_grad_x_1, hls_grad_x, diff_grad_x);
    absdiff(c_grad_y_1, hls_grad_y, diff_grad_y);
#endif
    // Write down output images:
    //cv::imwrite("hls_out.jpg", out_img); // kernel output
    cv::imwrite("ref_img.jpg", ocv_ref); // reference image
    cv::imwrite("out_errorx.jpg", diff_grad_x);
    cv::imwrite("out_errory.jpg", diff_grad_y);
    //absdiff(ocv_ref, out_img, diff);
    // Save the difference image for debugging purpose:
    float err_per, err_per1;
    int ret;

    xf::cv::analyzeDiff(diff_grad_x, 0, err_per);
    xf::cv::analyzeDiff(diff_grad_y, 0, err_per1);

    if (err_per > 0.0f) {
    	fprintf(stderr, "Test failed .... !!!\n ");
    	ret = 1;
    } else {
    	std::cout << "Test Passed .... !!!" << std::endl;
    	ret = 0;
    }

    return ret;
}
