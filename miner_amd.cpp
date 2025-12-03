#include <iostream>
#include <windows.h>
#include <cstring>
#include <CL/cl.h>

#define SHM_NAME "/shared_mem"
const int CTX_SIZE_BYTES = 160;
const int KEY_SIZE_BYTES = 32;
const int HASH_NO_SIG_SIZE_BYTES = 32;

struct SharedData {
    volatile uint64_t nonce;
    volatile uint8_t data[CTX_SIZE_BYTES + KEY_SIZE_BYTES + HASH_NO_SIG_SIZE_BYTES];
};

const char* opencl_kernel = R"CL(
#define ROTRIGHT(a,b) (((a)>>(b))|((a)<<(32-(b))))
#define ROTLEFT(a,b) (((a)<<(b))|((a)>>(32-(b))))
#define CH(x,y,z) (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x) (ROTRIGHT(x,2)^ROTRIGHT(x,13)^ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6)^ROTRIGHT(x,11)^ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7)^ROTRIGHT(x,18)^((x)>>3))
#define SIG1(x) (ROTRIGHT(x,17)^ROTRIGHT(x,19)^((x)>>10))

__constant uint k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__kernel void mining_kernel(
    __global uchar* gpu_num,
    __global uchar* key_data,
    __global uchar* ctx_data,
    __global uchar* hash_no_sig_in,
    __global ulong* nonce_out,
    __global ulong* nonce4hashrate)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    __local uint state[8];
    __local uchar local_key[32];

    if(lid < 8) state[lid] = (((__global uint*)ctx_data)[lid]);
    if(lid < 32) local_key[lid] = key_data[lid];

    barrier(CLK_LOCAL_MEM_FENCE);

    ulong nonce = gid;
    if(lid == 0) atomic_inc((volatile __global uint*)nonce4hashrate);
}
)CL";

int main(int argc, char* argv[]) {
    std::cout << "BTCW GPU Miner AMD v26.6 (OpenCL)" << std::endl;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 256, device_name, NULL);
    std::cout << "Device: " << device_name << std::endl;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    program = clCreateProgramWithSource(context, 1, &opencl_kernel, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = new char[log_size];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        std::cerr << "Build error: " << log << std::endl;
        delete[] log;
        return 1;
    }

    kernel = clCreateKernel(program, "mining_kernel", &err);

    HANDLE hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
                                         sizeof(SharedData), SHM_NAME);
    SharedData* shared_data = (SharedData*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0,
                                                          sizeof(SharedData));

    uint8_t* h_gpu_num = new uint8_t[1]; *h_gpu_num = 0;
    uint8_t* h_ctx_data = new uint8_t[CTX_SIZE_BYTES];
    uint8_t* h_key_data = new uint8_t[KEY_SIZE_BYTES];
    uint8_t* h_hash_no_sig = new uint8_t[HASH_NO_SIG_SIZE_BYTES];
    uint8_t* h_nonce = new uint8_t[8];
    uint8_t* h_hashrate = new uint8_t[8];

    cl_mem d_gpu_num = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, NULL, NULL);
    cl_mem d_ctx_data = clCreateBuffer(context, CL_MEM_READ_WRITE, CTX_SIZE_BYTES, NULL, NULL);
    cl_mem d_key_data = clCreateBuffer(context, CL_MEM_READ_ONLY, KEY_SIZE_BYTES, NULL, NULL);
    cl_mem d_hash_no_sig = clCreateBuffer(context, CL_MEM_READ_ONLY, HASH_NO_SIG_SIZE_BYTES, NULL, NULL);
    cl_mem d_nonce = clCreateBuffer(context, CL_MEM_READ_WRITE, 8, NULL, NULL);
    cl_mem d_hashrate = clCreateBuffer(context, CL_MEM_READ_WRITE, 8, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_gpu_num);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_key_data);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_ctx_data);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_hash_no_sig);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_nonce);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_hashrate);

    std::cout << "CONNECTED - Mining started..." << std::endl;

    size_t global_work_size = 32768;
    size_t local_work_size = 256;

    while(true) {
        memcpy(h_key_data, (void*)&shared_data->data[0], KEY_SIZE_BYTES);
        memcpy(h_ctx_data, (void*)&shared_data->data[32], CTX_SIZE_BYTES);
        memcpy(h_hash_no_sig, (void*)&shared_data->data[192], HASH_NO_SIG_SIZE_BYTES);

        clEnqueueWriteBuffer(queue, d_key_data, CL_FALSE, 0, KEY_SIZE_BYTES, h_key_data, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_ctx_data, CL_FALSE, 0, CTX_SIZE_BYTES, h_ctx_data, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_hash_no_sig, CL_FALSE, 0, HASH_NO_SIG_SIZE_BYTES, h_hash_no_sig, 0, NULL, NULL);

        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        clFinish(queue);

        clEnqueueReadBuffer(queue, d_nonce, CL_FALSE, 0, 8, h_nonce, 0, NULL, NULL);
        clFinish(queue);

        Sleep(100);
    }

    clReleaseMemObject(d_gpu_num);
    clReleaseMemObject(d_ctx_data);
    clReleaseMemObject(d_key_data);
    clReleaseMemObject(d_hash_no_sig);
    clReleaseMemObject(d_nonce);
    clReleaseMemObject(d_hashrate);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
