#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

#define BLOCK_SIZE 128


/*
 * since shared_device_buffer's copy constructor only
 * does shallow copy, it's okay to return by value
 */
gpu::gpu_mem_32u prefix_sum(ocl::Kernel& scan,
                            gpu::gpu_mem_32u& as_gpu,
                            unsigned int n,
                            unsigned int wg_size)
{
    unsigned int res_n = (n + wg_size - 1) / wg_size * wg_size;

    gpu::gpu_mem_32u sums;
    sums.resizeN(res_n);
    gpu::gpu_mem_32u b_sums;
    b_sums.resizeN(res_n / wg_size + 1);
    {
        // it's important to set first element to zero
        unsigned int z = 0;
        b_sums.writeN(&z, 1);
    }

    scan.exec(gpu::WorkSize(wg_size, n),
              as_gpu, b_sums, sums, n, 1);

    if (wg_size >= n)
        return sums;

    b_sums = prefix_sum(scan, b_sums, res_n / wg_size + 1, wg_size);
    scan.exec(gpu::WorkSize(wg_size, n),
              as_gpu, b_sums, sums, n, 0);

    return sums;
}

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        ocl::Kernel scan(sum_kernel, sum_kernel_length, "scan");
        scan.compile();

        gpu::gpu_mem_32u as_gpu;
        as_gpu.resize(n * sizeof(unsigned int));
        as_gpu.write(as.data(), n * sizeof(unsigned int));

        std::vector<unsigned int> res_cpu(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            auto res_gpu = prefix_sum(scan, as_gpu, n, BLOCK_SIZE);
            unsigned int res;
            res_gpu.read(&res, sizeof(unsigned int), (n - 1) * sizeof(unsigned int));
            EXPECT_THE_SAME(reference_sum, res, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}