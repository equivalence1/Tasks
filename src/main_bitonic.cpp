#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

#define BLOCK_SIZE 128

void print_buff(gpu::gpu_mem_32f& as_gpu, unsigned int n)
{
    std::cout << "printing buffer:\n";
    std::vector<float> a(n);
    as_gpu.readN(a.data(), n);
    for (int i = 0; i < (int)n; i++) {
        std::cout << i << ": " << a[i] << std::endl;
    }
}

void bitonic(ocl::Kernel& bitonic_large,
             ocl::Kernel& bitonic_small,
             gpu::gpu_mem_32f& as_gpu,
             unsigned int n)
{
    unsigned int wg_size = BLOCK_SIZE;
    unsigned int gw_size = (n + wg_size - 1) / wg_size * wg_size;

    for (unsigned int bsize = 2; bsize / 2 < n; bsize *= 2) {
        unsigned int skip = bsize / 2;
        for (; skip * 2 > wg_size; skip /= 2) {
            bitonic_large.exec(gpu::WorkSize(wg_size, gw_size),
                               as_gpu, n, bsize, skip);
        }
        bitonic_small.exec(gpu::WorkSize(wg_size, gw_size),
                           as_gpu, n, bsize, skip);
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel bitonic_large(bitonic_kernel, bitonic_kernel_length, "bitonic_large");
        bitonic_large.compile();
        ocl::Kernel bitonic_small(bitonic_kernel, bitonic_kernel_length, "bitonic_small", "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE));
        bitonic_small.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            bitonic(bitonic_large, bitonic_small, as_gpu, n);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
