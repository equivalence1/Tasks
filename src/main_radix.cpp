#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

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

void print_buff(gpu::gpu_mem_32u& as_gpu, unsigned int n)
{
    std::cout << "printing buffer:\n";
    std::vector<unsigned int> a(n);
    as_gpu.readN(a.data(), n);
    for (int i = 0; i < (int)n; i++) {
        std::cout << i << ": " << a[i] << std::endl;
    }
}

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

void radix(gpu::gpu_mem_32u& as_gpu,
           unsigned int n,
           ocl::Kernel& count,
           ocl::Kernel& scan,
           ocl::Kernel& reorder)
{
    unsigned int mask_width = 2;

    gpu::gpu_mem_32u counts_gpu;
    counts_gpu.resizeN(n * (1 << mask_width));
    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    unsigned int wg_size = BLOCK_SIZE;
    unsigned int gw_size = n;

    for (int i = 0; i < 32; i += mask_width) {
        count.exec(gpu::WorkSize(wg_size, gw_size),
                   as_gpu, counts_gpu, n, i, mask_width);
        gpu::gpu_mem_32u s = prefix_sum(scan, counts_gpu, n * (1 << mask_width), wg_size);
        reorder.exec(gpu::WorkSize(wg_size, gw_size),
                     as_gpu, s, bs_gpu, n, i, mask_width);
        as_gpu.swap(bs_gpu);
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
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            t.restart(); // Don't count vector's copy constructor time
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
    count.compile();
    ocl::Kernel scan(radix_kernel, radix_kernel_length, "scan", "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE));
    scan.compile();
    ocl::Kernel reorder(radix_kernel, radix_kernel_length, "reorder");
    reorder.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            radix(as_gpu, n, count, scan, reorder);
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
