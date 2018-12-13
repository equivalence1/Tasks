#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include <utility>
#include "cl/max_prefix_sum_cl.h"

#define BLOCK_SIZE 128

/*
 * since shared_device_buffer's copy constructor only
 * does shallow copy, it's okay to return by value
 */
gpu::gpu_mem_32i prefix_sum(ocl::Kernel& scan,
                            gpu::gpu_mem_32i& as_gpu,
                            unsigned int n,
                            unsigned int wg_size)
{
    unsigned int res_n = (n + wg_size - 1) / wg_size * wg_size;

    gpu::gpu_mem_32i sums;
    sums.resizeN(res_n);
    gpu::gpu_mem_32i b_sums;
    b_sums.resizeN(res_n / wg_size + 1);
    {
        // it's important to set first element to zero
        int z = 0;
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

gpu::gpu_mem_32i max_elem(ocl::Kernel& mx,
                          gpu::gpu_mem_32i& as_gpu,
                          gpu::gpu_mem_32i& maxes,
                          unsigned int n) {
    if (n == 1)
        return maxes;

    mx.exec(gpu::WorkSize(BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE),
              as_gpu, maxes, n);

    return max_elem(mx, as_gpu, maxes, n / 2);
}

std::pair<int, int> max_prefix_sum(ocl::Kernel& scan,
                                   ocl::Kernel& mx,
                                   gpu::gpu_mem_32i& as_gpu,
                                   unsigned int n) {
    auto sums_gpu = prefix_sum(scan, as_gpu, n, BLOCK_SIZE);
    std::vector<int> sums_cpu(n);
    sums_gpu.read(sums_cpu.data(), sizeof(int) * n, 0);

    std::vector<int> maxes_cpu(n);
    for (int i = 0; i < n; i++) {
        maxes_cpu[i] = i;
    }
    gpu::gpu_mem_32i maxes;
    maxes.resize(n * sizeof(int));
    maxes.write(maxes_cpu.data(), n * sizeof(int));

    auto gpu_res = max_elem(mx, sums_gpu, maxes, n);
    std::vector<int> cpu_res(1);
    gpu_res.read(cpu_res.data(), sizeof(int), 0);

    return std::make_pair(sums_cpu[cpu_res[0]], cpu_res[0]);
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
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            ocl::Kernel scan(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "scan");
            scan.compile();

            ocl::Kernel mx(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_scan");
            mx.compile();

            as.insert(as.begin(), 0);
            gpu::gpu_mem_32i as_gpu;
            as_gpu.resize(n * sizeof(as[0]));
            as_gpu.write(as.data(), n * sizeof(as[0]));

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                auto res = max_prefix_sum(scan, mx, as_gpu, (unsigned int)n);
                EXPECT_THE_SAME(reference_max_sum, res.first, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, res.second, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
