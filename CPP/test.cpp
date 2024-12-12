#include <iostream>
#include <omp.h>

int main() {
    // Set the number of threads explicitly
    int num_threads_to_set = 32; // Change this as needed
    omp_set_num_threads(num_threads_to_set);

    // Get the maximum number of threads OpenMP will use
    int max_threads = omp_get_max_threads();

    // Print the result
    std::cout << "Number of threads set: " << num_threads_to_set << std::endl;
    std::cout << "Maximum threads available (omp_get_max_threads): " << max_threads << std::endl;

    // Test a parallel region
    #pragma omp parallel
    {
        #pragma omp critical
        {
            std::cout << "Hello from thread " << omp_get_thread_num()
                      << " out of " << omp_get_num_threads() << " threads." << std::endl;
        }
    }

    return 0;
}
