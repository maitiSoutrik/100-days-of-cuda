#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <getopt.h>

// Default parameters
#define DEFAULT_POPULATION_SIZE 1024
#define DEFAULT_CHROMOSOME_SIZE 10
#define DEFAULT_GENERATIONS 100
#define DEFAULT_CROSSOVER_RATE 0.8f
#define DEFAULT_MUTATION_RATE 0.1f
#define DEFAULT_TOURNAMENT_SIZE 4
#define DEFAULT_ELITISM_COUNT 2
#define DEFAULT_MIN_VALUE -10.0f
#define DEFAULT_MAX_VALUE 10.0f

// Function types
enum FunctionType {
    SPHERE,
    RASTRIGIN,
    ROSENBROCK,
    ACKLEY,
    GRIEWANK
};

// Structure to hold GA parameters
typedef struct {
    int populationSize;
    int chromosomeSize;
    int generations;
    float crossoverRate;
    float mutationRate;
    int tournamentSize;
    int elitismCount;
    float minValue;
    float maxValue;
    FunctionType functionType;
} GAParams;

// Structure to hold chromosome data
typedef struct {
    float* genes;
    float fitness;
} Chromosome;

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Function prototypes
void runGeneticAlgorithmCPU(GAParams params);
void runGeneticAlgorithmGPU(GAParams params);
float evaluateFitness(float* chromosome, int size, FunctionType functionType, float minValue, float maxValue);
void initializePopulation(Chromosome* population, int populationSize, int chromosomeSize, float minValue, float maxValue);
void freePopulation(Chromosome* population, int populationSize);
int tournamentSelection(Chromosome* population, int populationSize, int tournamentSize);
void crossover(Chromosome* population, int parent1Idx, int parent2Idx, float* child1, float* child2, int chromosomeSize, float crossoverRate);
void mutate(float* chromosome, int chromosomeSize, float mutationRate, float minValue, float maxValue);
void printBestSolution(Chromosome* population, int populationSize, int generation, FunctionType functionType);
const char* getFunctionName(FunctionType functionType);

// CUDA kernels
__device__ float evaluateFitnessDevice(float* chromosome, int size, FunctionType functionType, float minValue, float maxValue);
__global__ void initializePopulationKernel(float* population, int populationSize, int chromosomeSize, float minValue, float maxValue, unsigned int seed);
__global__ void evaluateFitnessKernel(float* population, float* fitness, int populationSize, int chromosomeSize, FunctionType functionType, float minValue, float maxValue);
__global__ void tournamentSelectionKernel(float* fitness, int* selected, int populationSize, int tournamentSize, unsigned int seed);
__global__ void crossoverAndMutateKernel(float* population, float* newPopulation, int* selected, int populationSize, int chromosomeSize, float crossoverRate, float mutationRate, float minValue, float maxValue, unsigned int seed);
__global__ void elitismKernel(float* population, float* newPopulation, float* fitness, int populationSize, int chromosomeSize, int elitismCount);

// Fitness functions
__host__ __device__ float sphereFunction(float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

__host__ __device__ float rastriginFunction(float* x, int n) {
    float sum = 10.0f * n;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i] - 10.0f * cosf(2.0f * M_PI * x[i]);
    }
    return sum;
}

__host__ __device__ float rosenbrockFunction(float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n - 1; i++) {
        sum += 100.0f * powf(x[i+1] - x[i] * x[i], 2.0f) + powf(x[i] - 1.0f, 2.0f);
    }
    return sum;
}

__host__ __device__ float ackleyFunction(float* x, int n) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int i = 0; i < n; i++) {
        sum1 += x[i] * x[i];
        sum2 += cosf(2.0f * M_PI * x[i]);
    }
    sum1 = -0.2f * sqrtf(sum1 / n);
    sum2 = sum2 / n;
    return -20.0f * expf(sum1) - expf(sum2) + 20.0f + M_E;
}

__host__ __device__ float griewankFunction(float* x, int n) {
    float sum = 0.0f;
    float product = 1.0f;
    for (int i = 0; i < n; i++) {
        sum += (x[i] * x[i]) / 4000.0f;
        product *= cosf(x[i] / sqrtf(i + 1.0f));
    }
    return 1.0f + sum - product;
}

// Main function
int main(int argc, char** argv) {
    // Default parameters
    GAParams params;
    params.populationSize = DEFAULT_POPULATION_SIZE;
    params.chromosomeSize = DEFAULT_CHROMOSOME_SIZE;
    params.generations = DEFAULT_GENERATIONS;
    params.crossoverRate = DEFAULT_CROSSOVER_RATE;
    params.mutationRate = DEFAULT_MUTATION_RATE;
    params.tournamentSize = DEFAULT_TOURNAMENT_SIZE;
    params.elitismCount = DEFAULT_ELITISM_COUNT;
    params.minValue = DEFAULT_MIN_VALUE;
    params.maxValue = DEFAULT_MAX_VALUE;
    params.functionType = SPHERE;

    // Parse command line arguments
    int opt;
    static struct option long_options[] = {
        {"population", required_argument, 0, 'p'},
        {"chromosome", required_argument, 0, 'c'},
        {"generations", required_argument, 0, 'g'},
        {"crossover", required_argument, 0, 'x'},
        {"mutation", required_argument, 0, 'm'},
        {"tournament", required_argument, 0, 't'},
        {"elitism", required_argument, 0, 'e'},
        {"min", required_argument, 0, 'n'},
        {"max", required_argument, 0, 'a'},
        {"function", required_argument, 0, 'f'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "p:c:g:x:m:t:e:n:a:f:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'p':
                params.populationSize = atoi(optarg);
                break;
            case 'c':
                params.chromosomeSize = atoi(optarg);
                break;
            case 'g':
                params.generations = atoi(optarg);
                break;
            case 'x':
                params.crossoverRate = atof(optarg);
                break;
            case 'm':
                params.mutationRate = atof(optarg);
                break;
            case 't':
                params.tournamentSize = atoi(optarg);
                break;
            case 'e':
                params.elitismCount = atoi(optarg);
                break;
            case 'n':
                params.minValue = atof(optarg);
                break;
            case 'a':
                params.maxValue = atof(optarg);
                break;
            case 'f':
                if (strcmp(optarg, "sphere") == 0) {
                    params.functionType = SPHERE;
                } else if (strcmp(optarg, "rastrigin") == 0) {
                    params.functionType = RASTRIGIN;
                } else if (strcmp(optarg, "rosenbrock") == 0) {
                    params.functionType = ROSENBROCK;
                } else if (strcmp(optarg, "ackley") == 0) {
                    params.functionType = ACKLEY;
                } else if (strcmp(optarg, "griewank") == 0) {
                    params.functionType = GRIEWANK;
                } else {
                    fprintf(stderr, "Unknown function type: %s\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -p, --population=SIZE    Population size (default: %d)\n", DEFAULT_POPULATION_SIZE);
                printf("  -c, --chromosome=SIZE    Chromosome size (default: %d)\n", DEFAULT_CHROMOSOME_SIZE);
                printf("  -g, --generations=NUM    Number of generations (default: %d)\n", DEFAULT_GENERATIONS);
                printf("  -x, --crossover=RATE     Crossover rate (default: %.2f)\n", DEFAULT_CROSSOVER_RATE);
                printf("  -m, --mutation=RATE      Mutation rate (default: %.2f)\n", DEFAULT_MUTATION_RATE);
                printf("  -t, --tournament=SIZE    Tournament size (default: %d)\n", DEFAULT_TOURNAMENT_SIZE);
                printf("  -e, --elitism=COUNT      Elitism count (default: %d)\n", DEFAULT_ELITISM_COUNT);
                printf("  -n, --min=VALUE          Minimum gene value (default: %.2f)\n", DEFAULT_MIN_VALUE);
                printf("  -a, --max=VALUE          Maximum gene value (default: %.2f)\n", DEFAULT_MAX_VALUE);
                printf("  -f, --function=TYPE      Function type (sphere, rastrigin, rosenbrock, ackley, griewank) (default: sphere)\n");
                printf("  -h, --help               Show this help message\n");
                exit(EXIT_SUCCESS);
            default:
                fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    // Print parameters
    printf("Genetic Algorithm Parameters:\n");
    printf("  Population Size: %d\n", params.populationSize);
    printf("  Chromosome Size: %d\n", params.chromosomeSize);
    printf("  Generations: %d\n", params.generations);
    printf("  Crossover Rate: %.2f\n", params.crossoverRate);
    printf("  Mutation Rate: %.2f\n", params.mutationRate);
    printf("  Tournament Size: %d\n", params.tournamentSize);
    printf("  Elitism Count: %d\n", params.elitismCount);
    printf("  Min Value: %.2f\n", params.minValue);
    printf("  Max Value: %.2f\n", params.maxValue);
    printf("  Function: %s\n", getFunctionName(params.functionType));
    printf("\n");

    // Run CPU version
    printf("Running CPU implementation...\n");
    runGeneticAlgorithmCPU(params);

    // Run GPU version
    printf("\nRunning GPU implementation...\n");
    runGeneticAlgorithmGPU(params);

    return 0;
}

// Get function name
const char* getFunctionName(FunctionType functionType) {
    switch (functionType) {
        case SPHERE:
            return "Sphere";
        case RASTRIGIN:
            return "Rastrigin";
        case ROSENBROCK:
            return "Rosenbrock";
        case ACKLEY:
            return "Ackley";
        case GRIEWANK:
            return "Griewank";
        default:
            return "Unknown";
    }
}

// CPU implementation of the genetic algorithm
void runGeneticAlgorithmCPU(GAParams params) {
    clock_t start, end;
    double cpu_time_used;

    start = clock();

    // Initialize population
    Chromosome* population = (Chromosome*)malloc(params.populationSize * sizeof(Chromosome));
    for (int i = 0; i < params.populationSize; i++) {
        population[i].genes = (float*)malloc(params.chromosomeSize * sizeof(float));
    }
    initializePopulation(population, params.populationSize, params.chromosomeSize, params.minValue, params.maxValue);

    // Evaluate initial population
    for (int i = 0; i < params.populationSize; i++) {
        population[i].fitness = evaluateFitness(population[i].genes, params.chromosomeSize, params.functionType, params.minValue, params.maxValue);
    }

    // Print initial best solution
    printBestSolution(population, params.populationSize, 0, params.functionType);

    // Allocate memory for offspring
    Chromosome* offspring = (Chromosome*)malloc(params.populationSize * sizeof(Chromosome));
    for (int i = 0; i < params.populationSize; i++) {
        offspring[i].genes = (float*)malloc(params.chromosomeSize * sizeof(float));
    }

    // Main loop
    for (int generation = 1; generation <= params.generations; generation++) {
        // Elitism: Copy best individuals to offspring
        for (int e = 0; e < params.elitismCount; e++) {
            // Find eth best individual
            int bestIdx = 0;
            float bestFitness = population[0].fitness;
            for (int i = 1; i < params.populationSize; i++) {
                if (population[i].fitness < bestFitness) {
                    bestFitness = population[i].fitness;
                    bestIdx = i;
                }
            }
            
            // Copy to offspring
            memcpy(offspring[e].genes, population[bestIdx].genes, params.chromosomeSize * sizeof(float));
            offspring[e].fitness = population[bestIdx].fitness;
            
            // Set fitness to max to exclude from next search
            population[bestIdx].fitness = FLT_MAX;
        }
        
        // Reset fitness values
        for (int i = 0; i < params.populationSize; i++) {
            if (population[i].fitness == FLT_MAX) {
                population[i].fitness = evaluateFitness(population[i].genes, params.chromosomeSize, params.functionType, params.minValue, params.maxValue);
            }
        }

        // Create offspring
        for (int i = params.elitismCount; i < params.populationSize; i += 2) {
            // Select parents
            int parent1Idx = tournamentSelection(population, params.populationSize, params.tournamentSize);
            int parent2Idx = tournamentSelection(population, params.populationSize, params.tournamentSize);
            
            // Create offspring
            crossover(population, parent1Idx, parent2Idx, offspring[i].genes, 
                     (i + 1 < params.populationSize) ? offspring[i + 1].genes : NULL, 
                     params.chromosomeSize, params.crossoverRate);
            
            // Mutate offspring
            mutate(offspring[i].genes, params.chromosomeSize, params.mutationRate, params.minValue, params.maxValue);
            if (i + 1 < params.populationSize) {
                mutate(offspring[i + 1].genes, params.chromosomeSize, params.mutationRate, params.minValue, params.maxValue);
            }
        }
        
        // Evaluate offspring
        for (int i = params.elitismCount; i < params.populationSize; i++) {
            offspring[i].fitness = evaluateFitness(offspring[i].genes, params.chromosomeSize, params.functionType, params.minValue, params.maxValue);
        }
        
        // Replace population with offspring
        for (int i = 0; i < params.populationSize; i++) {
            memcpy(population[i].genes, offspring[i].genes, params.chromosomeSize * sizeof(float));
            population[i].fitness = offspring[i].fitness;
        }
        
        // Print best solution every 10 generations
        if (generation % 10 == 0 || generation == params.generations) {
            printBestSolution(population, params.populationSize, generation, params.functionType);
        }
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU execution time: %.4f seconds\n", cpu_time_used);

    // Free memory
    freePopulation(population, params.populationSize);
    freePopulation(offspring, params.populationSize);
}

// GPU implementation of the genetic algorithm
void runGeneticAlgorithmGPU(GAParams params) {
    cudaEvent_t start, stop;
    float gpu_time_used;
    
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    // Allocate memory on device
    float *d_population, *d_newPopulation, *d_fitness;
    int *d_selected;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_population, params.populationSize * params.chromosomeSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_newPopulation, params.populationSize * params.chromosomeSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_fitness, params.populationSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_selected, params.populationSize * sizeof(int)));

    // Initialize population on device
    int blockSize = 256;
    int gridSize = (params.populationSize + blockSize - 1) / blockSize;
    
    unsigned int seed = time(NULL);
    initializePopulationKernel<<<gridSize, blockSize>>>(d_population, params.populationSize, params.chromosomeSize, params.minValue, params.maxValue, seed);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Evaluate initial population
    evaluateFitnessKernel<<<gridSize, blockSize>>>(d_population, d_fitness, params.populationSize, params.chromosomeSize, params.functionType, params.minValue, params.maxValue);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Allocate memory for best solution on host
    float *h_bestSolution = (float*)malloc(params.chromosomeSize * sizeof(float));
    float h_bestFitness;

    // Copy best solution to host and print
    float *h_fitness = (float*)malloc(params.populationSize * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(h_fitness, d_fitness, params.populationSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    int bestIdx = 0;
    for (int i = 1; i < params.populationSize; i++) {
        if (h_fitness[i] < h_fitness[bestIdx]) {
            bestIdx = i;
        }
    }
    
    float *h_population = (float*)malloc(params.populationSize * params.chromosomeSize * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(h_population, d_population, params.populationSize * params.chromosomeSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    memcpy(h_bestSolution, &h_population[bestIdx * params.chromosomeSize], params.chromosomeSize * sizeof(float));
    h_bestFitness = h_fitness[bestIdx];
    
    printf("Generation 0: Best Fitness = %.6f\n", h_bestFitness);

    // Main loop
    for (int generation = 1; generation <= params.generations; generation++) {
        // Selection
        tournamentSelectionKernel<<<gridSize, blockSize>>>(d_fitness, d_selected, params.populationSize, params.tournamentSize, seed + generation);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Crossover and mutation
        crossoverAndMutateKernel<<<gridSize, blockSize>>>(d_population, d_newPopulation, d_selected, params.populationSize, params.chromosomeSize, params.crossoverRate, params.mutationRate, params.minValue, params.maxValue, seed + generation * 2);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Elitism
        elitismKernel<<<1, 1>>>(d_population, d_newPopulation, d_fitness, params.populationSize, params.chromosomeSize, params.elitismCount);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Swap populations
        float *temp = d_population;
        d_population = d_newPopulation;
        d_newPopulation = temp;
        
        // Evaluate new population
        evaluateFitnessKernel<<<gridSize, blockSize>>>(d_population, d_fitness, params.populationSize, params.chromosomeSize, params.functionType, params.minValue, params.maxValue);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Print best solution every 10 generations
        if (generation % 10 == 0 || generation == params.generations) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_fitness, d_fitness, params.populationSize * sizeof(float), cudaMemcpyDeviceToHost));
            
            bestIdx = 0;
            for (int i = 1; i < params.populationSize; i++) {
                if (h_fitness[i] < h_fitness[bestIdx]) {
                    bestIdx = i;
                }
            }
            
            CHECK_CUDA_ERROR(cudaMemcpy(h_population, d_population, params.populationSize * params.chromosomeSize * sizeof(float), cudaMemcpyDeviceToHost));
            
            memcpy(h_bestSolution, &h_population[bestIdx * params.chromosomeSize], params.chromosomeSize * sizeof(float));
            h_bestFitness = h_fitness[bestIdx];
            
            printf("Generation %d: Best Fitness = %.6f\n", generation, h_bestFitness);
        }
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_used, start, stop));
    
    printf("GPU execution time: %.4f seconds\n", gpu_time_used / 1000.0f);
    
    // Print final solution
    printf("\nFinal Solution:\n");
    printf("  Fitness: %.6f\n", h_bestFitness);
    printf("  Genes: [");
    for (int i = 0; i < params.chromosomeSize; i++) {
        printf("%.6f", h_bestSolution[i]);
        if (i < params.chromosomeSize - 1) {
            printf(", ");
        }
    }
    printf("]\n");

    // Free memory
    free(h_bestSolution);
    free(h_fitness);
    free(h_population);
    CHECK_CUDA_ERROR(cudaFree(d_population));
    CHECK_CUDA_ERROR(cudaFree(d_newPopulation));
    CHECK_CUDA_ERROR(cudaFree(d_fitness));
    CHECK_CUDA_ERROR(cudaFree(d_selected));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

// Initialize population
void initializePopulation(Chromosome* population, int populationSize, int chromosomeSize, float minValue, float maxValue) {
    for (int i = 0; i < populationSize; i++) {
        for (int j = 0; j < chromosomeSize; j++) {
            population[i].genes[j] = minValue + (maxValue - minValue) * ((float)rand() / RAND_MAX);
        }
    }
}

// Free population
void freePopulation(Chromosome* population, int populationSize) {
    for (int i = 0; i < populationSize; i++) {
        free(population[i].genes);
    }
    free(population);
}

// Evaluate fitness
float evaluateFitness(float* chromosome, int size, FunctionType functionType, float minValue, float maxValue) {
    switch (functionType) {
        case SPHERE:
            return sphereFunction(chromosome, size);
        case RASTRIGIN:
            return rastriginFunction(chromosome, size);
        case ROSENBROCK:
            return rosenbrockFunction(chromosome, size);
        case ACKLEY:
            return ackleyFunction(chromosome, size);
        case GRIEWANK:
            return griewankFunction(chromosome, size);
        default:
            return 0.0f;
    }
}

// Tournament selection
int tournamentSelection(Chromosome* population, int populationSize, int tournamentSize) {
    int bestIdx = rand() % populationSize;
    float bestFitness = population[bestIdx].fitness;
    
    for (int i = 1; i < tournamentSize; i++) {
        int idx = rand() % populationSize;
        if (population[idx].fitness < bestFitness) {
            bestFitness = population[idx].fitness;
            bestIdx = idx;
        }
    }
    
    return bestIdx;
}

// Crossover
void crossover(Chromosome* population, int parent1Idx, int parent2Idx, float* child1, float* child2, int chromosomeSize, float crossoverRate) {
    if ((float)rand() / RAND_MAX < crossoverRate) {
        // Single-point crossover
        int crossoverPoint = rand() % chromosomeSize;
        
        for (int i = 0; i < crossoverPoint; i++) {
            child1[i] = population[parent1Idx].genes[i];
            if (child2 != NULL) {
                child2[i] = population[parent2Idx].genes[i];
            }
        }
        
        for (int i = crossoverPoint; i < chromosomeSize; i++) {
            child1[i] = population[parent2Idx].genes[i];
            if (child2 != NULL) {
                child2[i] = population[parent1Idx].genes[i];
            }
        }
    } else {
        // No crossover, just copy parents
        memcpy(child1, population[parent1Idx].genes, chromosomeSize * sizeof(float));
        if (child2 != NULL) {
            memcpy(child2, population[parent2Idx].genes, chromosomeSize * sizeof(float));
        }
    }
}

// Mutation
void mutate(float* chromosome, int chromosomeSize, float mutationRate, float minValue, float maxValue) {
    for (int i = 0; i < chromosomeSize; i++) {
        if ((float)rand() / RAND_MAX < mutationRate) {
            chromosome[i] = minValue + (maxValue - minValue) * ((float)rand() / RAND_MAX);
        }
    }
}

// Print best solution
void printBestSolution(Chromosome* population, int populationSize, int generation, FunctionType functionType) {
    int bestIdx = 0;
    float bestFitness = population[0].fitness;
    
    for (int i = 1; i < populationSize; i++) {
        if (population[i].fitness < bestFitness) {
            bestFitness = population[i].fitness;
            bestIdx = i;
        }
    }
    
    printf("Generation %d: Best Fitness = %.6f\n", generation, bestFitness);
}

// CUDA kernel implementations
__device__ float evaluateFitnessDevice(float* chromosome, int size, FunctionType functionType, float minValue, float maxValue) {
    switch (functionType) {
        case SPHERE:
            return sphereFunction(chromosome, size);
        case RASTRIGIN:
            return rastriginFunction(chromosome, size);
        case ROSENBROCK:
            return rosenbrockFunction(chromosome, size);
        case ACKLEY:
            return ackleyFunction(chromosome, size);
        case GRIEWANK:
            return griewankFunction(chromosome, size);
        default:
            return 0.0f;
    }
}

__global__ void initializePopulationKernel(float* population, int populationSize, int chromosomeSize, float minValue, float maxValue, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        
        for (int j = 0; j < chromosomeSize; j++) {
            population[idx * chromosomeSize + j] = minValue + (maxValue - minValue) * curand_uniform(&state);
        }
    }
}

__global__ void evaluateFitnessKernel(float* population, float* fitness, int populationSize, int chromosomeSize, FunctionType functionType, float minValue, float maxValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        fitness[idx] = evaluateFitnessDevice(&population[idx * chromosomeSize], chromosomeSize, functionType, minValue, maxValue);
    }
}

__global__ void tournamentSelectionKernel(float* fitness, int* selected, int populationSize, int tournamentSize, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        
        int bestIdx = curand(&state) % populationSize;
        float bestFitness = fitness[bestIdx];
        
        for (int i = 1; i < tournamentSize; i++) {
            int randIdx = curand(&state) % populationSize;
            if (fitness[randIdx] < bestFitness) {
                bestFitness = fitness[randIdx];
                bestIdx = randIdx;
            }
        }
        
        selected[idx] = bestIdx;
    }
}

__global__ void crossoverAndMutateKernel(float* population, float* newPopulation, int* selected, int populationSize, int chromosomeSize, float crossoverRate, float mutationRate, float minValue, float maxValue, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize / 2) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        
        int parent1Idx = selected[idx * 2];
        int parent2Idx = selected[idx * 2 + 1];
        
        int child1Idx = idx * 2;
        int child2Idx = idx * 2 + 1;
        
        // Perform crossover
        if (curand_uniform(&state) < crossoverRate) {
            // Single-point crossover
            int crossoverPoint = curand(&state) % chromosomeSize;
            
            for (int i = 0; i < crossoverPoint; i++) {
                newPopulation[child1Idx * chromosomeSize + i] = population[parent1Idx * chromosomeSize + i];
                if (child2Idx < populationSize) {
                    newPopulation[child2Idx * chromosomeSize + i] = population[parent2Idx * chromosomeSize + i];
                }
            }
            
            for (int i = crossoverPoint; i < chromosomeSize; i++) {
                newPopulation[child1Idx * chromosomeSize + i] = population[parent2Idx * chromosomeSize + i];
                if (child2Idx < populationSize) {
                    newPopulation[child2Idx * chromosomeSize + i] = population[parent1Idx * chromosomeSize + i];
                }
            }
        } else {
            // No crossover, just copy parents
            for (int i = 0; i < chromosomeSize; i++) {
                newPopulation[child1Idx * chromosomeSize + i] = population[parent1Idx * chromosomeSize + i];
                if (child2Idx < populationSize) {
                    newPopulation[child2Idx * chromosomeSize + i] = population[parent2Idx * chromosomeSize + i];
                }
            }
        }
        
        // Perform mutation
        for (int i = 0; i < chromosomeSize; i++) {
            if (curand_uniform(&state) < mutationRate) {
                newPopulation[child1Idx * chromosomeSize + i] = minValue + (maxValue - minValue) * curand_uniform(&state);
            }
            
            if (child2Idx < populationSize) {
                if (curand_uniform(&state) < mutationRate) {
                    newPopulation[child2Idx * chromosomeSize + i] = minValue + (maxValue - minValue) * curand_uniform(&state);
                }
            }
        }
    }
}

__global__ void elitismKernel(float* population, float* newPopulation, float* fitness, int populationSize, int chromosomeSize, int elitismCount) {
    // This kernel is designed to run with a single thread
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int e = 0; e < elitismCount; e++) {
            // Find eth best individual
            int bestIdx = 0;
            float bestFitness = fitness[0];
            
            for (int i = 1; i < populationSize; i++) {
                if (fitness[i] < bestFitness) {
                    bestFitness = fitness[i];
                    bestIdx = i;
                }
            }
            
            // Copy to new population
            for (int i = 0; i < chromosomeSize; i++) {
                newPopulation[e * chromosomeSize + i] = population[bestIdx * chromosomeSize + i];
            }
            
            // Set fitness to max to exclude from next search
            fitness[bestIdx] = FLT_MAX;
        }
    }
}
