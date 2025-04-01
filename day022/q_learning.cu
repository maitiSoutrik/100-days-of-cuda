#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <unistd.h>
#include <getopt.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Constants
#define BLOCK_SIZE 256
#define MAX_GRID_SIZE 100
#define MAX_STEPS 1000

// Actions
#define UP 0
#define RIGHT 1
#define DOWN 2
#define LEFT 3
#define NUM_ACTIONS 4

// Cell types
#define EMPTY 0
#define OBSTACLE 1
#define GOAL 2

// Q-learning parameters
typedef struct {
    int gridSize;
    int numAgents;
    int numEpisodes;
    float learningRate;
    float discountFactor;
    float epsilon;
    int maxSteps;
    bool cpuOnly;
    bool useAscii;  // Use ASCII-only visualization
} QParams;

// Grid world environment
typedef struct {
    int* grid;
    int gridSize;
    int startX;
    int startY;
    int goalX;
    int goalY;
} GridWorld;

// Agent state
typedef struct {
    int x;
    int y;
    int steps;
    float totalReward;
    bool done;
} AgentState;

// Function to create a grid world environment
GridWorld createGridWorld(int gridSize) {
    GridWorld env;
    env.gridSize = gridSize;
    
    // Allocate memory for the grid
    env.grid = (int*)malloc(gridSize * gridSize * sizeof(int));
    
    // Initialize grid with empty cells
    for (int i = 0; i < gridSize * gridSize; i++) {
        env.grid[i] = EMPTY;
    }
    
    // Add obstacles (simple pattern for demonstration)
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            // Create a maze-like pattern
            if ((i % 3 == 0 && j % 4 != 0) || (j % 3 == 0 && i % 4 != 0)) {
                if (i > 0 && j > 0 && i < gridSize-1 && j < gridSize-1) {
                    env.grid[i * gridSize + j] = OBSTACLE;
                }
            }
        }
    }
    
    // Set start position (top-left corner)
    env.startX = 0;
    env.startY = 0;
    
    // Set goal position (bottom-right corner)
    env.goalX = gridSize - 1;
    env.goalY = gridSize - 1;
    
    // Ensure start and goal are empty
    env.grid[env.startY * gridSize + env.startX] = EMPTY;
    env.grid[env.goalY * gridSize + env.goalX] = GOAL;
    
    return env;
}

// Function to free grid world resources
void freeGridWorld(GridWorld* env) {
    free(env->grid);
}

// Function to reset an agent to initial state
void resetAgent(AgentState* agent, GridWorld* env) {
    agent->x = env->startX;
    agent->y = env->startY;
    agent->steps = 0;
    agent->totalReward = 0.0f;
    agent->done = false;
}

// Function to get state index from agent position
int getStateIndex(AgentState* agent, int gridSize) {
    return agent->y * gridSize + agent->x;
}

// Function to take an action and update agent state
float takeAction(AgentState* agent, GridWorld* env, int action) {
    int newX = agent->x;
    int newY = agent->y;
    
    // Update position based on action
    switch (action) {
        case UP:
            newY = max(0, agent->y - 1);
            break;
        case RIGHT:
            newX = min(env->gridSize - 1, agent->x + 1);
            break;
        case DOWN:
            newY = min(env->gridSize - 1, agent->y + 1);
            break;
        case LEFT:
            newX = max(0, agent->x - 1);
            break;
    }
    
    // Check if new position is an obstacle
    if (env->grid[newY * env->gridSize + newX] == OBSTACLE) {
        return -10.0f;  // Penalty for hitting an obstacle
    }
    
    // Update agent position
    agent->x = newX;
    agent->y = newY;
    agent->steps++;
    
    // Check if agent reached the goal
    if (env->grid[newY * env->gridSize + newX] == GOAL) {
        agent->done = true;
        return 100.0f;  // Reward for reaching the goal
    }
    
    // Check if maximum steps reached
    if (agent->steps >= MAX_STEPS) {
        agent->done = true;
    }
    
    return -1.0f;  // Small penalty for each step to encourage finding shortest path
}

// Function to choose an action using epsilon-greedy policy
int chooseAction(float* qTable, int stateIndex, int numActions, float epsilon, float* randVal) {
    // Exploration: choose a random action with probability epsilon
    if (*randVal < epsilon) {
        return (int)(*randVal * 4 / epsilon);  // Map [0,epsilon] to [0,4)
    }
    
    // Exploitation: choose the action with the highest Q-value
    int bestAction = 0;
    float bestValue = qTable[stateIndex * numActions + 0];
    
    for (int a = 1; a < numActions; a++) {
        float value = qTable[stateIndex * numActions + a];
        if (value > bestValue) {
            bestValue = value;
            bestAction = a;
        }
    }
    
    return bestAction;
}

// CUDA kernel for initializing random states
__global__ void initRandomStates(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// CUDA kernel for running Q-learning episodes in parallel
__global__ void qLearningKernel(
    float* qTable,
    int* grid,
    int gridSize,
    int startX,
    int startY,
    int goalX,
    int goalY,
    int numEpisodes,
    float learningRate,
    float discountFactor,
    float epsilon,
    int maxSteps,
    curandState* randStates,
    float* episodeRewards,
    int* episodeSteps
) {
    int agentIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numAgents = gridDim.x * blockDim.x;
    int numStates = gridSize * gridSize;
    
    // Get thread-specific random state
    curandState localRandState = randStates[agentIdx];
    
    // Each thread handles multiple episodes
    for (int episodeBase = 0; episodeBase < numEpisodes; episodeBase += numAgents) {
        int episodeIdx = episodeBase + agentIdx;
        
        // Skip if we've processed all episodes
        if (episodeIdx >= numEpisodes) break;
        
        // Initialize agent state
        int x = startX;
        int y = startY;
        int steps = 0;
        float totalReward = 0.0f;
        bool done = false;
        
        // Run episode
        while (!done && steps < maxSteps) {
            // Get current state index
            int stateIdx = y * gridSize + x;
            
            // Choose action using epsilon-greedy policy
            float randVal = curand_uniform(&localRandState);
            int action;
            
            // Exploration: choose a random action with probability epsilon
            if (randVal < epsilon) {
                action = (int)(curand_uniform(&localRandState) * 4);
            } else {
                // Exploitation: choose the action with the highest Q-value
                int bestAction = 0;
                float bestValue = qTable[stateIdx * NUM_ACTIONS + 0];
                
                for (int a = 1; a < NUM_ACTIONS; a++) {
                    float value = qTable[stateIdx * NUM_ACTIONS + a];
                    if (value > bestValue) {
                        bestValue = value;
                        bestAction = a;
                    }
                }
                
                action = bestAction;
            }
            
            // Take action and get reward
            int newX = x;
            int newY = y;
            float reward = 0.0f;
            
            // Update position based on action
            switch (action) {
                case UP:
                    newY = max(0, y - 1);
                    break;
                case RIGHT:
                    newX = min(gridSize - 1, x + 1);
                    break;
                case DOWN:
                    newY = min(gridSize - 1, y + 1);
                    break;
                case LEFT:
                    newX = max(0, x - 1);
                    break;
            }
            
            // Check if new position is an obstacle
            if (grid[newY * gridSize + newX] == OBSTACLE) {
                reward = -10.0f;  // Penalty for hitting an obstacle
            } else {
                // Update agent position
                x = newX;
                y = newY;
                
                // Check if agent reached the goal
                if (grid[y * gridSize + x] == GOAL) {
                    reward = 100.0f;  // Reward for reaching the goal
                    done = true;
                } else {
                    reward = -1.0f;  // Small penalty for each step
                }
            }
            
            // Update total reward
            totalReward += reward;
            
            // Get next state index
            int nextStateIdx = y * gridSize + x;
            
            // Find maximum Q-value for next state
            float maxNextQ = qTable[nextStateIdx * NUM_ACTIONS + 0];
            for (int a = 1; a < NUM_ACTIONS; a++) {
                float value = qTable[nextStateIdx * NUM_ACTIONS + a];
                if (value > maxNextQ) {
                    maxNextQ = value;
                }
            }
            
            // Update Q-value using Bellman equation
            float oldQ = qTable[stateIdx * NUM_ACTIONS + action];
            float newQ = oldQ + learningRate * (reward + discountFactor * maxNextQ - oldQ);
            
            // Update Q-table (using atomic operation to handle concurrent updates)
            atomicExch((unsigned int*)&qTable[stateIdx * NUM_ACTIONS + action], __float_as_int(newQ));
            
            // Increment step counter
            steps++;
            
            // Check if maximum steps reached
            if (steps >= maxSteps) {
                done = true;
            }
        }
        
        // Record episode statistics
        episodeRewards[episodeIdx] = totalReward;
        episodeSteps[episodeIdx] = steps;
    }
    
    // Save updated random state
    randStates[agentIdx] = localRandState;
}

// Function to run Q-learning on CPU
void runQlearningCPU(
    float* qTable,
    GridWorld* env,
    int numEpisodes,
    float learningRate,
    float discountFactor,
    float epsilon,
    int maxSteps,
    float* episodeRewards,
    int* episodeSteps
) {
    int numStates = env->gridSize * env->gridSize;
    AgentState agent;
    
    // Initialize random seed
    srand(time(NULL));
    
    // Run episodes
    for (int episode = 0; episode < numEpisodes; episode++) {
        // Reset agent
        resetAgent(&agent, env);
        
        // Run episode
        while (!agent.done && agent.steps < maxSteps) {
            // Get current state index
            int stateIdx = getStateIndex(&agent, env->gridSize);
            
            // Choose action using epsilon-greedy policy
            float randVal = (float)rand() / RAND_MAX;
            int action = chooseAction(qTable, stateIdx, NUM_ACTIONS, epsilon, &randVal);
            
            // Take action and get reward
            float reward = takeAction(&agent, env, action);
            
            // Get next state index
            int nextStateIdx = getStateIndex(&agent, env->gridSize);
            
            // Find maximum Q-value for next state
            float maxNextQ = qTable[nextStateIdx * NUM_ACTIONS + 0];
            for (int a = 1; a < NUM_ACTIONS; a++) {
                float value = qTable[nextStateIdx * NUM_ACTIONS + a];
                if (value > maxNextQ) {
                    maxNextQ = value;
                }
            }
            
            // Update Q-value using Bellman equation
            float oldQ = qTable[stateIdx * NUM_ACTIONS + action];
            float newQ = oldQ + learningRate * (reward + discountFactor * maxNextQ - oldQ);
            qTable[stateIdx * NUM_ACTIONS + action] = newQ;
        }
        
        // Record episode statistics
        episodeRewards[episode] = agent.totalReward;
        episodeSteps[episode] = agent.steps;
        
        // Print progress
        if (episode % 100 == 0 || episode == numEpisodes - 1) {
            printf("CPU - Episode %d: Steps = %d, Reward = %.2f\n", 
                   episode, agent.steps, agent.totalReward);
        }
    }
}

// Function to run Q-learning on GPU
void runQlearningGPU(
    float* qTable,
    GridWorld* env,
    QParams* params,
    float* episodeRewards,
    int* episodeSteps
) {
    int numStates = env->gridSize * env->gridSize;
    int numAgents = params->numAgents;
    
    // Allocate device memory
    float* d_qTable;
    int* d_grid;
    float* d_episodeRewards;
    int* d_episodeSteps;
    curandState* d_randStates;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_qTable, numStates * NUM_ACTIONS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid, env->gridSize * env->gridSize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_episodeRewards, params->numEpisodes * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_episodeSteps, params->numEpisodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_randStates, numAgents * sizeof(curandState)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_qTable, qTable, numStates * NUM_ACTIONS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_grid, env->grid, env->gridSize * env->gridSize * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize random states
    int numBlocks = (numAgents + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initRandomStates<<<numBlocks, BLOCK_SIZE>>>(d_randStates, time(NULL));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Launch kernel
    qLearningKernel<<<numBlocks, BLOCK_SIZE>>>(
        d_qTable,
        d_grid,
        env->gridSize,
        env->startX,
        env->startY,
        env->goalX,
        env->goalY,
        params->numEpisodes,
        params->learningRate,
        params->discountFactor,
        params->epsilon,
        params->maxSteps,
        d_randStates,
        d_episodeRewards,
        d_episodeSteps
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(qTable, d_qTable, numStates * NUM_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(episodeRewards, d_episodeRewards, params->numEpisodes * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(episodeSteps, d_episodeSteps, params->numEpisodes * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_qTable));
    CHECK_CUDA_ERROR(cudaFree(d_grid));
    CHECK_CUDA_ERROR(cudaFree(d_episodeRewards));
    CHECK_CUDA_ERROR(cudaFree(d_episodeSteps));
    CHECK_CUDA_ERROR(cudaFree(d_randStates));
}

// Function to visualize the learned policy
void visualizePolicy(float* qTable, GridWorld* env, bool useAscii = true) {
    printf("\nLearned Policy:\n");
    printf("  ");
    for (int x = 0; x < env->gridSize; x++) {
        printf("--");
    }
    printf("\n");
    
    for (int y = 0; y < env->gridSize; y++) {
        printf("| ");
        for (int x = 0; x < env->gridSize; x++) {
            int stateIdx = y * env->gridSize + x;
            int cellType = env->grid[stateIdx];
            
            if (cellType == OBSTACLE) {
                printf("# ");
            } else if (cellType == GOAL) {
                printf("G ");
            } else {
                // Find best action
                int bestAction = 0;
                float bestValue = qTable[stateIdx * NUM_ACTIONS + 0];
                
                for (int a = 1; a < NUM_ACTIONS; a++) {
                    float value = qTable[stateIdx * NUM_ACTIONS + a];
                    if (value > bestValue) {
                        bestValue = value;
                        bestAction = a;
                    }
                }
                
                // Print arrow for best action
                if (useAscii) {
                    // ASCII-only version for maximum compatibility
                    switch (bestAction) {
                        case UP:
                            printf("^ ");
                            break;
                        case RIGHT:
                            printf("> ");
                            break;
                        case DOWN:
                            printf("v ");
                            break;
                        case LEFT:
                            printf("< ");
                            break;
                    }
                } else {
                    // Unicode arrows for terminals that support it
                    switch (bestAction) {
                        case UP:
                            printf("↑ ");
                            break;
                        case RIGHT:
                            printf("→ ");
                            break;
                        case DOWN:
                            printf("↓ ");
                            break;
                        case LEFT:
                            printf("← ");
                            break;
                    }
                }
            }
        }
        printf("|\n");
    }
    
    printf("  ");
    for (int x = 0; x < env->gridSize; x++) {
        printf("--");
    }
    printf("\n");
}

// Function to parse command line arguments
void parseArgs(int argc, char** argv, QParams* params) {
    // Set default values
    params->gridSize = 10;
    params->numAgents = 256;
    params->numEpisodes = 1000;
    params->learningRate = 0.1f;
    params->discountFactor = 0.99f;
    params->epsilon = 0.1f;
    params->maxSteps = 1000;
    params->cpuOnly = false;
    params->useAscii = true;  // Default to ASCII for maximum compatibility
    
    // Define options
    static struct option long_options[] = {
        {"grid-size", required_argument, 0, 'g'},
        {"num-agents", required_argument, 0, 'a'},
        {"episodes", required_argument, 0, 'e'},
        {"learning-rate", required_argument, 0, 'l'},
        {"discount", required_argument, 0, 'd'},
        {"epsilon", required_argument, 0, 'p'},
        {"max-steps", required_argument, 0, 'm'},
        {"cpu-only", no_argument, 0, 'c'},
        {"unicode", no_argument, 0, 'u'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    // Parse options
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "g:a:e:l:d:p:m:cuh", long_options, &option_index)) != -1) {
        switch (c) {
            case 'g':
                params->gridSize = atoi(optarg);
                break;
            case 'a':
                params->numAgents = atoi(optarg);
                break;
            case 'e':
                params->numEpisodes = atoi(optarg);
                break;
            case 'l':
                params->learningRate = atof(optarg);
                break;
            case 'd':
                params->discountFactor = atof(optarg);
                break;
            case 'p':
                params->epsilon = atof(optarg);
                break;
            case 'm':
                params->maxSteps = atoi(optarg);
                break;
            case 'c':
                params->cpuOnly = true;
                break;
            case 'u':
                params->useAscii = false;  // Use Unicode arrows
                break;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  --grid-size SIZE       Set grid size (default: 10)\n");
                printf("  --num-agents NUM       Set number of parallel agents for GPU (default: 256)\n");
                printf("  --episodes NUM         Set number of episodes (default: 1000)\n");
                printf("  --learning-rate RATE   Set learning rate (default: 0.1)\n");
                printf("  --discount FACTOR      Set discount factor (default: 0.99)\n");
                printf("  --epsilon VALUE        Set exploration rate (default: 0.1)\n");
                printf("  --max-steps NUM        Set maximum steps per episode (default: 1000)\n");
                printf("  --cpu-only             Run only on CPU (default: false)\n");
                printf("  --unicode              Use Unicode arrows for visualization (default: ASCII)\n");
                printf("  --help                 Show this help message\n");
                exit(0);
                break;
            default:
                break;
        }
    }
    
    // Validate parameters
    if (params->gridSize <= 0 || params->gridSize > MAX_GRID_SIZE) {
        printf("Grid size must be between 1 and %d\n", MAX_GRID_SIZE);
        exit(1);
    }
    
    if (params->numAgents <= 0) {
        printf("Number of agents must be positive\n");
        exit(1);
    }
    
    if (params->numEpisodes <= 0) {
        printf("Number of episodes must be positive\n");
        exit(1);
    }
    
    if (params->learningRate <= 0.0f || params->learningRate > 1.0f) {
        printf("Learning rate must be between 0 and 1\n");
        exit(1);
    }
    
    if (params->discountFactor <= 0.0f || params->discountFactor > 1.0f) {
        printf("Discount factor must be between 0 and 1\n");
        exit(1);
    }
    
    if (params->epsilon < 0.0f || params->epsilon > 1.0f) {
        printf("Epsilon must be between 0 and 1\n");
        exit(1);
    }
    
    if (params->maxSteps <= 0) {
        printf("Maximum steps must be positive\n");
        exit(1);
    }
}

// Main function
int main(int argc, char** argv) {
    // Parse command line arguments
    QParams params;
    parseArgs(argc, argv, &params);
    
    // Print parameters
    printf("Q-Learning Parameters:\n");
    printf("  Grid Size: %d x %d\n", params.gridSize, params.gridSize);
    printf("  Number of Agents: %d\n", params.numAgents);
    printf("  Number of Episodes: %d\n", params.numEpisodes);
    printf("  Learning Rate: %.4f\n", params.learningRate);
    printf("  Discount Factor: %.4f\n", params.discountFactor);
    printf("  Exploration Rate (Epsilon): %.4f\n", params.epsilon);
    printf("  Maximum Steps per Episode: %d\n", params.maxSteps);
    printf("  Mode: %s\n", params.cpuOnly ? "CPU Only" : "CPU and GPU");
    printf("  Visualization: %s\n", params.useAscii ? "ASCII" : "Unicode");
    printf("\n");
    
    // Create grid world environment
    GridWorld env = createGridWorld(params.gridSize);
    int numStates = env.gridSize * env.gridSize;
    
    // Allocate memory for Q-table
    float* qTableCPU = (float*)malloc(numStates * NUM_ACTIONS * sizeof(float));
    float* qTableGPU = (float*)malloc(numStates * NUM_ACTIONS * sizeof(float));
    
    // Initialize Q-tables with zeros
    memset(qTableCPU, 0, numStates * NUM_ACTIONS * sizeof(float));
    memset(qTableGPU, 0, numStates * NUM_ACTIONS * sizeof(float));
    
    // Allocate memory for episode statistics
    float* episodeRewardsCPU = (float*)malloc(params.numEpisodes * sizeof(float));
    int* episodeStepsCPU = (int*)malloc(params.numEpisodes * sizeof(int));
    float* episodeRewardsGPU = (float*)malloc(params.numEpisodes * sizeof(float));
    int* episodeStepsGPU = (int*)malloc(params.numEpisodes * sizeof(int));
    
    // Run Q-learning on CPU
    printf("Running Q-learning on CPU...\n");
    auto cpuStart = std::chrono::high_resolution_clock::now();
    
    runQlearningCPU(
        qTableCPU,
        &env,
        params.numEpisodes,
        params.learningRate,
        params.discountFactor,
        params.epsilon,
        params.maxSteps,
        episodeRewardsCPU,
        episodeStepsCPU
    );
    
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;
    
    // Visualize CPU policy
    printf("\nCPU Policy:\n");
    visualizePolicy(qTableCPU, &env, params.useAscii);
    
    // Run Q-learning on GPU (if not CPU-only)
    double gpuTime = 0.0;
    
    if (!params.cpuOnly) {
        printf("\nRunning Q-learning on GPU...\n");
        auto gpuStart = std::chrono::high_resolution_clock::now();
        
        runQlearningGPU(
            qTableGPU,
            &env,
            &params,
            episodeRewardsGPU,
            episodeStepsGPU
        );
        
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gpuDuration = gpuEnd - gpuStart;
        gpuTime = gpuDuration.count();
        
        // Visualize GPU policy
        printf("\nGPU Policy:\n");
        visualizePolicy(qTableGPU, &env, params.useAscii);
    }
    
    // Print performance comparison
    printf("\nPerformance Comparison:\n");
    printf("  CPU Time: %.2f ms\n", cpuTime.count());
    
    if (!params.cpuOnly) {
        printf("  GPU Time: %.2f ms\n", gpuTime);
        printf("  Speedup: %.2fx\n", cpuTime.count() / gpuTime);
    }
    
    // Calculate average rewards and steps
    float avgRewardCPU = 0.0f;
    float avgStepsCPU = 0.0f;
    float avgRewardGPU = 0.0f;
    float avgStepsGPU = 0.0f;
    
    for (int i = 0; i < params.numEpisodes; i++) {
        avgRewardCPU += episodeRewardsCPU[i];
        avgStepsCPU += episodeStepsCPU[i];
        
        if (!params.cpuOnly) {
            avgRewardGPU += episodeRewardsGPU[i];
            avgStepsGPU += episodeStepsGPU[i];
        }
    }
    
    avgRewardCPU /= params.numEpisodes;
    avgStepsCPU /= params.numEpisodes;
    
    if (!params.cpuOnly) {
        avgRewardGPU /= params.numEpisodes;
        avgStepsGPU /= params.numEpisodes;
    }
    
    printf("\nTraining Results:\n");
    printf("  CPU Average Reward: %.2f\n", avgRewardCPU);
    printf("  CPU Average Steps: %.2f\n", avgStepsCPU);
    
    if (!params.cpuOnly) {
        printf("  GPU Average Reward: %.2f\n", avgRewardGPU);
        printf("  GPU Average Steps: %.2f\n", avgStepsGPU);
    }
    
    // Free resources
    free(qTableCPU);
    free(qTableGPU);
    free(episodeRewardsCPU);
    free(episodeStepsCPU);
    free(episodeRewardsGPU);
    free(episodeStepsGPU);
    freeGridWorld(&env);
    
    return 0;
}