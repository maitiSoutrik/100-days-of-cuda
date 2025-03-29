# Day 20: Monte Carlo Option Pricing with CUDA

This project implements Monte Carlo simulations for pricing financial options using CUDA. It demonstrates how to leverage GPU parallelism for computationally intensive financial calculations.

## Overview

Monte Carlo methods are widely used in finance for option pricing, especially for complex options where closed-form analytical solutions don't exist. This implementation focuses on:

1. **European Options**: Options that can only be exercised at maturity
2. **Asian Options**: Options where the payoff depends on the average price of the underlying asset over a period of time

## Key CUDA Features Demonstrated

- **Parallel Random Number Generation**: Using cuRAND for efficient generation of random numbers on the GPU
- **Path Simulation**: Simulating thousands to millions of price paths in parallel
- **Reduction Operations**: Efficiently calculating the average option price across all simulations
- **Coalesced Memory Access**: Optimizing memory access patterns for better performance
- **Performance Scaling**: Demonstrating how performance scales with the number of simulations

## Implementation Details

### European Option Pricing

The European option pricing implementation:
- Uses the Black-Scholes model for path simulation
- Compares Monte Carlo results with the analytical Black-Scholes formula
- Demonstrates how accuracy improves with more simulations

### Asian Option Pricing

The Asian option pricing implementation:
- Simulates multiple time steps to calculate the average price path
- Demonstrates pricing for options without simple closed-form solutions
- Shows how to handle more complex path-dependent options

## Performance Considerations

The code includes:
- Timing measurements to show performance at different simulation counts
- Shared memory usage for efficient reduction operations
- Comparison of execution times as the number of simulations increases

## Usage

### European Option Pricing

```bash
./european_option_pricing [options]

Options:
  --S0 <value>     Initial stock price (default: 100.0)
  --K <value>      Strike price (default: 100.0)
  --r <value>      Risk-free interest rate (default: 0.05)
  --sigma <value>  Volatility (default: 0.2)
  --T <value>      Time to maturity in years (default: 1.0)
  --type <call|put> Option type (default: call)
```

### Asian Option Pricing

```bash
./asian_option_pricing [options]

Options:
  --S0 <value>     Initial stock price (default: 100.0)
  --K <value>      Strike price (default: 100.0)
  --r <value>      Risk-free interest rate (default: 0.05)
  --sigma <value>  Volatility (default: 0.2)
  --T <value>      Time to maturity in years (default: 1.0)
  --type <call|put> Option type (default: call)
```

## Mathematical Background

### European Option Pricing

For European options, the Black-Scholes formula provides an analytical solution:

For a call option:
C = S₀N(d₁) - Ke^(-rT)N(d₂)

For a put option:
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)

Where:
- d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
- d₂ = d₁ - σ√T
- N(x) is the cumulative distribution function of the standard normal distribution

### Monte Carlo Simulation

The Monte Carlo method simulates many random price paths according to:

S(T) = S₀ × exp[(r - σ²/2)T + σ√T × Z]

Where Z is a standard normal random variable.

For Asian options, we calculate the average price over multiple time steps.

## Future Improvements

Potential enhancements to this implementation:
- Support for more exotic option types (Barrier, Lookback, etc.)
- Implementation of variance reduction techniques
- Multi-GPU support for even larger simulations
- Interactive visualization of price paths and convergence