#define RL_TOOLS_BACKEND_DISABLE_BLAS
#include <rl_tools/operations/cpu.h> // Agregado: Define los namespaces y math para que el .h no se queje

#include "../models/optimal_pilot_sac_seed_123_best.h"

#include <iostream>
#include <chrono>

// Función determinista que ejecuta el MLP (11 entradas -> 64 -> 64 -> 8 salidas)
void evaluar_politica(const float* input, float* output) {
    // Capa 1: Entrada (11) a Oculta (64) + ReLU
    float h1[64];
    const float* w1 = policy::layer_0::input_layer::weights::parameters_memory::memory;
    const float* b1 = policy::layer_0::input_layer::biases::parameters_memory::memory;
    for(int i = 0; i < 64; i++) {
        float sum = b1[i];
        for(int j = 0; j < 11; j++) sum += w1[i * 11 + j] * input[j];
        h1[i] = sum > 0 ? sum : 0; // Activación ReLU
    }

    // Capa 2: Oculta (64) a Oculta (64) + ReLU
    float h2[64];
    const float* w2 = policy::layer_0::hidden_layer_0::weights::parameters_memory::memory;
    const float* b2 = policy::layer_0::hidden_layer_0::biases::parameters_memory::memory;
    for(int i = 0; i < 64; i++) {
        float sum = b2[i];
        for(int j = 0; j < 64; j++) sum += w2[i * 64 + j] * h1[j];
        h2[i] = sum > 0 ? sum : 0; // Activación ReLU
    }

    // Capa 3: Oculta (64) a Salida (8) + Identidad (sin activación)
    const float* w3 = policy::layer_0::output_layer::weights::parameters_memory::memory;
    const float* b3 = policy::layer_0::output_layer::biases::parameters_memory::memory;
    for(int i = 0; i < 8; i++) {
        float sum = b3[i];
        for(int j = 0; j < 64; j++) sum += w3[i * 64 + j] * h2[j];
        output[i] = sum;
    }
}

int main() {
    float input[11] = {0.0f}; // Vector de estado de prueba
    float output[8] = {0.0f}; // 4 variables de media (acción) + 4 de log_std

    const int NUM_ITERACIONES = 100000;
    std::cout << "Iniciando benchmark de " << NUM_ITERACIONES << " iteraciones en el host...\n";

    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < NUM_ITERACIONES; i++) {
        evaluar_politica(input, output);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;

    double avg_time_us = duration.count() / NUM_ITERACIONES;

    std::cout << "Tiempo total: " << duration.count() << " microsegundos\n";
    std::cout << "Tiempo promedio por inferencia: " << avg_time_us << " microsegundos\n";

    return 0;
}
