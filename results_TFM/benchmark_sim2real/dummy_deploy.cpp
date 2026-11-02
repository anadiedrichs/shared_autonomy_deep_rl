// dummy_deploy.cpp

// 1. Desactivamos BLAS (librerías matemáticas pesadas no aptas para el microcontrolador)
#define RL_TOOLS_BACKEND_DISABLE_BLAS

// 2. Incluimos el núcleo de operaciones de RLtools PRIMERO
#include <rl_tools/operations/cpu.h>

// 3. AHORA SÍ, incluimos tu modelo exportado
#include "../models/optimal_pilot_sac_seed_123_best.h"

// 4. Forzamos la preservación de la memoria estática
volatile const void* keep_alive_pointer = &policy::module;

int main() {
    return 0;
}


