# Validación de despliegue Sim-to-Real (TinyRL): huella de memoria y benchmark de inferencia

Este directorio contiene las herramientas y scripts necesarios para validar empíricamente la viabilidad de alojar y ejecutar políticas entrenadas con **RLtools** sobre la arquitectura del nanocuadricóptero **Bitcraze Crazyflie 2.X** (microcontrolador STM32F405, ARM Cortex-M4F con 1 MB de Flash y 192 KB de RAM).

El proceso se divide en dos pruebas principales:
1. **Análisis de huella de memoria:** Compilación cruzada (*bare-metal*) para determinar el consumo exacto de memoria Flash (`.text`) y RAM (`.data`, `.bss`).
2. **Benchmark de tiempo de inferencia:** Emulación sobre arquitectura ARM mediante QEMU para medir el tiempo de ejecución determinista por paso de control.

---

## 1. Requisitos previos e instalación

En un entorno basado en Ubuntu/Debian, instale la cadena de herramientas de compilación cruzada para sistemas embebidos ARM (*bare-metal*), el compilador para Linux ARM y el emulador QEMU de espacio de usuario:

```bash
sudo apt update
sudo apt install build-essential git \
                 gcc-arm-none-eabi binutils-arm-none-eabi \
                 g++-arm-linux-gnueabihf qemu-user
```

### Clonar encabezados de RLtools
Dado que las políticas exportadas (`.h`) dependen de las estructuras base de RLtools, clone el repositorio oficial en el directorio de trabajo actual para proveer las cabeceras necesarias durante la compilación:

```bash
git clone https://github.com/rl-tools/rl-tools.git
```

---

## 2. Evaluación de la Huella de Memoria (Flash y RAM)

Para evitar que el compilador elimine los pesos de la red neuronal por optimización de código muerto (*dead code elimination*), se utiliza un contenedor mínimo que fuerza la preservación de la estructura en el binario final.

### 2.1. Crear el archivo contenedor (`dummy_deploy.cpp`)
Asegúrese de ajustar la ruta hacia su modelo exportado (`.h`):

```cpp
// dummy_deploy.cpp
#define RL_TOOLS_BACKEND_DISABLE_BLAS
#include <rl_tools/operations/cpu.h>

// Incluir el modelo exportado por RLtools
#include "../models/optimal_pilot_sac_seed_123_best.h"

// Puntero volátil para forzar al compilador a retener la estructura en el binario
volatile const void* keep_alive_pointer = &policy::module;

int main() {
    return 0;
}
```

### 2.2. Compilar para ARM Cortex-M4F y extraer métricas

Ejecute la compilación cruzada utilizando los *flags* específicos de punto flotante por hardware del STM32F405 y optimización por tamaño (`-Os`):

```bash
arm-none-eabi-g++ -c dummy_deploy.cpp -o dummy_deploy.o \
  -I./rl-tools/include \
  -mcpu=cortex-m4 \
  -mthumb \
  -mfloat-abi=hard \
  -mfpu=fpv4-sp-d16 \
  -Os
```

Analice las secciones del binario resultante:

```bash
arm-none-eabi-size -B dummy_deploy.o
```

**Resultado esperado:**
```text
   text    data     bss     dec     hex filename
  22556       4       0   22560    5820 dummy_deploy.o
```
* **`text` (22,556 bytes / ~22.03 KB):** Tamaño real que ocuparán el código y las constantes (pesos de la red SAC) en la memoria **Flash**.
* **`data` (4 bytes) y `bss` (0 bytes):** Consumo estático en memoria **RAM**.

---

## 3. Benchmark de tiempo de inferencia (emulación QEMU)

Durante el vuelo real, la política actúa de forma determinista evaluando únicamente la media ($\mu$) producida por el perceptrón multicapa (MLP), prescindiendo del muestreo estocástico. El siguiente benchmark evalúa directamente las operaciones matriciales del MLP sobre los pesos exportados durante $100\,000$ iteraciones.

### 3.1. Crear el script de evaluación (`benchmark_inferencia.cpp`)

```cpp
// Mira el contenido del archivo benchmark_inferencia.cpp creado
```

### 3.2. Compilar estáticamente y emular en ARM
Compile el binario con máxima optimización de velocidad (`-O3`) y empaquetado estático (`-static`) para su ejecución aislada:

```bash
arm-linux-gnueabihf-g++ -O3 benchmark_inferencia.cpp -o benchmark_inferencia_arm -static -I./rl-tools/include
```

Ejecute el binario dentro del emulador QEMU:

```bash
qemu-arm ./benchmark_inferencia_arm
```

**Resultado esperado:**
```text
Iniciando benchmark de 100000 iteraciones...
Tiempo total: 5.78117e+06 microsegundos
Tiempo promedio por inferencia: 57.8117 microsegundos
```
*(Considerando un ciclo de control a $32\text{ Hz}$ equivalente a $31,25\text{ ms}$, un tiempo de inferencia de $\sim 57,81\ \mu\text{s}$ representa apenas el $0,18\%$ del presupuesto temporal por paso).*
