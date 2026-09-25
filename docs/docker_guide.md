# Guía de Ejecución con Docker y Webots

Este documento describe cómo levantar el entorno contenerizado del proyecto con soporte para aceleración gráfica (GUI) y ejecutar Webots junto con los módulos de entrenamiento de aprendizaje por refuerzo.

---

## 1. Características del Entorno Contenerizado

La imagen de Docker definida en el [`Dockerfile`](../Dockerfile) y gestionada con [`docker-compose.yaml`](../docker-compose.yaml) incluye:
- **Sistema base**: Ubuntu 24.04 LTS.
- **Simulador**: Webots oficial (Cyberbotics) instalado y configurado en `$WEBOTS_HOME=/usr/local/webots`.
- **Soporte gráfico**: Servidor X11, librerías Mesa OpenGL y `xvfb` para simulación acelerada o sin pantalla (headless).
- **Entorno Python**: Virtualenv en `/opt/venv` con dependencias preinstaladas (`gymnasium`, `stable-baselines3`, `rltools`, `torch`, `tensorboard`, `pandas`, `pybind11`, etc.).
- **Montaje de volumen**: La raíz del repositorio se monta en `/workspace`.

---

## 2. Paso a Paso para la Ejecución

### Paso 1: Permitir acceso a la pantalla X11 desde el Host (Linux)
Para que Webots pueda abrir ventanas gráficas desde el contenedor hacia tu monitor, debes conceder permisos de visualización al servidor X local. Ejecuta en tu terminal del host:

```bash
xhost +local:root
# o alternativamente:
xhost +local:$USER
```

> [!NOTE]
> Este comando debe ejecutarse una vez por cada sesión de usuario en tu sistema operativo host antes de levantar el contenedor si planeas usar la interfaz gráfica.

---

### Paso 2: Construir y levantar el contenedor
Desde la raíz del repositorio (`0001-webots-rl`), compila la imagen (si es la primera vez o hubo cambios en el `Dockerfile`) e inicia el servicio en segundo plano:

```bash
docker compose up -d --build
```

*(Si ya construiste la imagen previamente y no hay cambios en dependencias, basta con ejecutar `docker compose up -d`).*

---

### Paso 3: Acceder a la terminal interactiva del contenedor
Para abrir una sesión interactiva de Bash dentro del contenedor en ejecución:

```bash
docker compose exec -it simulador bash
```
*(O utilizando el nombre del contenedor: `docker exec -it webots_rltools_env bash`).*

Al ingresar:
- Estarás automáticamente en el directorio `/workspace`.
- El entorno virtual de Python (`/opt/venv`) estará activo por defecto en tu `$PATH`.

---

### Paso 4: Ejecutar Webots

Una vez dentro de la terminal del contenedor, dispones de dos modalidades de ejecución:

#### Opción A: Modo con interfaz gráfica (GUI 3D)
Ideal para inspección visual, teleoperación y validación de escenarios:

```bash
webots /workspace/webots_project/worlds/crazyflie.wbt
```
O simplemente iniciar Webots y abrir el mundo desde el menú del simulador:
```bash
webots
```

#### Opción B: Modo Headless / Sin Interfaz (Entrenamiento Rápido)
Para entrenamientos masivos de RL donde no se requiere renderizado 3D, puedes acelerar drásticamente los pasos de simulación ejecutando con `xvfb-run` y flags de simulación rápida:

```bash
xvfb-run webots --batch --mode=fast --no-rendering /workspace/webots_project/worlds/crazyflie.wbt
```

---

### Paso 5: Ejecutar entrenamientos y pruebas dentro del contenedor

Desde `/workspace` dentro del contenedor puedes ejecutar los scripts directamente:

- **Correr las pruebas unitarias**:
  ```bash
  python -m pytest tests/
  # o individualmente:
  python tests/test_envs.py
  python tests/test_pilots.py
  ```

- **Entrenar el piloto óptimo con PPO (Stable-Baselines3)**:
  ```bash
  python src/training/train_sb3_ppo.py --timesteps 100000
  ```

- **Entrenar el piloto óptimo con SAC (RLtools)**:
  ```bash
  python src/training/train_rltools_sac.py --steps 100000
  ```

---

### Paso 6: Detener el contenedor al finalizar
Cuando termines tu sesión de trabajo, sal de la terminal del contenedor con `exit` y detén los servicios desde el host:

```bash
docker compose down
```

---

## 3. Solución de Problemas Comunes

- **Error `cannot connect to X server`**:
  Verifica que ejecutaste `xhost +local:root` en el host y que la variable `$DISPLAY` esté definida en tu terminal del host (`echo $DISPLAY`, usualmente `:0` o `:1`).
- **Problemas con permisos de archivos creados dentro del contenedor**:
  El contenedor comparte el usuario local (`USER=${USER}` en `docker-compose.yaml`). Si algún archivo generado dentro queda como `root`, puedes corregir los permisos desde el host con:
  ```bash
  sudo chown -R $USER:$USER .
  ```
