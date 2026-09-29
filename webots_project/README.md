# Proyecto Webots

## Escenario de Simulación (`worlds/crazyflie.wbt`)

El escenario reproduce la habitación cerrada de 2x2m con las 4 esquinas (`cone_1`, `cone_2`, `cone_3`, `cone_4`) descrita en la sección 3.5.1 de la tesis (Diedrichs, 2025).

El robot Crazyflie está configurado con **controlador externo** y modo supervisor:

```vrml
DEF crazyflie Crazyflie {
  ...
  controller "<extern>"
  supervisor TRUE
}
```

Esta configuración permite desacoplar Webots del código Python, ejecutando los entrenamientos y evaluaciones directamente desde el IDE (Visual Studio Code), la terminal o un contenedor Docker sin intermediarios.

---

## Flujo de Ejecución (Paso a Paso)

### 1. Iniciar el simulador Webots
Abre el mundo en Webots desde tu máquina anfitriona (host):

```bash
webots webots_project/worlds/crazyflie.wbt
```

*Webots cargará el mundo y quedará en espera en el instante 00:00:00 hasta que un controlador externo se conecte.*

Para entrenamientos masivos en segundo plano o servidores sin monitor (modo *headless* acelerado):
```bash
webots --batch --mode=fast --no-rendering webots_project/worlds/crazyflie.wbt
```

### 2. Ejecutar controladores o entrenamientos desde la terminal / VS Code
Una vez que Webots está activo, ejecuta cualquier script desde tu entorno virtual o dentro del contenedor Docker (`webots_rltools_env`):

* **Evaluación del Piloto Heurístico (línea recta o lazo cerrado)**:
  ```bash
  python src/evaluation/run_heuristic_episode.py --mode proportional --corner cone_1
  ```
* **Entrenamiento con PPO (Stable-Baselines3)**:
  ```bash
  python src/training/train_sb3_ppo.py --corner cone_1 --timesteps 100000
  ```
* **Entrenamiento con SAC (RLtools)**:
  ```bash
  python src/training/train_rltools_sac.py --corner cone_1 --steps 100000
  ```

---

## Estructura del Directorio

- `worlds/crazyflie.wbt`: Archivo del mundo con arena de 2x2m, 4 conos y Crazyflie configurado con `<extern>`.
- `controllers/drone_shared_autonomy/`: Controlador alternativo si se deseara ejecutar autónomamente dentro de la GUI de Webots.
