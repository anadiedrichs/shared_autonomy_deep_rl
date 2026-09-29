# Training guide

El how-to para entrenar y visualizar el entrenamiento en webots.

En este ejemplo observamos como entrenar un piloto óptimo con PPO.

## Requisito

Ya tener configurado el entorno acorde como menciona la guía docker_guide.md de este repositorio.

## 1. Permitir acceso a la interfaz gráfica (en tu terminal local del host)                                                                                                                                                             
                                                                                                                                                                                                                                           
  Si acabas de reiniciar sesión o abrir una terminal nueva en tu sistema, asegúrate de habilitar la salida gráfica X11 para Docker:                                                                                                        

```                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    xhost +local:root                                                                                                                                                                                                                                                                                                                                                                                                                                                    
```
## 2. Terminal 1: abrir el simulador Webots con GUI                                                                                                                                                                                     
                                                                                                                                                                                                                                           
  Entra al contenedor y abre el mundo crazyflie.wbt:                                                                                                                                                                                       
  ```                                                                                                                                                                                                                                         
    docker exec -it webots_rltools_env bash                                                                                                                                                                                                
    webots webots_project/worlds/crazyflie.wbt                                                                                                                                                                                             
  ```
                                                                                                                                                                                                           
> Se abrirá la ventana gráfica de Webots. Como el Crazyflie tiene configurado el controlador en modo <extern>, el simulador quedará a la espera de que el script de entrenamiento se conecte.                                            

## 3. Terminal 2: Lanzar el entrenamiento de prueba con PPO                                                                                                                                                                             
  
  Abre una segunda terminal en tu máquina, entra al contenedor y ejecuta train_sb3_ppo.py:
  
    docker exec -it webots_rltools_env bash
    python src/training/train_sb3_ppo.py --timesteps 20000 --max-episode-steps 1000 --corner cone_1
  
  Puedes ajustar --timesteps (por ejemplo, 20000 para una prueba rápida de pocos minutos, o 50000 / 100000 para un entrenamiento completo).

## 4. Cómo controlar la visualización dentro de Webots
  
Una vez que el script se conecte, la simulación arrancará automáticamente:
  
* Velocidad de simulación (barra superior de Webots):
  - Play (Tiempo real): Te permite observar la física del cuadricóptero, la inclinación (pitch/roll) y la altitud a velocidad natural.
  - Fast / Run: Acelera la simulación computacional al máximo de tu CPU/GPU. Verás al dron moverse en cámara rápida de un intento a otro, lo que permite entrenar más rápido mientras sigues viendo la animación 3D.
* Qué esperar en los episodios:
  - En cada reset el dron despegará y se estabilizará en el centro.
  - En los primeros episodios explorará con movimientos más ruidosos; si choca contra las paredes o cae, el episodio se reiniciará inmediatamente.
  - Tras varios miles de pasos, comenzarás a notar trayectorias consistentes orientadas hacia cone_1 (+0.9, +0.9, esquina superior derecha).
  
 ## 5. (Opcional) Ver las métricas en TensorBoard
  
  Si deseas monitorear las curvas de recompensa (rollout/ep_rew_mean), longitud de episodios y pérdidas del algoritmo en tu navegador:
  
  Se puede ejecutar directamente en tu máquina host o en otra terminal de Docker:
  
  ```
  tensorboard --logdir ./logs/ppo
  ```
  
  Luego abrir en el navegador: http://localhost:6006
