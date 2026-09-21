# Proyecto Webots

## Migración del mundo (.wbt)

El escenario (habitación cerrada 2x2m, sin obstáculos, con las 4 esquinas como
posibles objetivos, descrito en la sección 3.5.1 de la tesis 2025) no requiere
cambios geométricos para pasar a acciones continuas. Pasos para migrarlo:

1. Copiar el/los archivo(s) `.wbt` desde
   `drone-deep-rl/v1_shared_autonomy/.../worlds/` a `worlds/` en este repo.
2. Copiar cualquier proto/textura asociada que use el mundo.
3. Actualizar la referencia al controlador en el `.wbt` para que apunte a
   `controllers/drone_shared_autonomy/drone_shared_autonomy.py` (o renombrar
   este último para que coincida con el nombre que ya usa el mundo).
4. Verificar en Webots (Tools → Preferences → General → Python command) que el
   intérprete configurado sea el del entorno virtual de este repo (ver
   `../INSTALL.md`).

## Estructura

- `worlds/`: archivo(s) `.wbt` del escenario.
- `controllers/drone_shared_autonomy/`: controlador Python, entry point de
  Webots (ver docstring del archivo para el TODO de implementación).
