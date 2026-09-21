# Instalación

1. Instala el simulador Webots siguiendo las instrucciones según tu sistema operativo:
   <https://cyberbotics.com/doc/guide/installation-procedure>

2. Instala Python 3.10 o superior.

3. Clona este repositorio:

   ```bash
   git clone https://github.com/anadiedrichs/shared_autonomy_deep_rl.git
   cd shared_autonomy_deep_rl
   ```

4. Crea y activa un entorno virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

5. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

   Nota: `rltools` usa `pybind11` y compila código C++ en tiempo de instalación/ejecución.
   En Linux necesitás los headers de Python (`apt install python3-dev`). En macOS puede
   ser necesario aceptar la licencia de las CLI tools de Xcode
   (`sudo xcodebuild -license`).

6. Configura Webots para que use el Python de tu entorno virtual:
   Tools → Preferences → General → Python command → ruta absoluta a
   `venv/bin/python` (o `venv\Scripts\python.exe` en Windows).

## Verificación rápida

```bash
python -c "import gymnasium, rltools, stable_baselines3; print('OK')"
```
