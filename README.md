# yolov11-slurm

### Instrucciones

1. Ingresar a Patagón
2. Instalar el contenedor de Ultralytics

```bash
srun --container-name=ultralytics --container-image='ultralytics/ultralytics:latest' pip freeze
```

2. Clonar repositorio

```bash
git clone https://github.com/victorvargass/yolov11-slurm
cd yolov11-slurm
```

3. Crear una carpeta **datasets/** y alojar el dataset con la siguiente estructura:

```python
    datasets/NOMBRE_DATASET/
        train/
            images/ #*.jpg, *.png, etc
            labels/ #*.txt
        val/
            images/ #*.jpg, *.png, etc
            labels/ #*.txt
        test/
            images/ #*.jpg, *.png, etc
            labels/ #*.txt
```

4. Para setear las rutas del dataset y las categorías del mismo, crear un archivo **.yaml** en la carpeta principal. (Ver ejemplo **data.yaml**)

5. Crear un archivo .slurm que tenga la configuración del experimento a lanzar en Patagon. (Ver ejemplo **yolo.slurm**)

6. Lanzar un experimento en Patagon

```
sbatch yolo.slurm
```

Cada experimento creará una carpeta dentro del contenedor en la carpeta **/ultralytics/runs/detect/**, la cual contendrá información valiosa de cada experimento como los resultados, las métricas de evaluación, entre otros.

7. Para ingresar el contenedor:
```
srun --container-name=ultralytics --pty bash
cd /ultralytics/runs/detect/
```

### Comandos útiles en Patagón

- Ver experimentos corriendo: `squeue`
- Cancelar la ejecución de un experiment: `scancel {JOB_ID}`
- Ver logs de una ejecución:
    - Todo el archivo de logs: `cat slurm-{JOB_ID}.out`
    - Sólo las últimas 50 líneas de log y en tiempo real: `tail -n 50 -f slurm-{JOB_ID}.out`