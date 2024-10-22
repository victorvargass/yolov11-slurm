import argparse
from ultralytics import YOLO

def train_yolo_model(weights, data, epochs, imgsz, batch, lr0, lrf, momentum, weight_decay, warmup_epochs, 
                     warmup_momentum, warmup_bias_lr, box, cls, hsv_h, hsv_s, hsv_v, degrees, translate, 
                     scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste, workers, patience, 
                     name, optimizer):

    model = YOLO(weights)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        box=box,
        cls=cls,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        workers=workers,
        patience=patience,
        name=name,
        optimizer=optimizer,
        device=[0, 1],
        save_dir='/home/victorvargass/yolov11/outputs'
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo YOLO con parámetros personalizados")
    
    # Parámetros esenciales
    parser.add_argument('--weights', type=str, required=True, help='Nombre del modelo pre entrenado (.pt)')
    parser.add_argument('--data', type=str, required=True, help='Ruta al archivo de configuración de datos (YAML)')
    parser.add_argument('--epochs', type=int, default=1000, help='Número de épocas para entrenar')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamaño de la imagen')
    parser.add_argument('--batch', type=int, default=512, help='Tamaño del batch')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizador a utilizar')
    parser.add_argument('--name', type=str, help='Nombre del experimento')
    parser.add_argument('--patience', type=int, default=80, help='Tolerancia de épocas sin mejora')
    parser.add_argument('--workers', type=int, default=16, help='Número de trabajadores')

    # Parámetros adicionales
    parser.add_argument('--lr0', type=float, default=0.001, help='Tasa de aprendizaje inicial')
    parser.add_argument('--lrf', type=float, default=0.01, help='Tasa de aprendizaje final (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937, help='Momentum para SGD/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Decaimiento de peso del optimizador')
    parser.add_argument('--warmup_epochs', type=float, default=10.0, help='Épocas de calentamiento')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='Momentum inicial para warmup')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='Learning rate inicial para bias en warmup')
    parser.add_argument('--box', type=float, default=0.05, help='Ganancia de la pérdida de la caja (bounding box)')
    parser.add_argument('--cls', type=float, default=0.5, help='Ganancia de la pérdida de clasificación')
    parser.add_argument('--hsv_h', type=float, default=0.015, help='Ajuste de tono (hue) en la imagen (HSV)')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='Ajuste de saturación (saturation) en la imagen (HSV)')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='Ajuste de valor (value) en la imagen (HSV)')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotación de la imagen (+/- grados)')
    parser.add_argument('--translate', type=float, default=0.1, help='Traslación de la imagen (+/- fracción)')
    parser.add_argument('--scale', type=float, default=0.5, help='Escalado de la imagen (+/- ganancia)')
    parser.add_argument('--shear', type=float, default=0.0, help='Cizallamiento de la imagen (+/- grados)')
    parser.add_argument('--perspective', type=float, default=0.0, help='Perspectiva de la imagen (+/- fracción)')
    parser.add_argument('--flipud', type=float, default=0.0, help='Probabilidad de volteo vertical de la imagen')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Probabilidad de volteo horizontal de la imagen')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Probabilidad de mosaico en la imagen')
    parser.add_argument('--mixup', type=float, default=0.0, help='Probabilidad de aplicar mixup')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='Probabilidad de copy-paste en segmentación')

    args = parser.parse_args()

    train_yolo_model(
        weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        box=args.box,
        cls=args.cls,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        optimizer=args.optimizer
    )
