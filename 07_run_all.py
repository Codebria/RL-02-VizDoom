#!/usr/bin/env python3
"""
Script Maestro - Ejecución Completa del Proyecto
Ejecuta todos los experimentos y análisis de forma secuencial
Universidad Nacional del Altiplano - Puno
Autor: Edson Denis Zanabria Ticona.
Doctorado en ciencias de la computación
Inteligencia artificial
"""

import os
import sys
import time
import json
from datetime import datetime

# ============================================
# Configuración
# ============================================

class ProjectRunner:
    """Ejecutor maestro del proyecto"""
    
    def __init__(self):
        self.start_time = None
        self.results = {
            'experiments': [],
            'timestamp': datetime.now().isoformat(),
            'status': 'running'
        }
    
    def print_header(self, title):
        """Imprimir encabezado decorado"""
        print("\n" + "="*80)
        print(f"{title:^80}")
        print("="*80 + "\n")
    
    def print_section(self, title):
        """Imprimir sección"""
        print("\n" + "-"*80)
        print(f">>> {title}")
        print("-"*80 + "\n")
    
    def run_experiment(self, name, script_path, description):
        """Ejecutar un experimento individual"""
        
        self.print_section(f"{name}: {description}")
        
        exp_start = time.time()
        
        try:
            # Verificar que el script existe
            if not os.path.exists(script_path):
                print(f" ERROR: Script no encontrado: {script_path}")
                return False
            
            print(f" Ejecutando: {script_path}")
            print(f" Inicio: {datetime.now().strftime('%H:%M:%S')}\n")
            
            # Ejecutar script
            exit_code = os.system(f'python "{script_path}"')
            
            exp_duration = time.time() - exp_start
            
            if exit_code == 0:
                print(f"\n COMPLETADO: {name}")
                print(f"  Tiempo: {exp_duration/60:.2f} minutos")
                
                self.results['experiments'].append({
                    'name': name,
                    'script': script_path,
                    'status': 'success',
                    'duration_minutes': exp_duration/60,
                    'timestamp': datetime.now().isoformat()
                })
                return True
            else:
                print(f"\n ERROR en {name}")
                print(f"Código de salida: {exit_code}")
                
                self.results['experiments'].append({
                    'name': name,
                    'script': script_path,
                    'status': 'failed',
                    'error_code': exit_code,
                    'timestamp': datetime.now().isoformat()
                })
                return False
                
        except Exception as e:
            print(f"\n EXCEPCIÓN en {name}: {str(e)}")
            self.results['experiments'].append({
                'name': name,
                'script': script_path,
                'status': 'exception',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def save_results(self):
        """Guardar resultados de la ejecución"""
        os.makedirs('results', exist_ok=True)
        
        results_file = f'results/execution_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n Log de ejecución guardado: {results_file}")
    
    def print_summary(self):
        """Imprimir resumen final"""
        
        self.print_header("RESUMEN DE EJECUCIÓN")
        
        total_time = time.time() - self.start_time
        successful = sum(1 for exp in self.results['experiments'] if exp['status'] == 'success')
        failed = len(self.results['experiments']) - successful
        
        print(f"Tiempo total de ejecución: {total_time/3600:.2f} horas ({total_time/60:.2f} minutos)")
        print(f"Experimentos exitosos: {successful}")
        print(f"Experimentos fallidos: {failed}")
        print(f"\nDetalle de experimentos:\n")
        
        for i, exp in enumerate(self.results['experiments'], 1):
            status_icon = "ok" if exp['status'] == 'success' else "error"
            duration = f"{exp.get('duration_minutes', 0):.2f} min" if 'duration_minutes' in exp else "N/A"
            print(f"{i}. {status_icon} {exp['name']:<40} {duration:>15}")
        
        print("\n" + "="*80)
        
        if failed == 0:
            print(" ¡TODOS LOS EXPERIMENTOS COMPLETADOS EXITOSAMENTE!")
        else:
            print(f"  {failed} experimento(s) fallaron. Revise los logs para detalles.")
        
        print("="*80 + "\n")
    
    def run_all(self, skip_training=False, quick_mode=False):
        """Ejecutar todos los experimentos del proyecto"""
        
        self.start_time = time.time()
        
        self.print_header("PROYECTO VIZDOOM - DEEP Q-LEARNING")
        print("Universidad Nacional del Altiplano - Puno")
        print("Trabajo RL-02: Implementación y pruebas de VizDoom")
        print(f"\nFecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Modo rápido: {'Activado' if quick_mode else 'Desactivado'}")
        print(f"Saltar entrenamiento: {'Sí' if skip_training else 'No'}")
        
        # Lista de experimentos
        experiments = []
        
        if not skip_training:
            experiments.extend([
                ("Experimento 1 - DQN Basic (Cacodemon)", 
                 "01_basic_cacodemon.py",
                 "Entrenamiento con DQN estándar en escenario Basic"),
                
                ("Experimento 2 - DQN Health Gathering", 
                 "02_health_gathering.py",
                 "Entrenamiento con DQN estándar en escenario Health Gathering"),
                
                ("Experimento 3 - DQN + PER (Ambos escenarios)", 
                 "04_DQN_with_PER.py",
                 "Entrenamiento con Prioritized Experience Replay"),
            ])
        
        experiments.extend([
            ("Análisis de Resultados", 
             "05_analyze_results.py",
             "Comparación y análisis de DQN vs DQN+PER"),
            
            ("Análisis de Performance", 
             "06_performance_analysis.py",
             "Análisis de tiempos, CPU y memoria"),
        ])
        
        # Ejecutar experimentos
        print(f"\n Total de tareas a ejecutar: {len(experiments)}\n")
        
        for i, (name, script, desc) in enumerate(experiments, 1):
            print(f"\n{'='*80}")
            print(f"TAREA {i}/{len(experiments)}")
            print(f"{'='*80}")
            
            success = self.run_experiment(name, script, desc)
            
            if not success and not quick_mode:
                response = input("\n ¿Desea continuar con los siguientes experimentos? (s/n): ")
                if response.lower() != 's':
                    print("\n  Ejecución interrumpida por el usuario.")
                    break
            
            # Pequeña pausa entre experimentos
            if i < len(experiments):
                print("\n  Pausa de 5 segundos antes del siguiente experimento...")
                time.sleep(5)
        
        # Resumen final
        self.results['status'] = 'completed'
        self.results['total_duration_hours'] = (time.time() - self.start_time) / 3600
        
        self.print_summary()
        self.save_results()
        
        # Información sobre archivos generados
        self.print_header("ARCHIVOS GENERADOS")
        print("Puede encontrar los resultados en los siguientes directorios:\n")
        print(" models/              - Modelos entrenados (.pth)")
        print(" results/             - Métricas de entrenamiento (.npz)")
        print(" results/analysis/    - Gráficos y reportes comparativos")
        print(" results/performance/ - Análisis de performance")
        print("\n" + "="*80 + "\n")

# ============================================
# Funciones auxiliares
# ============================================

def check_dependencies():
    """Verificar que todas las dependencias estén instaladas"""
    print("Verificando dependencias...\n")
    
    dependencies = {
        'vizdoom': 'VizDoom',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'psutil': 'psutil'
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f" {name}")
        except ImportError:
            print(f" {name} - NO INSTALADO")
            missing.append(name)
    
    if missing:
        print(f"\n  Faltan dependencias: {', '.join(missing)}")
        print("\nPara instalar:")
        print("  pip install vizdoom torch numpy matplotlib psutil")
        return False
    
    print("\n Todas las dependencias están instaladas\n")
    return True

def print_usage():
    """Imprimir instrucciones de uso"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   PROYECTO VIZDOOM - DEEP Q-LEARNING                         ║
║              Universidad Nacional del Altiplano - Puno                       ║
║                       Edson Denis Zanabria Ticona                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

USO:
    python 07_run_all.py [opciones]

OPCIONES:
    --help              Mostrar esta ayuda
    --check             Solo verificar dependencias
    --skip-training     Saltar experimentos de entrenamiento (solo análisis)
    --quick             Modo rápido (no pausar en errores)

EJEMPLOS:
    python 07_run_all.py
    python 07_run_all.py --skip-training
    python 07_run_all.py --quick

DESCRIPCIÓN:
    Este script ejecuta de forma secuencial:
    1. Experimento Basic con DQN estándar (~30-60 min)
    2. Experimento Health Gathering con DQN estándar (~60-90 min)
    3. Ambos experimentos con PER (~120-180 min)
    4. Análisis comparativo de resultados (~5 min)
    5. Análisis de performance (~5 min)

    Tiempo total estimado: 4-6 horas (en CPU)

NOTA:
    - Los experimentos de entrenamiento son intensivos en CPU
    - Se recomienda ejecutar en una computadora sin otras tareas pesadas
    - Los resultados se guardan automáticamente en cada etapa
    - Puede interrumpir con Ctrl+C y retomar después ejecutando experimentos individuales

═══════════════════════════════════════════════════════════════════════════════
""")

# ============================================
# Función principal
# ============================================

def main():
    """Función principal"""
    
    # Procesar argumentos
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        print_usage()
        return
    
    # Verificar dependencias
    if '--check' in args:
        check_dependencies()
        return
    
    if not check_dependencies():
        print("\n Por favor instale las dependencias faltantes antes de continuar.")
        return
    
    # Configurar opciones
    skip_training = '--skip-training' in args
    quick_mode = '--quick' in args
    
    # Confirmar ejecución
    if not skip_training:
        print("\n  ADVERTENCIA: La ejecución completa puede tomar 4-6 horas.")
        print("    Se ejecutarán entrenamientos intensivos en CPU.\n")
        response = input("¿Desea continuar? (s/n): ")
        if response.lower() != 's':
            print("\n Ejecución cancelada por el usuario.")
            return
    
    # Crear runner y ejecutar
    runner = ProjectRunner()
    
    try:
        runner.run_all(skip_training=skip_training, quick_mode=quick_mode)
    except KeyboardInterrupt:
        print("\n\n  Ejecución interrumpida por el usuario (Ctrl+C)")
        runner.results['status'] = 'interrupted'
        runner.save_results()
        print("\n Progreso guardado. Puede continuar ejecutando scripts individuales.")
    except Exception as e:
        print(f"\n\n Error crítico: {str(e)}")
        runner.results['status'] = 'error'
        runner.results['error'] = str(e)
        runner.save_results()

if __name__ == "__main__":
    main()
