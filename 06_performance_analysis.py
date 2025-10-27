#!/usr/bin/env python3
"""
Script de Análisis de Performance
Mide y compara tiempo de ejecución, uso de CPU y memoria
Universidad Nacional del Altiplano - Puno
Autor: Edson Denis Zanabria Ticona.
Doctorado en ciencias de la computación
Inteligencia artificial
"""

import psutil
import time
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ============================================
# Monitor de Performance
# ============================================

class PerformanceMonitor:
    """Monitor para medir uso de recursos durante entrenamiento"""
    
    def __init__(self, name="Experiment"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.cpu_samples = []
        self.memory_samples = []
        self.process = psutil.Process()
        self.monitoring = False
    
    def start(self):
        """Iniciar monitoreo"""
        self.start_time = time.time()
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        print(f"\n{'='*60}")
        print(f"Iniciando monitoreo: {self.name}")
        print(f"{'='*60}")
    
    def sample(self):
        """Tomar muestra de uso de recursos"""
        if self.monitoring:
            try:
                cpu_percent = self.process.cpu_percent(interval=0.1)
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convertir a MB
                
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_mb)
            except:
                pass
    
    def stop(self):
        """Detener monitoreo y calcular estadísticas"""
        self.end_time = time.time()
        self.monitoring = False
        
        elapsed_time = self.end_time - self.start_time
        
        stats = {
            'name': self.name,
            'duration_seconds': elapsed_time,
            'duration_minutes': elapsed_time / 60,
            'duration_hours': elapsed_time / 3600,
            'cpu_usage': {
                'mean': np.mean(self.cpu_samples) if self.cpu_samples else 0,
                'max': np.max(self.cpu_samples) if self.cpu_samples else 0,
                'min': np.min(self.cpu_samples) if self.cpu_samples else 0,
                'std': np.std(self.cpu_samples) if self.cpu_samples else 0
            },
            'memory_usage_mb': {
                'mean': np.mean(self.memory_samples) if self.memory_samples else 0,
                'max': np.max(self.memory_samples) if self.memory_samples else 0,
                'min': np.min(self.memory_samples) if self.memory_samples else 0,
                'std': np.std(self.memory_samples) if self.memory_samples else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self._print_summary(stats)
        return stats
    
    def _print_summary(self, stats):
        """Imprimir resumen de performance"""
        print(f"\n{'='*60}")
        print(f"Resumen de Performance: {self.name}")
        print(f"{'='*60}")
        print(f"\nTiempo de ejecución:")
        print(f"  Total: {stats['duration_seconds']:.2f} segundos")
        print(f"        ({stats['duration_minutes']:.2f} minutos)")
        print(f"        ({stats['duration_hours']:.2f} horas)")
        
        print(f"\nUso de CPU:")
        print(f"  Promedio: {stats['cpu_usage']['mean']:.2f}%")
        print(f"  Máximo:   {stats['cpu_usage']['max']:.2f}%")
        print(f"  Mínimo:   {stats['cpu_usage']['min']:.2f}%")
        
        print(f"\nUso de Memoria:")
        print(f"  Promedio: {stats['memory_usage_mb']['mean']:.2f} MB")
        print(f"  Máximo:   {stats['memory_usage_mb']['max']:.2f} MB")
        print(f"  Mínimo:   {stats['memory_usage_mb']['min']:.2f} MB")
        print(f"{'='*60}\n")

# ============================================
# Funciones de análisis comparativo
# ============================================

def compare_performance(stats_list, save_dir='results/performance'):
    """Comparar performance entre diferentes experimentos"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Crear gráficos de comparación
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis Comparativo de Performance', fontsize=16, fontweight='bold')
    
    names = [s['name'] for s in stats_list]
    
    # 1. Tiempo de ejecución
    ax1 = axes[0, 0]
    durations = [s['duration_minutes'] for s in stats_list]
    bars1 = ax1.bar(names, durations, color=['blue', 'red', 'green', 'orange'][:len(names)])
    ax1.set_ylabel('Tiempo (minutos)')
    ax1.set_title('Tiempo de Ejecución')
    ax1.tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}min',
                ha='center', va='bottom')
    
    # 2. Uso promedio de CPU
    ax2 = axes[0, 1]
    cpu_means = [s['cpu_usage']['mean'] for s in stats_list]
    bars2 = ax2.bar(names, cpu_means, color=['blue', 'red', 'green', 'orange'][:len(names)])
    ax2.set_ylabel('CPU (%)')
    ax2.set_title('Uso Promedio de CPU')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # 3. Uso promedio de memoria
    ax3 = axes[1, 0]
    mem_means = [s['memory_usage_mb']['mean'] for s in stats_list]
    bars3 = ax3.bar(names, mem_means, color=['blue', 'red', 'green', 'orange'][:len(names)])
    ax3.set_ylabel('Memoria (MB)')
    ax3.set_title('Uso Promedio de Memoria')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}MB',
                ha='center', va='bottom')
    
    # 4. Tabla comparativa
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_text = "RESUMEN COMPARATIVO\n\n"
    for stat in stats_list:
        table_text += f"{stat['name']}:\n"
        table_text += f"  Tiempo: {stat['duration_minutes']:.1f} min\n"
        table_text += f"  CPU:    {stat['cpu_usage']['mean']:.1f}%\n"
        table_text += f"  RAM:    {stat['memory_usage_mb']['mean']:.0f} MB\n\n"
    
    ax4.text(0.1, 0.9, table_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico comparativo guardado: {save_dir}/performance_comparison.png")
    plt.close()

def generate_performance_report(stats_list, save_dir='results/performance'):
    """Generar reporte de performance"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    report = f"""
{'='*80}
REPORTE DE ANÁLISIS DE PERFORMANCE
Universidad Nacional del Altiplano - Puno
{'='*80}

Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total de experimentos analizados: {len(stats_list)}

{'='*80}
RESULTADOS INDIVIDUALES
{'='*80}
"""
    
    for i, stat in enumerate(stats_list, 1):
        report += f"""
{'-'*80}
{i}. {stat['name']}
{'-'*80}

TIEMPO DE EJECUCIÓN:
  Total:      {stat['duration_seconds']:.2f} segundos
  Minutos:    {stat['duration_minutes']:.2f} min
  Horas:      {stat['duration_hours']:.2f} hrs

USO DE CPU:
  Promedio:   {stat['cpu_usage']['mean']:.2f}%
  Máximo:     {stat['cpu_usage']['max']:.2f}%
  Mínimo:     {stat['cpu_usage']['min']:.2f}%
  Desv. std:  {stat['cpu_usage']['std']:.2f}%

USO DE MEMORIA:
  Promedio:   {stat['memory_usage_mb']['mean']:.2f} MB
  Máximo:     {stat['memory_usage_mb']['max']:.2f} MB
  Mínimo:     {stat['memory_usage_mb']['min']:.2f} MB
  Desv. std:  {stat['memory_usage_mb']['std']:.2f} MB

"""
    
    # Análisis comparativo
    if len(stats_list) > 1:
        report += f"""
{'='*80}
ANÁLISIS COMPARATIVO
{'='*80}

TIEMPO DE EJECUCIÓN:
"""
        durations = [(s['name'], s['duration_minutes']) for s in stats_list]
        durations_sorted = sorted(durations, key=lambda x: x[1])
        report += f"  Más rápido:  {durations_sorted[0][0]} ({durations_sorted[0][1]:.2f} min)\n"
        report += f"  Más lento:   {durations_sorted[-1][0]} ({durations_sorted[-1][1]:.2f} min)\n"
        report += f"  Diferencia:  {durations_sorted[-1][1] - durations_sorted[0][1]:.2f} min ({((durations_sorted[-1][1]/durations_sorted[0][1]-1)*100):.1f}% más lento)\n"
        
        report += "\nUSO DE CPU:\n"
        cpus = [(s['name'], s['cpu_usage']['mean']) for s in stats_list]
        cpus_sorted = sorted(cpus, key=lambda x: x[1])
        report += f"  Menor uso:   {cpus_sorted[0][0]} ({cpus_sorted[0][1]:.2f}%)\n"
        report += f"  Mayor uso:   {cpus_sorted[-1][0]} ({cpus_sorted[-1][1]:.2f}%)\n"
        
        report += "\nUSO DE MEMORIA:\n"
        mems = [(s['name'], s['memory_usage_mb']['mean']) for s in stats_list]
        mems_sorted = sorted(mems, key=lambda x: x[1])
        report += f"  Menor uso:   {mems_sorted[0][0]} ({mems_sorted[0][1]:.0f} MB)\n"
        report += f"  Mayor uso:   {mems_sorted[-1][0]} ({mems_sorted[-1][1]:.0f} MB)\n"
    
    report += f"""
{'='*80}
OBSERVACIONES
{'='*80}

1. EFICIENCIA COMPUTACIONAL:
   - Los experimentos se ejecutaron en CPU (sin aceleración GPU)
   - El tiempo de ejecución depende de la complejidad del escenario
   - El uso de memoria es relativamente constante durante el entrenamiento

2. PRIORITIZED EXPERIENCE REPLAY:
   - PER añade overhead computacional por el SumTree
   - Mayor uso de CPU debido al cálculo de prioridades
   - Mayor uso de memoria por estructura de datos adicional
   - El trade-off puede ser beneficioso si mejora significativamente el aprendizaje

3. RECOMENDACIONES:
   - Para entrenamientos largos, considerar usar GPU si está disponible
   - El buffer de replay (10,000 experiencias) tiene un costo de memoria manejable
   - Los episodios más largos (Health Gathering) consumen más tiempo

{'='*80}
FIN DEL REPORTE
{'='*80}
"""
    
    report_file = f'{save_dir}/performance_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Reporte de performance guardado: {report_file}")
    
    # Guardar JSON con datos
    json_file = f'{save_dir}/performance_data.json'
    with open(json_file, 'w') as f:
        json.dump(stats_list, f, indent=2)
    print(f"✓ Datos JSON guardados: {json_file}")

# ============================================
# Función para cargar estadísticas guardadas
# ============================================

def load_performance_stats(stats_dir='results/performance'):
    """Cargar estadísticas de performance guardadas"""
    json_file = f'{stats_dir}/performance_data.json'
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    else:
        print(f"⚠ No se encontró archivo de estadísticas: {json_file}")
        return []

# ============================================
# Función principal
# ============================================

def main():
    """Analizar performance de experimentos guardados"""
    
    print("\n" + "="*80)
    print("ANÁLISIS DE PERFORMANCE DE EXPERIMENTOS")
    print("="*80 + "\n")
    
    # Intentar cargar datos guardados
    stats = load_performance_stats()
    
    if stats:
        print(f"✓ Cargadas {len(stats)} estadísticas de performance")
        compare_performance(stats)
        generate_performance_report(stats)
    else:
        print("⚠ No se encontraron estadísticas guardadas.")
        print("\nPara generar estadísticas de performance:")
        print("1. Modifique los scripts de entrenamiento para incluir PerformanceMonitor")
        print("2. Ejecute los experimentos")
        print("3. Ejecute este script nuevamente")
        
        # Crear ejemplo de cómo usar el monitor
        example_code = """
# Ejemplo de uso en scripts de entrenamiento:

from performance_monitor import PerformanceMonitor

# Al inicio del entrenamiento
monitor = PerformanceMonitor("DQN Basic")
monitor.start()

# Durante el entrenamiento (en cada episodio)
for episode in range(episodes):
    # ... código de entrenamiento ...
    monitor.sample()  # Tomar muestra de recursos

# Al final
stats = monitor.stop()

# Guardar estadísticas
import json
with open('results/performance/basic_dqn_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
"""
        print("\n" + "="*80)
        print("CÓDIGO DE EJEMPLO:")
        print("="*80)
        print(example_code)

if __name__ == "__main__":
    main()
