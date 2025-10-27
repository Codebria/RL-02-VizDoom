#!/usr/bin/env python3
"""
Script de Análisis y Comparación de Resultados
Compara DQN estándar vs DQN con Prioritized Experience Replay
Universidad Nacional del Altiplano - Puno
Autor: Edson Denis Zanabria Ticona.
Doctorado en ciencias de la computación
Inteligencia artificial
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

# ============================================
# Configuración de Matplotlib
# ============================================
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================
# Funciones de Análisis
# ============================================

def smooth_curve(values, weight=0.9):
    """Suavizar curva usando media exponencial"""
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def calculate_statistics(rewards):
    """Calcular estadísticas de recompensas"""
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'median': np.median(rewards),
        'last_100_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    }

def plot_comparison(scenario_name, dqn_rewards, per_rewards, save_path):
    """Crear gráfico de comparación entre DQN y DQN+PER"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Comparación DQN vs DQN+PER - {scenario_name}', fontsize=16, fontweight='bold')
    
    # 1. Recompensas por episodio (raw)
    ax1 = axes[0, 0]
    ax1.plot(dqn_rewards, alpha=0.3, color='blue', label='DQN')
    ax1.plot(per_rewards, alpha=0.3, color='red', label='DQN+PER')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Recompensas por Episodio (Raw)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Recompensas suavizadas
    ax2 = axes[0, 1]
    ax2.plot(smooth_curve(dqn_rewards, 0.9), color='blue', linewidth=2, label='DQN (suavizado)')
    ax2.plot(smooth_curve(per_rewards, 0.9), color='red', linewidth=2, label='DQN+PER (suavizado)')
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Recompensa')
    ax2.set_title('Recompensas Suavizadas (EMA)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Media móvil (ventana de 100 episodios)
    ax3 = axes[1, 0]
    window = 100
    if len(dqn_rewards) >= window:
        dqn_moving_avg = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
        per_moving_avg = np.convolve(per_rewards, np.ones(window)/window, mode='valid')
        ax3.plot(dqn_moving_avg, color='blue', linewidth=2, label=f'DQN (MA-{window})')
        ax3.plot(per_moving_avg, color='red', linewidth=2, label=f'DQN+PER (MA-{window})')
    ax3.set_xlabel('Episodio')
    ax3.set_ylabel('Recompensa Promedio')
    ax3.set_title(f'Media Móvil (ventana={window})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Estadísticas comparativas
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    dqn_stats = calculate_statistics(dqn_rewards)
    per_stats = calculate_statistics(per_rewards)
    
    stats_text = f"""
    ESTADÍSTICAS COMPARATIVAS
    
    {'='*40}
    DQN Estándar:
    {'='*40}
    Media total:          {dqn_stats['mean']:.2f}
    Desv. estándar:       {dqn_stats['std']:.2f}
    Mínimo:               {dqn_stats['min']:.2f}
    Máximo:               {dqn_stats['max']:.2f}
    Mediana:              {dqn_stats['median']:.2f}
    Media últimos 100:    {dqn_stats['last_100_mean']:.2f}
    
    {'='*40}
    DQN + PER:
    {'='*40}
    Media total:          {per_stats['mean']:.2f}
    Desv. estándar:       {per_stats['std']:.2f}
    Mínimo:               {per_stats['min']:.2f}
    Máximo:               {per_stats['max']:.2f}
    Mediana:              {per_stats['median']:.2f}
    Media últimos 100:    {per_stats['last_100_mean']:.2f}
    
    {'='*40}
    MEJORA CON PER:
    {'='*40}
    Δ Media total:        {per_stats['mean'] - dqn_stats['mean']:+.2f} ({((per_stats['mean']/dqn_stats['mean']-1)*100):+.1f}%)
    Δ Últimos 100:        {per_stats['last_100_mean'] - dqn_stats['last_100_mean']:+.2f}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {save_path}")
    plt.close()

def plot_losses_comparison(scenario_name, dqn_losses, per_losses, save_path):
    """Comparar pérdidas durante entrenamiento"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Comparación de Pérdidas - {scenario_name}', fontsize=14, fontweight='bold')
    
    # Pérdidas raw
    ax1 = axes[0]
    ax1.plot(dqn_losses, alpha=0.5, color='blue', label='DQN')
    ax1.plot(per_losses, alpha=0.5, color='red', label='DQN+PER')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Pérdida (MSE)')
    ax1.set_title('Pérdida por Episodio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Pérdidas suavizadas
    ax2 = axes[1]
    ax2.plot(smooth_curve(dqn_losses, 0.95), color='blue', linewidth=2, label='DQN (suavizado)')
    ax2.plot(smooth_curve(per_losses, 0.95), color='red', linewidth=2, label='DQN+PER (suavizado)')
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Pérdida (MSE)')
    ax2.set_title('Pérdida Suavizada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de pérdidas guardado: {save_path}")
    plt.close()

def generate_report(scenario_name, dqn_data, per_data):
    """Generar reporte textual detallado"""
    
    report = f"""
{'='*80}
REPORTE DE ANÁLISIS - {scenario_name.upper()}
Comparación: DQN Estándar vs DQN con Prioritized Experience Replay
{'='*80}

1. INFORMACIÓN GENERAL
{'-'*80}
Fecha de análisis:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Escenario:            {scenario_name}
Episodios totales:    {len(dqn_data['rewards'])}

2. RESULTADOS DQN ESTÁNDAR
{'-'*80}
"""
    
    dqn_stats = calculate_statistics(dqn_data['rewards'])
    report += f"""
Recompensa media:              {dqn_stats['mean']:.2f}
Desviación estándar:           {dqn_stats['std']:.2f}
Recompensa mínima:             {dqn_stats['min']:.2f}
Recompensa máxima:             {dqn_stats['max']:.2f}
Mediana:                       {dqn_stats['median']:.2f}
Media últimos 100 episodios:   {dqn_stats['last_100_mean']:.2f}

Longitud promedio episodio:    {np.mean(dqn_data['lengths']):.1f} pasos
Pérdida promedio:              {np.mean(dqn_data['losses']):.6f}
"""
    
    report += f"""
3. RESULTADOS DQN CON PER
{'-'*80}
"""
    
    per_stats = calculate_statistics(per_data['rewards'])
    report += f"""
Recompensa media:              {per_stats['mean']:.2f}
Desviación estándar:           {per_stats['std']:.2f}
Recompensa mínima:             {per_stats['min']:.2f}
Recompensa máxima:             {per_stats['max']:.2f}
Mediana:                       {per_stats['median']:.2f}
Media últimos 100 episodios:   {per_stats['last_100_mean']:.2f}

Longitud promedio episodio:    {np.mean(per_data['lengths']):.1f} pasos
Pérdida promedio:              {np.mean(per_data['losses']):.6f}
"""
    
    # Análisis de mejora
    delta_mean = per_stats['mean'] - dqn_stats['mean']
    delta_percent = (per_stats['mean'] / dqn_stats['mean'] - 1) * 100 if dqn_stats['mean'] != 0 else 0
    delta_last100 = per_stats['last_100_mean'] - dqn_stats['last_100_mean']
    
    report += f"""
4. ANÁLISIS DE MEJORA CON PER
{'-'*80}
Diferencia en media total:         {delta_mean:+.2f} ({delta_percent:+.1f}%)
Diferencia últimos 100 episodios:  {delta_last100:+.2f}
Convergencia más rápida:           {'SÍ' if per_stats['last_100_mean'] > dqn_stats['last_100_mean'] else 'NO'}

Interpretación:
"""
    
    if delta_mean > 0:
        report += f"✓ PER muestra una mejora de {abs(delta_percent):.1f}% en recompensa promedio\n"
    else:
        report += f"✗ PER muestra un rendimiento {abs(delta_percent):.1f}% menor en recompensa promedio\n"
    
    if delta_last100 > 0:
        report += f"✓ PER converge a mejores políticas (últimos 100 episodios)\n"
    else:
        report += f"✗ DQN estándar converge a mejores políticas (últimos 100 episodios)\n"
    
    report += f"""
5. OBSERVACIONES Y CONCLUSIONES
{'-'*80}
"""
    
    if scenario_name.lower() == "basic":
        report += """
Escenario BASIC (Cacodemon):
- Objetivo: Eliminar al enemigo lo más rápido posible
- DQN aprende a disparar y moverse para evadir
"""
    else:
        report += """
Escenario HEALTH GATHERING:
- Objetivo: Sobrevivir recolectando medikits en ácido
- DQN aprende navegación y priorización de recursos
"""
    
    report += f"""

Prioritized Experience Replay:
- Prioriza experiencias con alto TD-error
- Importance sampling corrige sesgo de muestreo
- Teóricamente mejora eficiencia de aprendizaje

Resultados observados:
- Mejora absoluta: {delta_mean:+.2f} puntos
- Mejora relativa: {delta_percent:+.1f}%
- Estabilidad: {'Mayor' if per_stats['std'] < dqn_stats['std'] else 'Menor'} (std: {per_stats['std']:.2f} vs {dqn_stats['std']:.2f})

{'='*80}
FIN DEL REPORTE
{'='*80}
"""
    
    return report

# ============================================
# Función principal de análisis
# ============================================

def analyze_results():
    """Analizar y comparar todos los resultados"""
    
    print("\n" + "="*80)
    print("ANÁLISIS Y COMPARACIÓN DE RESULTADOS")
    print("DQN Estándar vs DQN con Prioritized Experience Replay")
    print("="*80 + "\n")
    
    os.makedirs('results/analysis', exist_ok=True)
    
    scenarios = {
        'basic': 'Basic (Cacodemon)',
        'health_gathering': 'Health Gathering (Medikit)'
    }
    
    for scenario_key, scenario_name in scenarios.items():
        print(f"\n{'='*80}")
        print(f"Analizando escenario: {scenario_name}")
        print(f"{'='*80}")
        
        # Cargar datos DQN estándar
        dqn_file = f'results/{scenario_key}_cacodemon_metrics.npz' if scenario_key == 'basic' else f'results/{scenario_key}_metrics.npz'
        per_file = f'results/{scenario_key}_PER_metrics.npz'
        
        try:
            # Verificar existencia de archivos
            if not os.path.exists(dqn_file):
                print(f" Archivo no encontrado: {dqn_file}")
                print("  Ejecute primero los scripts de entrenamiento.")
                continue
            
            if not os.path.exists(per_file):
                print(f" Archivo no encontrado: {per_file}")
                print("  Ejecute primero el script de PER.")
                continue
            
            # Cargar métricas
            dqn_data = np.load(dqn_file)
            per_data = np.load(per_file)
            
            print(f"✓ Datos cargados correctamente")
            print(f"  - DQN: {len(dqn_data['rewards'])} episodios")
            print(f"  - PER: {len(per_data['rewards'])} episodios")
            
            # Generar gráficos de comparación
            print("\nGenerando visualizaciones...")
            plot_comparison(
                scenario_name,
                dqn_data['rewards'],
                per_data['rewards'],
                f'results/analysis/{scenario_key}_comparison.png'
            )
            
            plot_losses_comparison(
                scenario_name,
                dqn_data['losses'],
                per_data['losses'],
                f'results/analysis/{scenario_key}_losses.png'
            )
            
            # Generar reporte
            print("Generando reporte...")
            report = generate_report(scenario_name, dqn_data, per_data)
            
            report_file = f'results/analysis/{scenario_key}_report.txt'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"✓ Reporte guardado: {report_file}")
            
            # Mostrar resumen
            print("\n" + "-"*80)
            print("RESUMEN DE RESULTADOS:")
            print("-"*80)
            dqn_stats = calculate_statistics(dqn_data['rewards'])
            per_stats = calculate_statistics(per_data['rewards'])
            print(f"DQN Estándar    - Media: {dqn_stats['mean']:.2f} | Últimos 100: {dqn_stats['last_100_mean']:.2f}")
            print(f"DQN + PER       - Media: {per_stats['mean']:.2f} | Últimos 100: {per_stats['last_100_mean']:.2f}")
            print(f"Mejora con PER: {per_stats['mean'] - dqn_stats['mean']:+.2f} ({((per_stats['mean']/dqn_stats['mean']-1)*100):+.1f}%)")
            print("-"*80)
            
        except Exception as e:
            print(f"✗ Error al procesar {scenario_name}: {e}")
            continue
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print("\nArchivos generados en: ./results/analysis/")
    print("  - *_comparison.png: Gráficos comparativos de recompensas")
    print("  - *_losses.png: Gráficos de pérdidas")
    print("  - *_report.txt: Reportes detallados")

# ============================================
# Ejecutar análisis
# ============================================

if __name__ == "__main__":
    analyze_results()
