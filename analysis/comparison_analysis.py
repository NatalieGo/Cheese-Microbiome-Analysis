#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARISON ANALYSIS: MetaPhlAn vs Kraken2/Bracken
Сравнение двух методов метагеномного анализа

Описание: Скрипт для сравнения таксономических профилей,
         полученных с помощью MetaPhlAn и Kraken2/Bracken.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
import sys
from datetime import datetime

# путь к файлам
METAPHLAN_FILE = "comparison_results/RCGS_99z_taxonomic_profile_table.csv"
BRACKEN_FILE = "comparison_results/bracken_species.txt"
OUTPUT_DIR = "comparison_results/comparison_results"

# папка для результатов
os.makedirs(OUTPUT_DIR, exist_ok=True)

# функции
def load_metaphlan_data(filepath):
    """
    Загружает и обрабатывает данные MetaPhlAn
    
    Parameters:
    -----------
    filepath : str
        Путь к файлу MetaPhlAn
        
    Returns:
    --------
    pd.DataFrame
        Обработанный DataFrame с колонками:
        - species: название вида
        - abundance: относительное обилие (%)
        - full_taxonomy: полная таксономия
    """
    print(f"📥 Загрузка MetaPhlAn: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Найдено записей: {len(df)}")
    
    # очистка данных
    result = pd.DataFrame({
        'species': df['Taxon'].apply(lambda x: x if x != 'UNCLASSIFIED' else 'Unclassified'),
        'abundance': pd.to_numeric(df['Abundance (%)']),
        'full_taxonomy': df['Full Taxonomy']
    })
    
    # сортировка по убыванию обилия
    result = result.sort_values('abundance', ascending=False).reset_index(drop=True)
    
    print(f" ✅ Готово: {len(result)} видов")
    return result

def load_bracken_data(filepath):
    """
    Загружает и обрабатывает данные Bracken
    """
    print(f"Загрузка Bracken: {filepath}")
    
    df = pd.read_csv(filepath, sep='\t')
    print(f"Найдено записей: {len(df)}")
    
    # фильтрация видов
    df_species = df[df['taxonomy_lvl'] == 'S'].copy()
    print(f"Видов после фильтрации: {len(df_species)}")
    
    # очистка данных
    result = pd.DataFrame({
        'species': df_species['name'],
        'abundance': df_species['fraction_total_reads'] * 100,  # переводим в проценты
        'reads': df_species['new_est_reads'],
        'taxonomy_id': df_species['taxonomy_id']
    })
    
    # сортировка по убыванию обилия
    result = result.sort_values('abundance', ascending=False).reset_index(drop=True)
    
    print(f"   ✅ Готово: {len(result)} видов")
    return result

def merge_datasets(meta_df, bracken_df):
    """
    Объединяет данные MetaPhlAn и Bracken
    """
    print("\n🔄 Объединение данных...")
    
    # переименование колонок
    meta_renamed = meta_df.rename(columns={
        'abundance': 'metaphlan_abundance',
        'full_taxonomy': 'meta_taxonomy'
    })
    
    bracken_renamed = bracken_df.rename(columns={
        'abundance': 'bracken_abundance',
        'reads': 'bracken_reads'
    })
    
    # объединение
    merged = pd.merge(
        meta_renamed, 
        bracken_renamed, 
        on='species', 
        how='outer'
    ).fillna(0)
    
    # категория обнаружения
    def get_detection_category(row):
        meta = row['metaphlan_abundance'] > 0
        brack = row['bracken_abundance'] > 0
        if meta and brack:
            return 'Оба метода'
        elif meta and not brack:
            return 'Только MetaPhlAn'
        elif not meta and brack:
            return 'Только Bracken'
        else:
            return 'Не обнаружен'
    
    merged['detection'] = merged.apply(get_detection_category, axis=1)
    
    # сортировка по сумме обилий
    merged['total_abundance'] = merged['metaphlan_abundance'] + merged['bracken_abundance']
    merged = merged.sort_values('total_abundance', ascending=False).reset_index(drop=True)
    merged = merged.drop('total_abundance', axis=1)
    
    print("   ✅ Объединение завершено")
    return merged

def calculate_correlations(merged_df):
    """
    Рассчитывает корреляции между методами
    """
    print("\nРасчет корреляций...")
    
    # только виды, обнаруженные обоими методами
    both = merged_df[merged_df['detection'] == 'Оба метода']
    
    if len(both) < 2:
        print("   ⚠️ Недостаточно видов для корреляции")
        return None
    
    # логарифмическое преобразование
    both['log_meta'] = np.log10(both['metaphlan_abundance'] + 0.01)
    both['log_bracken'] = np.log10(both['bracken_abundance'] + 0.01)
    
    # корреляции
    pearson_r, pearson_p = pearsonr(both['metaphlan_abundance'], both['bracken_abundance'])
    spearman_r, spearman_p = spearmanr(both['metaphlan_abundance'], both['bracken_abundance'])
    
    # для логарифмических данных
    pearson_log_r, pearson_log_p = pearsonr(both['log_meta'], both['log_bracken'])
    
    results = {
        'n_species_both': len(both),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pearson_log_r': pearson_log_r,
        'pearson_log_p': pearson_log_p
    }
    
    print(f"   Видов в обоих методах: {len(both)}")
    print(f"   Корреляция Пирсона: r = {pearson_r:.3f} (p={pearson_p:.4f})")
    print(f"   Корреляция Спирмена: ρ = {spearman_r:.3f} (p={spearman_p:.4f})")
    print(f"   Лог-корреляция: r = {pearson_log_r:.3f}")
    
    return results, both

def plot_abundance_comparison(both_df, output_dir):
    """
    Строит графики сравнения
    """
    print("\nПостроение графиков сравнения...")
    
    # Настройка стиля
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. scatter plot (обычный)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # график 1: обычный scatter
    ax = axes[0, 0]
    ax.scatter(both_df['metaphlan_abundance'], both_df['bracken_abundance'], 
               alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)
    
    # линия идеального соответствия
    max_val = max(both_df['metaphlan_abundance'].max(), both_df['bracken_abundance'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Идеальное соответствие')
    
    # линия регрессии
    z = np.polyfit(both_df['metaphlan_abundance'], both_df['bracken_abundance'], 1)
    p = np.poly1d(z)
    ax.plot([0, max_val], p([0, max_val]), 'g-', alpha=0.7, label=f'Регрессия (y={z[0]:.2f}x+{z[1]:.2f})')
    
    ax.set_xlabel('MetaPhlAn Abundance (%)', fontsize=11)
    ax.set_ylabel('Bracken Abundance (%)', fontsize=11)
    ax.set_title('Сравнение относительного обилия видов', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # подписи для топ-5 видов
    top5 = both_df.nlargest(5, 'metaphlan_abundance')
    for _, row in top5.iterrows():
        ax.annotate(row['species'][:20], 
                   (row['metaphlan_abundance'], row['bracken_abundance']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    # график 2: логарифмический scatter
    ax = axes[0, 1]
    ax.scatter(both_df['log_meta'], both_df['log_bracken'], 
               alpha=0.6, s=50, c='coral', edgecolors='white', linewidth=0.5)
    
    # линия идеального соответствия
    min_log = min(both_df['log_meta'].min(), both_df['log_bracken'].min())
    max_log = max(both_df['log_meta'].max(), both_df['log_bracken'].max())
    ax.plot([min_log, max_log], [min_log, max_log], 'r--', alpha=0.5, label='Идеальное соответствие')
    
    ax.set_xlabel('log10(MetaPhlAn Abundance + 0.01)', fontsize=11)
    ax.set_ylabel('log10(Bracken Abundance + 0.01)', fontsize=11)
    ax.set_title('Сравнение в логарифмическом масштабе', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # график 3: bland-altman plot
    ax = axes[1, 0]
    mean_ab = (both_df['metaphlan_abundance'] + both_df['bracken_abundance']) / 2
    diff_ab = both_df['metaphlan_abundance'] - both_df['bracken_abundance']
    
    ax.scatter(mean_ab, diff_ab, alpha=0.6, s=50, c='purple', edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Нулевое расхождение')
    ax.axhline(y=diff_ab.mean(), color='g', linestyle='-', alpha=0.7, label=f'Среднее = {diff_ab.mean():.2f}')
    ax.axhline(y=diff_ab.mean() + 1.96*diff_ab.std(), color='orange', linestyle=':', alpha=0.7, label='±1.96 SD')
    ax.axhline(y=diff_ab.mean() - 1.96*diff_ab.std(), color='orange', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Среднее обилие (%)', fontsize=11)
    ax.set_ylabel('Разность (MetaPhlAn - Bracken)', fontsize=11)
    ax.set_title('Bland-Altman plot: Сравнение методов', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # график 4: гистограмма расхождений
    ax = axes[1, 1]
    ax.hist(diff_ab, bins=20, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Нулевое расхождение')
    ax.axvline(x=diff_ab.mean(), color='g', linestyle='-', alpha=0.7, label=f'Среднее = {diff_ab.mean():.2f}')
    
    ax.set_xlabel('Разность abundance (MetaPhlAn - Bracken)', fontsize=11)
    ax.set_ylabel('Количество видов', fontsize=11)
    ax.set_title('Распределение расхождений между методами', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   ✅ Графики сохранены: {output_dir}/comparison_scatter_plots.png")

def plot_top_species_comparison(meta_df, bracken_df, merged_df, output_dir, top_n=15):
    """
    Сравнивает топ-N видов по обоим методам
    """
    print(f"\nПостроение графика топ-{top_n} видов...")
    
    # Берем топ-N из объединенных данных по сумме
    top_species = merged_df.nlargest(top_n, 'metaphlan_abundance')[['species', 'metaphlan_abundance', 'bracken_abundance']]
    
    # Для графика нужно отсортировать по убыванию MetaPhlAn
    top_species = top_species.sort_values('metaphlan_abundance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # позиции для баров
    y_pos = np.arange(len(top_species))
    height = 0.35
    
    # бары для MetaPhlAn
    meta_bars = ax.barh(y_pos - height/2, top_species['metaphlan_abundance'], 
                        height, label='MetaPhlAn', color='steelblue', alpha=0.8)
    
    # бары для Bracken
    bracken_bars = ax.barh(y_pos + height/2, top_species['bracken_abundance'], 
                           height, label='Bracken', color='coral', alpha=0.8)
    
    # настройка осей
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[:30] + '...' if len(s) > 30 else s for s in top_species['species']])
    ax.set_xlabel('Относительное обилие (%)', fontsize=12)
    ax.set_title(f'Сравнение топ-{top_n} видов: MetaPhlAn vs Bracken', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # значения на бары
    for bars in [meta_bars, bracken_bars]:
        for bar in bars:
            width = bar.get_width()
            if width > 0.5:  # Показываем только значимые значения
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{width:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_{top_n}_species_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   ✅ График сохранен: {output_dir}/top_{top_n}_species_comparison.png")

def plot_venn_diagram(merged_df, output_dir):
    """
    Строит диаграмму Венна для обнаруженных видов
    """
    try:
        from matplotlib_venn import venn2
        
        # подсчеты
        meta_only = len(merged_df[merged_df['detection'] == 'Только MetaPhlAn'])
        bracken_only = len(merged_df[merged_df['detection'] == 'Только Bracken'])
        both = len(merged_df[merged_df['detection'] == 'Оба метода'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        venn = venn2(subsets=(meta_only, bracken_only, both),
                     set_labels=('MetaPhlAn', 'Bracken'))
        
        plt.title('Сравнение обнаруженных видов', fontsize=14, fontweight='bold')
        plt.savefig(f'{output_dir}/venn_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Диаграмма Венна:")
        print(f"   Только MetaPhlAn: {meta_only} видов")
        print(f"   Только Bracken: {bracken_only} видов")
        print(f"   Оба метода: {both} видов")
        print(f"   ✅ Диаграмма сохранена: {output_dir}/venn_diagram.png")
        
    except ImportError:
        print("   ⚠️ matplotlib_venn не установлен. Пропускаем диаграмму Венна.")
        print("   Установите: pip install matplotlib-venn")

def generate_report(merged_df, correlations, output_dir):
    """
    Готовит текстовый отчет с результатами
    """
    print("\nГенерация отчета...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Отчет по сравнениею методов метагеномного анализа")
    report_lines.append(f"MetaPhlAn vs Kraken2/Bracken")
    report_lines.append(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. Общая статистика
    report_lines.append("1. Общая статистика")
    report_lines.append("-" * 40)
    report_lines.append(f"MetaPhlAn: {len(merged_df[merged_df['metaphlan_abundance'] > 0])} видов")
    report_lines.append(f"Bracken: {len(merged_df[merged_df['bracken_abundance'] > 0])} видов")
    report_lines.append("")
    
    # 2. Пересечение методов
    both = merged_df[merged_df['detection'] == 'Оба метода']
    meta_only = merged_df[merged_df['detection'] == 'Только MetaPhlAn']
    bracken_only = merged_df[merged_df['detection'] == 'Только Bracken']
    
    report_lines.append("2. Пересечение методов")
    report_lines.append("-" * 40)
    report_lines.append(f"Виды, обнаруженные обоими методами: {len(both)}")
    report_lines.append(f"Виды только в MetaPhlAn: {len(meta_only)}")
    report_lines.append(f"Виды только в Bracken: {len(bracken_only)}")
    report_lines.append("")
    
    # 3. Корреляции
    if correlations:
        report_lines.append("3. Корреляции")
        report_lines.append("-" * 40)
        report_lines.append(f"Количество видов для корреляции: {correlations['n_species_both']}")
        report_lines.append(f"Корреляция Пирсона: r = {correlations['pearson_r']:.3f} (p={correlations['pearson_p']:.4f})")
        report_lines.append(f"Корреляция Спирмена: ρ = {correlations['spearman_r']:.3f} (p={correlations['spearman_p']:.4f})")
        report_lines.append(f"Лог-корреляция Пирсона: r = {correlations['pearson_log_r']:.3f}")
        report_lines.append("")
    
    # 4. Топ-10 видов по обоим методам
    report_lines.append("4. Топ-10 видов по обоим методам")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Вид':<35} {'MetaPhlAn (%)':>12} {'Bracken (%)':>12} {'Разность':>10}")
    report_lines.append("-" * 60)
    
    top10 = both.nlargest(10, 'metaphlan_abundance')[['species', 'metaphlan_abundance', 'bracken_abundance']]
    for _, row in top10.iterrows():
        diff = row['metaphlan_abundance'] - row['bracken_abundance']
        species_short = row['species'][:35] if len(row['species']) > 35 else row['species']
        report_lines.append(f"{species_short:<35} {row['metaphlan_abundance']:>11.2f}% {row['bracken_abundance']:>11.2f}% {diff:>9.2f}")
    report_lines.append("")
    
    # 5. Виды, обнаруженные только одним методом
    if len(meta_only) > 0:
        report_lines.append("5. Виды только в MetaPhlAn (топ-5)")
        report_lines.append("-" * 40)
        top_meta_only = meta_only.nlargest(5, 'metaphlan_abundance')[['species', 'metaphlan_abundance']]
        for _, row in top_meta_only.iterrows():
            report_lines.append(f"{row['species'][:40]:<40} {row['metaphlan_abundance']:>6.2f}%")
        report_lines.append("")
    
    if len(bracken_only) > 0:
        report_lines.append("6. Виды только в Bracken (топ-5)")
        report_lines.append("-" * 40)
        top_bracken_only = bracken_only.nlargest(5, 'bracken_abundance')[['species', 'bracken_abundance']]
        for _, row in top_bracken_only.iterrows():
            report_lines.append(f"{row['species'][:40]:<40} {row['bracken_abundance']:>6.2f}%")
        report_lines.append("")
    
    # 7. Заключение
    report_lines.append("7. Заключение")
    report_lines.append("-" * 40)
    
    if correlations and correlations['pearson_r'] > 0.8:
        report_lines.append("Сильная положительная корреляция между методами.")
        report_lines.append("   Оба метода дают схожие оценки относительного обилия.")
    elif correlations and correlations['pearson_r'] > 0.5:
        report_lines.append("Умеренная положительная корреляция между методами.")
        report_lines.append("   Методы согласованы, но есть некоторые расхождения.")
    else:
        report_lines.append("Слабая корреляция между методами.")
        report_lines.append("   Методы могут по-разному оценивать таксономический состав.")
    
    # Доминирующий вид
    dominant_meta = both.iloc[0]['species'] if len(both) > 0 else "N/A"
    dominant_meta_ab = both.iloc[0]['metaphlan_abundance'] if len(both) > 0 else 0
    dominant_bracken = both.iloc[0]['species'] if len(both) > 0 else "N/A"
    dominant_bracken_ab = both.iloc[0]['bracken_abundance'] if len(both) > 0 else 0
    
    report_lines.append(f"\nДоминирующий вид по MetaPhlAn: {dominant_meta} ({dominant_meta_ab:.1f}%)")
    report_lines.append(f"Доминирующий вид по Bracken: {dominant_bracken} ({dominant_bracken_ab:.1f}%)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("Конец отчета")
    report_lines.append("=" * 80)
    
    # сохранение отчета
    report_text = "\n".join(report_lines)
    with open(f'{output_dir}/comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Отчет сохранен: {output_dir}/comparison_report.txt")
    print("\n" + report_text[:500] + "...\n") 

def save_all_data(merged_df, meta_df, bracken_df, output_dir):
    """
    Сохраняет все данные в различных форматах
    """
    print("\nСохранение данных...")
    
    # 1. Полные данные
    merged_df.to_csv(f'{output_dir}/complete_comparison.csv', index=False, encoding='utf-8-sig')
    
    # 2. Только общие виды
    both = merged_df[merged_df['detection'] == 'Оба метода']
    both.to_csv(f'{output_dir}/common_species.csv', index=False, encoding='utf-8-sig')
    
    # 3. Топ-20 для презентации
    top20 = merged_df.nlargest(20, 'metaphlan_abundance')[
        ['species', 'metaphlan_abundance', 'bracken_abundance', 'detection']
    ]
    top20.to_csv(f'{output_dir}/top20_for_report.csv', index=False, encoding='utf-8-sig')
    
    # 4. MetaPhlAn чистые данные
    meta_df.to_csv(f'{output_dir}/metaphlan_clean.csv', index=False, encoding='utf-8-sig')
    
    # 5. Bracken чистые данные
    bracken_df.to_csv(f'{output_dir}/bracken_clean.csv', index=False, encoding='utf-8-sig')
    
    print(f"   ✅ Сохранено 5 файлов в {output_dir}/")


# основной скрипт
def main():
    """Основная функция программы"""
    
    print("\n" + "="*80)
    print("Сравнительный анализ: MetaPhlAn vs Kraken2/Bracken")
    print("="*80 + "\n")
    
    # проверка наличия файлов
    if not os.path.exists(METAPHLAN_FILE):
        print(f"❌ Ошибка: Файл {METAPHLAN_FILE} не найден!")
        print(f"   Текущая директория: {os.getcwd()}")
        print(f"   Доступные файлы: {os.listdir('.')[:10]}")
        return
    
    if not os.path.exists(BRACKEN_FILE):
        print(f"❌ Ошибка: Файл {BRACKEN_FILE} не найден!")
        return
    
    # загрузка данных
    print("\n1. Загрузка данных")
    meta_df = load_metaphlan_data(METAPHLAN_FILE)
    bracken_df = load_bracken_data(BRACKEN_FILE)
    
    # объединение данных
    print("\n2. Объединение данных")
    merged_df = merge_datasets(meta_df, bracken_df)
    
    # сохранение данных
    print("\n3. Сохранение данных")
    save_all_data(merged_df, meta_df, bracken_df, OUTPUT_DIR)
    
    # корреляционный анализ
    print("\n4. Корреляционный анализ")
    correlations, both_df = calculate_correlations(merged_df)
    
    # визуализация
    print("\n5. Визуализация")
    if both_df is not None and len(both_df) > 0:
        plot_abundance_comparison(both_df, OUTPUT_DIR)
        plot_top_species_comparison(meta_df, bracken_df, merged_df, OUTPUT_DIR, top_n=15)
    
    # диаграмма Венна
    plot_venn_diagram(merged_df, OUTPUT_DIR)
    
    # генерация отчета
    print("\n6. Генерация отчета")
    generate_report(merged_df, correlations, OUTPUT_DIR)
    
    # финал
    print("\n" + "="*80)
    print("✅ Анализ завершен!")
    print("="*80)
    print(f"\nВсе результаты сохранены в папке: {OUTPUT_DIR}/")
    print("\nСозданные файлы:")
    print(f"complete_comparison.csv - полные данные")
    print(f"common_species.csv - только общие виды")
    print(f"top20_for_report.csv - топ-20 для отчета")
    print(f"comparison_report.txt - текстовый отчет")
    print(f"comparison_scatter_plots.png - графики рассеяния")
    print(f"top_15_species_comparison.png - сравнение топ-15 видов")
    print(f"venn_diagram.png - диаграмма Венна")

if __name__ == "__main__":
    main()
