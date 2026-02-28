import csv
import glob
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def get_latest_results_file(directory="results"):
    """Find the latest results CSV file in the specified directory. If none found, check current directory."""
    search_pattern = os.path.join(directory, "auto_experiment_results_*.csv")
    files = glob.glob(search_pattern)
    if not files:
        fallback = os.path.join(directory, "auto_results.csv")
        if os.path.exists(fallback):
            return fallback
        local_files = glob.glob("auto_experiment_results_*.csv")
        if local_files:
            return max(local_files, key=os.path.getctime)
        return None
    
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def analyze_by_category(dataset_csv="dataset/dataset.csv", results_csv=None):
    """Analyze results by category, expected action, and instruction type. Generates summary tables and graphs."""
    if not results_csv:
        results_csv = get_latest_results_file()
        if not results_csv:
             results_csv = get_latest_results_file(directory=".")
        
    if not results_csv:
        print("❌ No results file found to analyze.")
        return

    if not os.path.exists(dataset_csv):
        if os.path.exists("dataset.csv"):
            dataset_csv = "dataset.csv"
        elif os.path.exists("dataset.csv"):
            dataset_csv = "dataset.csv"

    print(f"📂 Loading base dataset: {dataset_csv}")
    print(f"📂 Loading results:  {results_csv}\n")

    categories_by_id = {}
    if os.path.exists(dataset_csv):
        try:
            with open(dataset_csv, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                cat_key = 'category' if 'category' in reader.fieldnames else 'categoria'
                
                for row in reader:
                    categories_by_id[row.get('id', '').strip()] = row.get(cat_key, 'General')
        except Exception as e:
            print(f"⚠️ Error reading original dataset: {e}")
    else:
        print(f"⚠️ {dataset_csv} not found. All commands will be grouped in 'Unknown'.")

    stats = defaultdict(lambda: {"total_cmds": 0, "sum_accuracy": 0.0, "structural_errors": 0, "latencies": []})
    
    action_stats = defaultdict(lambda: {"total": 0, "sum_acc": 0.0, "total_runs": 0, "failed_runs": 0})
    type_stats = defaultdict(lambda: {"total_runs": 0, "failed_runs": 0})
    
    scatter_latencies = []
    scatter_accuracies = []
    
    global_cmds = 0
    global_sum_acc = 0.0

    try:
        with open(results_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_id = row.get('id', '').strip()
                cat = categories_by_id.get(q_id, "Desconocida")
                
                try:
                    acc = float(row.get('accuracy_%', 0.0))
                    lat = float(row.get('avg_latency', 0.0))
                    expected_action = row.get('Expected Action', row.get('action expected', 'Unknown')).strip()
                    actions_obtained = row.get('Obtained Actions', row.get('actions obtained (10 runs)', ''))
                except ValueError:
                    continue
                
                stats[cat]["total_cmds"] += 1
                stats[cat]["sum_accuracy"] += acc
                stats[cat]["latencies"].append(lat)
                stats[cat]["structural_errors"] += actions_obtained.count("Structural Error")
                
                global_cmds += 1
                global_sum_acc += acc
                scatter_latencies.append(lat)
                scatter_accuracies.append(acc)
                
                if expected_action:
                    try:
                        runs = len(json.loads(actions_obtained))
                    except:
                        runs = 10
                        
                    correct_runs = round((acc / 100.0) * runs)
                    failed_runs = runs - correct_runs
                    
                    action_stats[expected_action]["total"] += 1
                    action_stats[expected_action]["sum_acc"] += acc
                    action_stats[expected_action]["total_runs"] += runs
                    action_stats[expected_action]["failed_runs"] += failed_runs
                    
                    if expected_action.lower() == 'none':
                        cmd_type = 'none'
                    elif '=' in expected_action:
                        cmd_type = 'move'
                    else:
                        cmd_type = 'action'
                        
                    type_stats[cmd_type]["total_runs"] += runs
                    type_stats[cmd_type]["failed_runs"] += failed_runs

    except Exception as e:
        print(f"❌ Error processing {results_csv}: {e}")
        return

    print("█" * 78)
    print(f"📊 ACCURACY ANALYSIS BY CATEGORY (Multiple Runs)")
    print("█" * 78)
    print(f"{'CATEGORY':<15} | {'ACCURACY':<10} | {'COMMANDS':<10} | {'AVG LATENCY':<15} | {'STRUCTURAL ERRORS'}")
    print("-" * 78)

    plot_categories = []
    plot_accuracies = []
    plot_latencies_mean = []
    plot_struct_errs = []
    box_latencies = []

    for cat in sorted(stats.keys()):
        data = stats[cat]
        total = data["total_cmds"]
        if total == 0: continue
        
        accuracy = data["sum_accuracy"] / total
        avg_lat = sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0.0
        struct_errs = data["structural_errors"]
        
        plot_categories.append(cat.capitalize())
        plot_accuracies.append(accuracy)
        plot_latencies_mean.append(avg_lat)
        plot_struct_errs.append(struct_errs)
        box_latencies.append(data["latencies"])
        
        color = "\033[92m" if accuracy >= 80 else "\033[93m" if accuracy >= 50 else "\033[91m"
        reset = "\033[0m"
        print(f"{cat:<15} | {color}{accuracy:>6.1f}%{reset}   | {total:>8} | {avg_lat:>10.2f}s    | {struct_errs:>5}")

    print("-" * 78)
    global_acc = (global_sum_acc / global_cmds) if global_cmds > 0 else 0
    print(f"\n🎯 GLOBAL ACCURACY: {global_acc:.1f}% (Calculated on {global_cmds} commands)")
    
    print("\n" + "█" * 78)
    print(f"📉 FAILURE ANALYSIS BY EXPECTED ACTION")
    print("█" * 78)
    print(f"{'EXPECTED ACTION':<20} | {'FAILURES':<10} | {'TOTAL ATTEMPTS':<15} | {'ERROR RATE'}")
    print("-" * 78)

    sorted_fail_actions = sorted(action_stats.items(), key=lambda x: x[1]["failed_runs"], reverse=True)
    for act, data in sorted_fail_actions:
        fails = data["failed_runs"]
        tot_runs = data["total_runs"]
        error_rate = (fails / tot_runs * 100) if tot_runs > 0 else 0
        
        color = "\033[91m" if error_rate >= 20 else "\033[93m" if error_rate > 0 else "\033[92m"
        reset = "\033[0m"
        print(f"{act:<20} | {color}{fails:<10}{reset} | {tot_runs:<15} | {color}{error_rate:.1f}%{reset}")
        
    print("\n" + "█" * 78)
    print(f"🧠 FAILURE ANALYSIS BY INSTRUCTION TYPE")
    print("█" * 78)
    print(f"{'TYPE (INTENT)':<20} | {'FAILURES':<10} | {'TOTAL ATTEMPTS':<15} | {'ERROR RATE'}")
    print("-" * 78)

    sorted_type_stats = sorted(type_stats.items(), key=lambda x: x[1]["failed_runs"], reverse=True)
    for t_name, data in sorted_type_stats:
        fails = data["failed_runs"]
        tot_runs = data["total_runs"]
        error_rate = (fails / tot_runs * 100) if tot_runs > 0 else 0
        
        color = "\033[91m" if error_rate >= 20 else "\033[93m" if error_rate > 0 else "\033[92m"
        reset = "\033[0m"
        print(f"{t_name.upper():<20} | {color}{fails:<10}{reset} | {tot_runs:<15} | {color}{error_rate:.1f}%{reset}")

    print("█" * 78)

    try:
        plt.style.use('bmh')
        
        base_filename = os.path.basename(results_csv)
        folder_name = base_filename.replace('.csv', '').replace('auto_experiment_results_', 'graficas_')
        if not folder_name.startswith('graficas_'):
            folder_name = 'graficas_' + folder_name
            
        output_dir = os.path.join(os.path.dirname(results_csv), folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating graphs in directory: {output_dir}/")

        # --- Graph 1: Category Accuracy ---
        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(plot_categories, plot_accuracies, color='#4C72B0', edgecolor='black')
        plt.title('Exactitud Semántica por Categoría', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, 115)
        plt.axhline(global_acc, color='#C44E52', linestyle='--', linewidth=2, label=f'Media Global ({global_acc:.1f}%)')
        plt.legend(loc='lower right')
        for bar in bars1:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "1_accuracy_por_categoria.png"), dpi=300)
        plt.close()

        # --- Graph 2: Average Latency ---
        plt.figure(figsize=(10, 6))
        bars2 = plt.bar(plot_categories, plot_latencies_mean, color='#55A868', edgecolor='black')
        plt.title('Latencia Media de Inferencia', fontsize=14, fontweight='bold')
        plt.ylabel('Tiempo (segundos)', fontsize=12)
        if plot_latencies_mean: plt.ylim(0, max(plot_latencies_mean) * 1.2)
        for bar in bars2:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{bar.get_height():.2f}s', ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "2_latencia_media.png"), dpi=300)
        plt.close()

        # --- Graph 3: Structural Errors ---
        plt.figure(figsize=(10, 6))
        bars3 = plt.bar(plot_categories, plot_struct_errs, color='#C44E52', edgecolor='black')
        plt.title('Frecuencia de Errores Estructurales (Pydantic)', fontsize=14, fontweight='bold')
        plt.ylabel('Cantidad de Errores', fontsize=12)
        for bar in bars3:
            if bar.get_height() > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_errores_estructurales.png"), dpi=300)
        plt.close()

        # --- Graph 4: Latency Distribution (Boxplot) ---
        plt.figure(figsize=(10, 6))
        plt.boxplot(box_latencies, tick_labels=plot_categories, patch_artist=True,
                    boxprops=dict(facecolor='#8172B3', color='black'),
                    medianprops=dict(color='yellow', linewidth=2))
        plt.title('Distribución de Latencias por Categoría', fontsize=14, fontweight='bold')
        plt.ylabel('Tiempo (segundos)', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "4_distribucion_latencias.png"), dpi=300)
        plt.close()

        # --- Graph 5: Correlation Latency vs Accuracy (Scatter Plot) ---
        plt.figure(figsize=(10, 6))
        plt.scatter(scatter_latencies, scatter_accuracies, alpha=0.6, color='#64B5CD', edgecolors='black', s=60)
        plt.title('Correlación: Latencia vs Exactitud', fontsize=14, fontweight='bold')
        plt.xlabel('Latencia de Inferencia (s)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "5_correlacion_latencia_accuracy.png"), dpi=300)
        plt.close()
        
        # --- Graph 6: Accuracy Ranking by Expected Action ---
        actions = []
        accs = []
        for act, act_data in action_stats.items():
            if act_data["total"] > 0:
                actions.append(act)
                accs.append(act_data["sum_acc"] / act_data["total"])
        
        if actions:
            sorted_indices = np.argsort(accs)
            actions_sorted = [actions[i] for i in sorted_indices]
            accs_sorted = [accs[i] for i in sorted_indices]
            
            plt.figure(figsize=(10, max(6, len(actions_sorted)*0.3)))
            bars6 = plt.barh(actions_sorted, accs_sorted, color='#CCB974', edgecolor='black')
            plt.title('Exactitud (%) por Acción Específica', fontsize=14, fontweight='bold')
            plt.xlabel('Accuracy (%)', fontsize=12)
            plt.xlim(0, 115)
            for bar in bars6:
                plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}%', va='center', fontweight='bold', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "6_ranking_accuracy_acciones.png"), dpi=300)
            plt.close()

        # --- Graph 7: Failure Ranking by Expected Action ---
        fail_actions = [act for act, data in sorted_fail_actions]
        fail_counts = [data["failed_runs"] for act, data in sorted_fail_actions]
        
        if fail_actions:
            plt.figure(figsize=(10, max(6, len(fail_actions)*0.3)))
            bars7 = plt.barh(fail_actions[::-1], fail_counts[::-1], color='#E74C3C', edgecolor='black')
            plt.title('Cantidad Total de Fallos por Tipo de Acción', fontsize=14, fontweight='bold')
            plt.xlabel('Número de iteraciones falladas', fontsize=12)
            max_fails = max(fail_counts) if fail_counts else 10
            plt.xlim(0, max_fails * 1.15)
            for bar in bars7:
                if bar.get_width() > 0:
                    plt.text(bar.get_width() + (max_fails*0.02), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', fontweight='bold', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "7_ranking_fallos.png"), dpi=300)
            plt.close()

        # --- Graph 8: Error Rate by Instruction Type ---
        types = []
        error_rates = []
        fail_labels = []

        sorted_by_rate = sorted(type_stats.items(), key=lambda x: (x[1]["failed_runs"]/x[1]["total_runs"]) if x[1]["total_runs"] > 0 else 0, reverse=True)

        for t_name, data in sorted_by_rate:
            fails = data["failed_runs"]
            tot_runs = data["total_runs"]
            rate = (fails / tot_runs * 100) if tot_runs > 0 else 0
            
            types.append(t_name.upper())
            error_rates.append(rate)
            fail_labels.append(f"{fails}/{tot_runs}\nfallos")

        if types:
            plt.figure(figsize=(8, 6))
            colors_err = ['#E74C3C', '#F39C12', '#3498DB']
            bars8 = plt.bar(types, error_rates, color=colors_err[:len(types)], edgecolor='black', alpha=0.85)
            
            plt.title('Tasa de Error por Tipo de Intención', fontsize=14, fontweight='bold')
            plt.ylabel('Porcentaje de Error (%)', fontsize=12)
            
            max_rate = max(error_rates) if error_rates else 10
            plt.ylim(0, min(120, max_rate + (max_rate * 0.3) + 10))
            
            for bar, label in zip(bars8, fail_labels):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + (max_rate * 0.05) + 1, 
                         f'{height:.1f}%\n({label})', 
                         ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "8_tasa_error_por_tipo.png"), dpi=300)
            plt.close()

        # --- Graph 9: Failure Distribution by Instruction Type (Donut Chart) ---
        fail_counts_only = [data["failed_runs"] for _, data in sorted_type_stats if data["failed_runs"] > 0]
        fail_types_only = [t_name.upper() for t_name, data in sorted_type_stats if data["failed_runs"] > 0]

        if fail_counts_only:
            plt.figure(figsize=(8, 6))
            colors_donut = ['#FF9999', '#66B2FF', '#99FF99']
            
            wedges, texts, autotexts = plt.pie(fail_counts_only, labels=fail_types_only, autopct='%1.1f%%', 
                                               startangle=140, colors=colors_donut, pctdistance=0.80, 
                                               wedgeprops=dict(width=0.4, edgecolor='black'))
            
            plt.title('Distribución Global del Total de Fallos', fontsize=14, fontweight='bold')
            
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_color('black')
                autotext.set_fontweight('bold')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "9_distribucion_fallos_donut.png"), dpi=300)
            plt.close()

        print("✅ Graph generation completed successfully.")
        
    except NameError:
        print("\n⚠️ Matplotlib is not available. Please install matplotlib to generate graphs: pip install matplotlib")
    except Exception as e:
        print(f"\n⚠️ Error during graph generation: {e}")

if __name__ == "__main__":
    analyze_by_category(dataset_csv="dataset/dataset.csv")