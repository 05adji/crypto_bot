# run_comprehensive_test.py
import os
import subprocess
import datetime
import pandas as pd

# Konfigurasikan pengujian
configs = ["default", "aggressive", "conservative"]
periods = [30, 60, 90, 180, 365]  # dalam hari
results = []

# Buat direktori hasil jika belum ada
os.makedirs("test_results/backtests", exist_ok=True)

# Jalankan backtest untuk setiap kombinasi
for config in configs:
    for period in periods:
        print(f"Running backtest with {config} config for {period} days...")
        
        # Tentukan output file
        output_file = f"test_results/backtests/{config}_{period}days.txt"
        
        # Jalankan backtest dan simpan output
        command = f"python main.py --mode backtest --days {period} --config configs/{config}.json --no-plot > {output_file}"
        subprocess.run(command, shell=True)
        
        # Baca hasil backtest dari file output
        with open(output_file, 'r') as f:
            output = f.read()
        
        # Ekstrak metrik kinerja (ini perlu disesuaikan sesuai format output Anda)
        total_return = None
        annual_return = None
        max_drawdown = None
        sharpe_ratio = None
        
        for line in output.split('\n'):
            if "Total Return:" in line:
                total_return = float(line.split(':')[1].strip().replace('%', ''))
            elif "Annual Return:" in line:
                annual_return = float(line.split(':')[1].strip().replace('%', ''))
            elif "Maximum Drawdown:" in line:
                max_drawdown = float(line.split(':')[1].strip().replace('%', ''))
            elif "Sharpe Ratio:" in line:
                sharpe_ratio = float(line.split(':')[1].strip())
        
        # Tambahkan hasil ke daftar
        results.append({
            'config': config,
            'period': period,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        })

# Konversi hasil ke DataFrame
results_df = pd.DataFrame(results)

# Simpan hasil
results_df.to_csv("test_results/backtest_summary.csv", index=False)
print(results_df)

# Identifikasi konfigurasi terbaik berdasarkan Sharpe ratio
best_config = results_df.loc[results_df['sharpe_ratio'].idxmax()]
print(f"\nKonfigurasi terbaik berdasarkan Sharpe ratio:")
print(f"Config: {best_config['config']}")
print(f"Period: {best_config['period']} days")
print(f"Total Return: {best_config['total_return']}%")
print(f"Sharpe Ratio: {best_config['sharpe_ratio']}")