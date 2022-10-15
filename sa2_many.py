import json

from optimize import OptimizedResult, optimize_sa2
import os
from multiprocessing import Process


def find_best_sa2(times, dest_path, k, markov, esp_p, esp_r, ws):
    print(f"start k={k}...")
    all_results = []
    best_contrast = 0
    best_time = -1

    for t in range(times):
        res = optimize_sa2(k, markov=markov * k, ws=ws, initial_temp=0.1, terminated_temp=0.1 / 2000,
                           esp_p=esp_p, esp_r=esp_r)
        all_results.append(
            {'id': t, 'nitr': res.nitr, 'variables': res.variables.tolist(), 'contrast': res.contrast,
             'safety': list(res.safety)})
        if res.contrast > best_contrast and sum(res.safety) <= 0.001 * (k - 1):
            best_contrast = res.contrast
            best_time = t
    final = {'best_one': best_time, 'results': all_results}
    with open(f"{dest_path}/k={k}.json", 'w') as f:
        json.dump(final, f)
    print(f"k={k} Done.")


if __name__ == "__main__":
    TIMES = 50
    MARKOV = 50000
    WS = 50
    ESP_R, ESP_P = 0.25, 0.25
    dir = f"sa2_many_mk={MARKOV}k_ws={WS}_espr={ESP_R}_espp={ESP_P}"
    os.mkdir(dir)
    for k in range(2, 8):
        p = Process(target=find_best_sa2, args=(TIMES, dir, k, MARKOV * k, ESP_P, ESP_R, WS))
        p.start()
