import json
import random
import time

from optimize import OptimizedResult, optimize_sa2
import os
from multiprocessing import Process


def find_best_sa2(times, dest_path, k, markov, esp_p, esp_r, ws):
    start = time.time()
    all_results = []
    best_contrast = 0
    best_time = -1

    for t in range(times):
        res = optimize_sa2(k, markov=markov * k, ws=ws, initial_temp=0.1, terminated_temp=0.1 / 2000,
                           esp_p=esp_p, esp_r=esp_r)
        res_map = {'id': t, 'nitr': res.nitr, 'variables': res.variables.tolist(), 'contrast': res.contrast,
               'safety': list(res.safety)}
        print(res_map)
        all_results.append(res_map)
        if res.contrast > best_contrast and sum(res.safety) <= 0.001 * (k - 1):
            best_contrast = res.contrast
            best_time = t
    final = {'best_one': best_time, 'results': all_results}
    print(final)
    with open(dest_path, 'w') as f:
        json.dump(final, f)
    print(f"k={k} Done. cost {time.time() - start}")


if __name__ == "__main__":
    TIMES = 5
    MARKOV = 45000
    WS = 5000
    ESP_R, ESP_P = 0.25, 0.25
    K = 5
    dir = f"sa2_many_one_TIMES={TIMES}_k={K}_mk={MARKOV}k_ws={WS}_espr={ESP_R}_espp={ESP_P}"
    os.mkdir(dir)
    for t in range(8):
        print(f"start k={K}...")
        p = Process(target=find_best_sa2, args=(TIMES, f"{dir}/{t}.json", K, MARKOV * K, ESP_P, ESP_R, WS))
        p.start()
