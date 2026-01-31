import numpy as np
from deap import base, creator, tools
import joblib
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

Temp_options = [300]
epsilon_dot = 0.06          # ε̇
QT_options = [25, 75]      # QT
A1_t_options = list(range(6, 13, 2))   # A1-t
t_ten_hold = 60            # t-ten-hold
ST1_t_options = list(range(2, 25, 2))  # ST1-t

# ========== Model loading path ==========
model_path = 'best_random_forest_model.pkl'


def setup_toolbox():
    if 'FitnessMax' not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if 'Individual' not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("Cu", np.random.uniform, 0, 9)
    toolbox.register("Mg", np.random.uniform, 0, 10)
    toolbox.register("Ti", np.random.uniform, 0, 2)
    toolbox.register("Mn", np.random.uniform, 0, 4)

    def create_individual():
        while True:
            Cu = toolbox.Cu()
            Mg = toolbox.Mg()
            Ti = toolbox.Ti()
            Mn = toolbox.Mn()
            Al = 100 - (Cu + Mg + Ti + Mn)
            if Al >= 0:
                return creator.Individual([Cu, Mg, Ti, Mn])

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def calculate_bulk_and_cov(Al, Cu, Mg, Ti, Mn):
    bulk_prop = 0.01 * (Al * 76 + Cu * 140 + Mg * 45 + Ti * 110 + Mn * 120)
    cov_prop = 0.01 * (Al * 118 + Cu * 138 + Mg * 130 + Ti * 136 + Mn * 139)
    return bulk_prop, cov_prop


def clip_individual(ind):
    bounds = {
        'Cu': (0, 9),
        'Mg': (0, 10),
        'Ti': (0, 2),
        'Mn': (0, 4)
    }
    Cu, Mg, Ti, Mn = ind

    Cu = np.clip(Cu, *bounds['Cu'])
    Mg = np.clip(Mg, *bounds['Mg'])
    Ti = np.clip(Ti, *bounds['Ti'])
    Mn = np.clip(Mn, *bounds['Mn'])

    total = Cu + Mg + Ti + Mn
    if total > 100:
        ratio = 100.0 / total
        Cu *= ratio
        Mg *= ratio
        Ti *= ratio
        Mn *= ratio

    return creator.Individual([Cu, Mg, Ti, Mn])


def run_ga_for_params(Temp, QT, A1_t, ST1_t):
    model = joblib.load(model_path)
    toolbox = setup_toolbox()

    def evaluate(individual):
        Cu, Mg, Ti, Mn = individual
        Al = 100 - (Cu + Mg + Ti + Mn)
        if Al < 0:
            return (0,)

        bulk_prop, cov_prop = calculate_bulk_and_cov(Al, Cu, Mg, Ti, Mn)

        feature_array = np.array([[Temp, A1_t, ST1_t, epsilon_dot,
                                   QT, t_ten_hold, Al, Mn, Mg, Ti, Cu,
                                   bulk_prop, cov_prop]])

        UTS = max(0, model.predict(feature_array)[0])
        return (UTS,)

    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=5000)
    NGEN = 5000

    for gen in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.5:
                toolbox.mate(child1, child2)
                child1[:] = clip_individual(child1)
                child2[:] = clip_individual(child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < 0.2:
                toolbox.mutate(mutant)
                mutant[:] = clip_individual(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    fits = [ind.fitness.values[0] for ind in population]
    best_ind = population[np.argmax(fits)]

    Cu, Mg, Ti, Mn = best_ind
    Al = 100 - (Cu + Mg + Ti + Mn)
    bulk_prop, cov_prop = calculate_bulk_and_cov(Al, Cu, Mg, Ti, Mn)

    best_result = {
        'Temp': Temp,
        'QT': QT,
        'A1-t': A1_t,
        'ST1-t': ST1_t,
        'Cu': Cu, 'Mg': Mg, 'Ti': Ti, 'Mn': Mn, 'Al': Al,
        'bulk Prop.': bulk_prop,
        'Cov. Prop.': cov_prop,
        'UTS': best_ind.fitness.values[0]
    }

    all_data = []
    for ind in population:
        Cu, Mg, Ti, Mn = ind
        Al = 100 - (Cu + Mg + Ti + Mn)
        bulk_prop, cov_prop = calculate_bulk_and_cov(Al, Cu, Mg, Ti, Mn)

        all_data.append({
            'Temp': Temp,
            'QT': QT,
            'A1-t': A1_t,
            'ST1-t': ST1_t,
            'Cu': Cu, 'Mg': Mg, 'Ti': Ti, 'Mn': Mn, 'Al': Al,
            'bulk Prop.': bulk_prop,
            'Cov. Prop.': cov_prop,
            'UTS': ind.fitness.values[0]
        })

    return best_result, all_data


def main_ga_parallel():
    all_individuals_data = []
    best_per_condition = []

    param_list = []
    for Temp in Temp_options:
        for QT in QT_options:
            for A1_t in A1_t_options:
                for ST1_t in ST1_t_options:
                    param_list.append((Temp, QT, A1_t, ST1_t))

    max_workers = 128

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_ga_for_params, *params): params for params in param_list}

        for future in as_completed(futures):
            best_result, all_data = future.result()
            best_per_condition.append(best_result)
            all_individuals_data.extend(all_data)

            p = futures[future]
            print(f"Completed GA for params: Temp={p[0]}, QT={p[1]}, A1-t={p[2]}, ST1-t={p[3]}")

    all_df = pd.DataFrame(all_individuals_data)
    best_df = pd.DataFrame(best_per_condition)

    os.makedirs("GA_results_4", exist_ok=True)
    all_df.to_excel("GA_results_4/all_individuals.xlsx", index=False)
    best_df.to_excel("GA_results_4/best_per_condition.xlsx", index=False)

    print("\n✔ All results saved to 'GA_results_4/all_individuals.xlsx'")
    print("✔ Best result for each condition saved to 'GA_results_4/best_per_condition.xlsx'")


if __name__ == "__main__":
    main_ga_parallel()
