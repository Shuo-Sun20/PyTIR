from typing import Dict, List, Callable, Tuple
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from random import random, choice
from time import perf_counter
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from _collections_abc import dict_keys, dict_values, Iterable

from tqdm import tqdm
from deap import creator, base, tools, algorithms

from .locale import _

OUTPUT_FUNC = Callable[[str], None]
EVALUATE_FUNC = Callable[[dict], dict]
KEY_FUNC = Callable[[list], float]


# Create individual class used in genetic algorithm optimization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """
    Setting for runnning optimization.
    """

    def __init__(self):
        """"""
        self.params: Dict[str, List] = {}
        self.target_name: str = ""

    def add_parameter(
        self, name, start, end = None, step = None):
        """"""
        if end is None and step is None:
            self.params[name] = [start]
            return True, _("固定参数添加成功")

        if start >= end:
            return False, _("参数优化起始点必须小于终止点")

        if step <= 0:
            return False, _("参数优化步进必须大于0")

        value: float = start
        value_list: List[float] = []

        while value <= end:
            value_list.append(value)
            value += step

        self.params[name] = value_list

        return True, _("范围参数添加成功，数量{}").format(len(value_list))

    def set_target(self, target_name):
        """"""
        self.target_name = target_name

    def generate_settings(self):
        """"""
        keys: dict_keys = self.params.keys()
        values: dict_values = self.params.values()
        products: list = list(product(*values))

        settings: list = []
        for p in products:
            setting: dict = dict(zip(keys, p))
            settings.append(setting)

        return settings


def check_optimization_setting(
    optimization_setting, output = print):
    """"""
    if not optimization_setting.generate_settings():
        output(_("优化参数组合为空，请检查"))
        return False

    if not optimization_setting.target_name:
        output(_("优化目标未设置，请检查"))
        return False

    return True


def run_bf_optimization(
    evaluate_func, optimization_setting, key_func, max_workers = None, output = print):
    """Run brutal force optimization"""
    settings: List[Dict] = optimization_setting.generate_settings()

    output(_("开始执行穷举算法优化"))
    output(_("参数优化空间：{}").format(len(settings)))

    start: int = perf_counter()

    with ProcessPoolExecutor(
        max_workers,
        mp_context=get_context("spawn")
    ) as executor:
        it: Iterable = tqdm(
            executor.map(evaluate_func, settings),
            total=len(settings)
        )
        results: List[Tuple] = list(it)
        results.sort(reverse=True, key=key_func)

        end: int = perf_counter()
        cost: int = int((end - start))
        output(_("穷举算法优化完成，耗时{}秒").format(cost))

        return results


def run_ga_optimization(
    evaluate_func, optimization_setting, key_func, max_workers = None, population_size = 100, ngen_size = 30, output = print):
    """Run genetic algorithm optimization"""
    # Define functions for generate parameter randomly
    buf: List[Dict] = optimization_setting.generate_settings()
    settings: List[Tuple] = [list(d.items()) for d in buf]

    def generate_parameter() -> list:
        """"""
        return choice(settings)

    def mutate_individual(individual: list, indpb: float) -> tuple:
        """"""
        size: int = len(individual)
        paramlist: list = generate_parameter()
        for i in range(size):
            if random() < indpb:
                individual[i] = paramlist[i]
        return individual,

    # Set up multiprocessing Pool and Manager
    ctx: BaseContext = get_context("spawn")
    with ctx.Manager() as manager, ctx.Pool(max_workers) as pool:
        # Create shared dict for result cache
        cache: Dict[Tuple, Tuple] = manager.dict()

        # Set up toolbox
        toolbox: base.Toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_parameter)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, indpb=1)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("map", pool.map)
        toolbox.register(
            "evaluate",
            ga_evaluate,
            cache,
            evaluate_func,
            key_func
        )

        total_size: int = len(settings)
        pop_size: int = population_size                      # number of individuals in each generation
        lambda_: int = pop_size                              # number of children to produce at each generation
        mu: int = int(pop_size * 0.8)                        # number of individuals to select for the next generation

        cxpb: float = 0.95         # probability that an offspring is produced by crossover
        mutpb: float = 1 - cxpb    # probability that an offspring is produced by mutation
        ngen: int = ngen_size    # number of generation

        pop: list = toolbox.population(pop_size)

        # Run ga optimization
        output(_("开始执行遗传算法优化"))
        output(_("参数优化空间：{}").format(total_size))
        output(_("每代族群总数：{}").format(pop_size))
        output(_("优良筛选个数：{}").format(mu))
        output(_("迭代次数：{}").format(ngen))
        output(_("交叉概率：{:.0%}").format(cxpb))
        output(_("突变概率：{:.0%}").format(mutpb))

        start: int = perf_counter()

        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu,
            lambda_,
            cxpb,
            mutpb,
            ngen,
            verbose=True
        )

        end: int = perf_counter()
        cost: int = int((end - start))

        output(_("遗传算法优化完成，耗时{}秒").format(cost))

        results: list = list(cache.values())
        results.sort(reverse=True, key=key_func)
        return results


def ga_evaluate(
    cache, evaluate_func, key_func, parameters):
    """
    Functions to be run in genetic algorithm optimization.
    """
    tp: tuple = tuple(parameters)
    if tp in cache:
        result: tuple = cache[tp]
    else:
        setting: dict = dict(parameters)
        result: dict = evaluate_func(setting)
        cache[tp] = result

    value: float = key_func(result)
    return (value, )
