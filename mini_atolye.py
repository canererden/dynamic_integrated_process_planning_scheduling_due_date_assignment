import numpy as np
import pandas as pd
from copy import deepcopy
from math import ceil, log, cos, exp, fabs
import salabim as sim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
from pprint import pprint
import time
# import glob
from cycler import cycler
import codecs  # dosyaları utf8 kodlamasında kaydetmek için
from scipy.stats import truncnorm  # normal dağılıma uyan sayı üretmek için
import pickle
import json

plt.style.use('seaborn-deep')
# SF Conf
ddrule_size = 36
dsprule_size = 23
sum_to = 1

SEED = 12345
# SEED = int(input("Please enter random number generation seed: "))

np.random.seed(SEED)


class ProblemSet:
    DATA = [[4, 25, 50, 75, 100, 125, 150, 175, 200],  # 0. Is Sayilari
            [4, 10, 10, 10, 10, 10, 10, 10, 10],  # 1. Operasyon Sayilari
            [2, 5, 10, 15, 20, 25, 30, 35, 40],  # 2. Makine Sayilari
            [2, 5, 5, 5, 5, 3, 3, 3, 3],  # 3. Rota Sayilari
            [10, 150, 150, 100, 100, 75, 75, 50, 50]]  # iter_size

    def __init__(self, ps_id):
        self.ps_id = ps_id
        self.job_size = ProblemSet.DATA[0][ps_id]
        self.operation_size = ProblemSet.DATA[1][ps_id]
        self.machine_size = ProblemSet.DATA[2][ps_id]
        self.route_size = ProblemSet.DATA[3][ps_id]
        self.iter_size = ProblemSet.DATA[4][ps_id]


def sum_to_x(n, x):
    """gives n random value whose sum is equal to x"""
    values = [0.0, x] + list(np.random.uniform(low=0.0, high=x, size=n - 1))
    values.sort()
    return [values[i + 1] - values[i] for i in range(n)]


def exec_time(start, end):
    diff_time = end - start
    m, s = divmod(diff_time, 60)
    h, m = divmod(m, 60)
    s, m, h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
    print("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))
    return [h, m, s]


def make_second(liste):
    total_seconds = 0
    total_seconds += liste[2] + 60 * liste[1] + 60 * 60 * liste[0]
    return total_seconds


# mean ortalama, sd standart sapma, low upp alt ve üst limit
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    """https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy"""
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def writeToJSONFile(path, fileName, data):
    filePathNameWExt = './' + path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)


# Choose Shop Floor
at_no = 0
ps = ProblemSet(at_no)

machine_numbers = np.loadtxt(fname='inputs\\machine_numbers_' + str(
    at_no) + '.txt', dtype=int).reshape(ps.job_size, ps.route_size, ps.operation_size)

operation_durations = np.loadtxt(fname='inputs\\operation_durations_' + str(
    at_no) + '.txt', dtype=int).reshape(ps.job_size, ps.route_size, ps.operation_size)

weights = np.loadtxt('inputs/weights_' + str(at_no) + '.txt', dtype=float)

arrival_times = np.loadtxt('inputs/arrivals_' + str(at_no) + '.txt', dtype=int)


# Classes

class Chromosome:
    ddrule_values = list(range(0, ddrule_size))
    dsprule_values = list(range(0, dsprule_size))
    routes_values = []
    for i in range(ps.job_size):
        routes_values.append(list(range(0, ps.route_size)))

    def __init__(self, chr_id=None, particle_id=None):
        self.chr_id = chr_id
        self.env = sim.Environment(trace=False)
        self.env.jobs = sim.Queue('.jobs')
        self.env.joblist = [Job(chromosome=self, job_id=i)
                            for i in range(ps.job_size)]
        self.env.machinelist = [
            Machine(chromosome=self, mc_id=_) for _ in range(ps.machine_size)]
        self.genes = []  # pso'daki konum yani
        self.fitness = 0

        # PSO için
        self.particle_id = particle_id
        # values
        self.ddrule = -1
        self.dsprule = -1
        self.routes = []
        # probs
        self.ddrule_probs = []
        self.dsprule_probs = []
        self.routes_probs = []
        self.position_probs = self.ddrule_probs + \
                              self.dsprule_probs + self.routes_probs
        # pbest
        self.pbest = float('inf')  # Personal Best Fitness değeri
        self.pbest_ddrule_value = -1
        self.pbest_dsprule_value = -1
        self.pbest_routes_value = []
        self.pbest_ddrule_probs = []
        self.pbest_dsprule_probs = []
        self.pbest_routes_probs = []

        # velocity
        self.ddrule_velocity = []
        self.dsprule_velocity = []
        self.routes_velocity = []

    def set_positions(self):
        self.position_values = []
        self.position_values.append(self.ddrule)
        self.position_values.append(self.dsprule)
        self.position_values += self.routes

    def generate_genes(self):
        self.reset_genes()
        self.ddrule = np.random.randint(0, ddrule_size)
        self.dsprule = np.random.randint(0, dsprule_size)
        self.routes = np.random.randint(0, ps.route_size, size=ps.job_size)
        self.genes.append(self.ddrule)
        self.genes.append(self.dsprule)
        for r in self.routes:
            self.genes.append(r)

    def generate_data(self):
        self.env.machine_numbers = []
        self.env.operation_durations = []
        for i, j in enumerate(machine_numbers):
            self.env.machine_numbers.append(j[self.routes[i]])
        for i, j in enumerate(operation_durations):
            self.env.operation_durations.append(j[self.routes[i]])

    def reset_genes(self):
        self.ddrule = None
        self.dsprule = None
        self.routes = None
        self.genes = []

    def reset_job_machines(self):
        del self.env.joblist[:]
        del self.env.machinelist[:]

    def change_genes(self, genes):
        self.reset_genes()
        self.ddrule = genes[0]
        self.dsprule = genes[1]
        self.routes = genes[2:]
        self.genes = []
        self.genes.append(self.ddrule)
        self.genes.append(self.dsprule)
        for r in self.routes:
            self.genes.append(r)

    def due_date_assignment(self, q=1.4, k=2.5, w_k=0.6):
        #         q=0.4, k=1.5, w_k=0.3 olarak değiştiğinde tardy işler çıktı
        pav = sum(sum(self.env.operation_durations)) / ps.job_size
        if self.ddrule in [0, 3, 19, 22, 25, 28, 31, 34]:
            # print("q değeri için girildi")
            q = q * pav
            # print("q", q)
        elif self.ddrule in [1, 4, 20, 23, 26, 29, 32, 35]:
            q = (q + 0.5) * pav
        elif self.ddrule in [2, 5, 21, 24, 27, 30, 33, 36]:
            q = (q + 1) * pav
        if self.ddrule in [6, 9, 12, 15, 19, 20, 21, 28, 29, 30]:
            twkk = k
        elif self.ddrule in [7, 10, 13, 16, 22, 23, 24, 31, 32, 33]:
            twkk = k + 1
        elif self.ddrule in [8, 11, 14, 17, 25, 26, 27, 34, 35, 36]:
            twkk = k + 2
        if 0 <= self.ddrule <= 2:  # SLK Test edildi
            for i, job in enumerate(self.env.joblist):
                """print("iş{} için geliş zamanı {}, proses zamanı: {}, pav değeri: {}".format(
                    i, job.arrival_time, sum(self.env.operation_durations[i]), q))"""
                job.due_date_time = job.arrival_time + \
                                    sum(self.env.operation_durations[i]) + q
        elif 3 <= self.ddrule <= 5:  # WSLK # Ağırlık grubu sayısına göre bölündü Test edildi #TODO k parametresi ile optimize edilecek.
            # Ağırlığı küçük olana büyük katsayı, büyük olana küçük katsayı veriyoruz
            for i, job in enumerate(self.env.joblist):
                job.due_date_time = job.arrival_time + \
                                    sum(self.env.operation_durations[i]) + w_k * (1 / job.weight) * q
        elif 6 <= self.ddrule <= 8:  # TWK Test edildi
            for i, job in enumerate(self.env.joblist):
                job.due_date_time = job.arrival_time + twkk * \
                                    sum(self.env.operation_durations[i])
        elif 9 <= self.ddrule <= 11:  # WTWK Job.weight ile mi çarpılacak?
            for i, job in enumerate(self.env.joblist):
                job.due_date_time = job.arrival_time + \
                                    w_k * (1 / job.weight) * twkk * \
                                    sum(self.env.operation_durations[i])
        elif 12 <= self.ddrule <= 14:  # NOPPT
            for i, job in enumerate(self.env.joblist):
                # p1 = 0.5 * twkk
                # p2 = 0.5 * twkk
                # result = 5 * p1 * (ps.job_size * ps.operation_size) + p2 * sum(
                #    self.env.operation_durations[i]) + job.arrival_time  # TODO Noper toplam operasyon sayısı mı?
                # if result <= sum(self.env.operation_durations[i]):
                #    result = sum(self.env.operation_durations[i])
                # job.due_date_time = result
                job.due_date_time = job.arrival_time + \
                                    sum(self.env.operation_durations[i]
                                        ) + 5 * twkk * len(operation_durations)
        elif 15 <= self.ddrule <= 17:  # WNOPPT
            for i, job in enumerate(self.env.joblist):
                job.due_date_time = job.arrival_time + w_k * (1 / job.weight) * \
                                    sum(self.env.operation_durations[i]) + \
                                    5 * w_k * (1 / job.weight) * twkk * len(operation_durations)
        elif self.ddrule == 18:  # RDM
            """pav ve 3pav  üretilen normal dağılıma uyan bir sayı ile geliş zamanını topladık."""
            for i, job in enumerate(self.env.joblist):
                X = get_truncated_normal(mean=3 * pav, sd=pav, low=0, upp=6 * pav)
                job.due_date_time = job.arrival_time + X.rvs()
        elif 19 <= self.ddrule <= 27:  # PPW
            for i, job in enumerate(self.env.joblist):
                job.due_date_time = job.arrival_time + q + twkk * \
                                    sum(self.env.operation_durations[i])
        elif 28 <= self.ddrule <= 36:  # WPPW
            for i, job in enumerate(self.env.joblist):
                job.due_date_time = job.arrival_time + w_k * (1 / job.weight) * 0.75 * q + w_k * (
                            1 / job.weight) * 0.75 * twkk * \
                                    sum(self.env.operation_durations[i])

    def run_simulation(self, yazdir=False):
        self.fitness = 0.0
        self.env.trace(yazdir)
        self.env.run()

    def calculate_fitness(self):
        performance = 0
        for i, job in enumerate(self.env.joblist):
            tardiness = max(job.departure_time - job.due_date_time, 0)
            earliness = max(job.due_date_time - job.departure_time, 0)
            penalty_dd = job.weight * \
                         ((8 / 480) * (job.due_date_time - job.arrival_time))
            """print("tardiness: {}, earliness: {}, penalty_dd: {}".format(
                tardiness, earliness, penalty_dd))"""
            if tardiness != 0:
                penalty_t = job.weight * (6 + 6 * (tardiness / 480))
            else:
                penalty_t = 0
            if earliness != 0:
                penalty_e = job.weight * (5 + 4 * (earliness / 480))
            else:
                penalty_e = 0
            penalty_total = penalty_dd + penalty_e + penalty_t
            performance += penalty_total
            """print("penalty_t: {}, penalty_e: {}, penalty_dd: {}, penalty_total: {}, performance: {}".format(
                penalty_t, penalty_e, penalty_dd, penalty_total, performance))"""
        self.fitness = performance

    def __str__(self):
        return self.genes.__str__()

    def __repr__(self):
        return self.genes.__str__()

    def __eq__(self):
        return self.genes.__str__()


class Machine(sim.Component):

    def setup(self, chromosome, mc_id):
        self.mc_id = mc_id
        self.queue = sim.Queue(self.name() + '.queue')
        self.chromosome = chromosome
        self.name('machine' + str(mc_id))
        self.op_list = []

    def process(self):
        while True:
            sel_operation = None
            yield self.hold(0)
            if self.queue:  # there is an operation in queue
                # Dispatching rules
                if self.chromosome.dsprule == 0:  # WATC
                    k = 1
                    # sel_operation = self.queue.head()
                    # print"self.queue: {}".format(self.queue))
                    watc = 0
                    for operation in self.queue:
                        # printoperation)
                        processing_time = sum(
                            op.duration for op in operation.job.operation_list)
                        # print"processing_time: {}".format(processing_time))
                        dif = operation.job.due_date_time - processing_time - self.chromosome.env.now()
                        p_ort = sum(
                            op.duration / ps.operation_size for op in operation.job.operation_list)
                        # printp_ort, "p_ort")
                        this_watc = (operation.job.weight / operation.duration) * \
                                    (2.71 ** ((max(dif, 0)) / (k * p_ort)))
                        # printthis_watc,"this_watc")
                        if this_watc > watc:
                            watc = this_watc
                            sel_operation = operation
                            # printsel_operation, "sel_operation")
                elif self.chromosome.dsprule == 1:  # WATC
                    k = 2
                    # sel_operation = self.queue.head()
                    watc = 0
                    for operation in self.queue:
                        processing_time = sum(
                            op.duration for op in operation.job.operation_list)
                        dif = operation.job.due_date_time - processing_time - self.chromosome.env.now()
                        p_ort = sum(
                            op.duration / ps.operation_size for op in operation.job.operation_list)
                        this_watc = (operation.job.weight / operation.duration) * \
                                    (2.71 ** ((max(dif, 0)) / (k * p_ort)))
                        if this_watc > watc:
                            watc = this_watc
                            sel_operation = operation
                elif self.chromosome.dsprule == 2:  # WATC
                    k = 3
                    # sel_operation = self.queue.head()
                    watc = 0
                    for operation in self.queue:
                        processing_time = sum(
                            op.duration for op in operation.job.operation_list)
                        dif = operation.job.due_date_time - processing_time - self.chromosome.env.now()
                        p_ort = sum(
                            op.duration / ps.operation_size for op in operation.job.operation_list)
                        this_watc = (operation.job.weight / operation.duration) * \
                                    (2.71 ** ((max(dif, 0)) / (k * p_ort)))
                        if this_watc > watc:
                            watc = this_watc
                            sel_operation = operation
                elif self.chromosome.dsprule == 3:  # ATC
                    k = 1
                    # sel_operation = self.queue.head()
                    atc = 0
                    for operation in self.queue:
                        processing_time = sum(
                            op.duration for op in operation.job.operation_list)
                        dif = operation.job.due_date_time - processing_time - self.chromosome.env.now()
                        p_ort = sum(
                            op.duration / ps.operation_size for op in operation.job.operation_list)
                        this_atc = (1 / operation.duration) * \
                                   (2.71 ** ((max(dif, 0)) / (k * p_ort)))
                        if this_atc > atc:
                            atc = this_atc
                            sel_operation = operation
                elif self.chromosome.dsprule == 4:  # ATC
                    k = 2
                    # sel_operation = self.queue.head()
                    atc = 0
                    for operation in self.queue:
                        processing_time = sum(
                            op.duration for op in operation.job.operation_list)
                        dif = operation.job.due_date_time - processing_time - self.chromosome.env.now()
                        p_ort = sum(
                            op.duration / ps.operation_size for op in operation.job.operation_list)
                        this_atc = (1 / operation.duration) * \
                                   (2.71 ** ((max(dif, 0)) / (k * p_ort)))
                        if this_atc > atc:
                            atc = this_atc
                            sel_operation = operation
                elif self.chromosome.dsprule == 5:  # ATC
                    k = 3
                    # sel_operation = self.queue.head()
                    atc = 0
                    for operation in self.queue:
                        processing_time = sum(
                            op.duration for op in operation.job.operation_list)
                        dif = operation.job.due_date_time - processing_time - self.chromosome.env.now()
                        p_ort = sum(
                            op.duration / ps.operation_size for op in operation.job.operation_list)
                        this_atc = (1 / operation.duration) * \
                                   (2.71 ** ((max(dif, 0)) / (k * p_ort)))
                        if this_atc > atc:
                            atc = this_atc
                            sel_operation = operation
                elif self.chromosome.dsprule == 6:  # WMS
                    # sel_operation = self.queue.head()
                    slack = -50000
                    for operation in self.queue:
                        processing_time = sum(
                            op.duration for op in operation.job.operation_list)
                        dif = operation.job.due_date_time - processing_time - self.chromosome.env.now()
                        this_slack = -dif * operation.job.weight
                        if this_slack > slack:
                            slack = this_slack
                            sel_operation = operation
                elif self.chromosome.dsprule == 7:  # MS
                    # sel_operation = self.queue.head()
                    slack = -50000
                    for operation in self.queue:
                        processing_time = sum(
                            op.duration for op in operation.job.operation_list)
                        dif = operation.job.due_date_time - processing_time - self.chromosome.env.now()
                        this_slack = -dif
                        if this_slack > slack:
                            slack = this_slack
                            sel_operation = operation
                elif self.chromosome.dsprule == 8:  # WSPT
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = sum(
                            op.job.weight / op.duration for op in operation.job.operations)
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 9:  # SPT
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = sum(
                            1 / op.duration for op in operation.job.operations)
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 10:  # WLPT
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = sum(
                            op.duration / op.job.weight for op in operation.job.operations)
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 11:  # LPT
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = sum(
                            op.duration for op in operation.job.operations)
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 12:  # WSOT
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = operation.job.weight / operation.duration
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 13:  # SOT
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = 1 / operation.duration
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 14:  # WLOT
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = operation.duration / operation.job.weight
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 15:  # LOT
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = operation.duration
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 16:  # EDD
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = 1 / operation.job.due_date_time
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 17:  # WEDD
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = operation.job.weight / operation.job.due_date_time
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 18:  # ERD
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = 1 / operation.job.arrival_time
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 19:  # WERD
                    # sel_operation = self.queue.head()
                    sel_sum = 0
                    for operation in self.queue:
                        this_sum = operation.job.weight / operation.job.arrival_time
                        if this_sum > sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                elif self.chromosome.dsprule == 20:  # SIRO
                    sel_operation = np.random.choice(self.queue)
                elif self.chromosome.dsprule == 21:  # FIFO
                    sel_operation = self.queue.head()
                elif self.chromosome.dsprule == 22:  # LIFO
                    sel_operation = self.queue.tail()
                else:
                    sel_operation = np.random.choice(self.queue)
                sel_operation.start_time = self.chromosome.env.now()
                yield self.hold(sel_operation.duration)
                sel_operation.leave()  # leave both job operations and machine queue
                sel_operation.finish_time = self.chromosome.env.now()  # for departure times
                sel_operation.job.activate()
            else:
                yield self.passivate()


class Operation(sim.Component):
    def setup(self, job, machine, duration):
        self.job = job
        self.machine = machine
        self.duration = duration
        self.start_time = 0
        self.finish_time = 0

    def __str__(self):
        return str(self.job)


class Job(sim.Component):
    def setup(self, job_id, chromosome):
        self.job_id = job_id
        self.name('job' + str(job_id))
        self.chromosome = chromosome
        self.operations = sim.Queue(name=self.name() + '.operations')
        self.operation_list = []
        self.arrival_time = arrival_times[job_id]
        self.weight = weights[job_id]
        self.due_date_time = -1
        self.departure_time = -1
        self.color = None

    def process(self):
        yield self.hold(self.arrival_time)
        self.enter(self.chromosome.env.jobs)
        for j in range(ps.operation_size):
            machine = self.chromosome.env.machinelist[self.chromosome.env.machine_numbers[self.job_id][j]]
            duration = self.chromosome.env.operation_durations[self.job_id][j]
            op = Operation(
                name='operation.' +
                     str(self.job_id) + '.' + str(j) + ' m=' +
                     machine.name() + ' d=' + str(duration),
                job=self, machine=machine, duration=duration)
            op.enter(self.operations)
            self.operation_list.append(op)
        while self.operations:
            next_operation = self.operations.head()
            next_operation.enter(next_operation.machine.queue)
            if next_operation.machine.ispassive():
                next_operation.machine.activate()
            yield self.passivate()
        self.leave()
        self.departure_time = self.chromosome.env.now()
        if self.chromosome.env.peek() == sim.inf:  # check for end of simulation
            self.chromosome.env.main().activate()

    def __str__(self):
        return 'job' + self.job_id.__str__()

    def __repr__(self):
        return 'job' + self.job_id.__str__()


class Population:

    def __init__(self, size):
        self.size = size
        self.chromosomes = []
        self.best = 0
        self.avg = 0
        self.worst = 0

    def initialize_population(self, gen_num):
        for i in range(self.size):
            chrom = Chromosome(chr_id=i)
            self.chromosomes.append(chrom)
        for ind in self.chromosomes:
            ind.generate_genes()
            ind.generate_data()
            ind.due_date_assignment()
            ind.run_simulation()
            ind.calculate_fitness()
        self.sort_pop()
        self.print_pop(generation_number=gen_num)

    def sort_pop(self):
        return self.chromosomes.sort(key=lambda x: x.fitness, reverse=False)

    def print_pop(self, generation_number):
        # best avg worst
        if mode == 'ga':
            ga_text_file = codecs.open("results\\ga-" + str(at_no) + ".txt", "a")
            self.best = round(self.chromosomes[0].fitness, 3)
            _total = 0
            for ind in self.chromosomes:
                _total += ind.fitness
            self.avg = round(_total / self.size, 2)
            self.worst = round(self.chromosomes[-1].fitness, 2)
            ga_text_file.write(
                "--------------------------------------------------\n")
            ga_text_file.write(
                "Population # {} | Best: {} Avg: {}, Worst: {} ".format(generation_number, self.best, self.avg,
                                                                        self.worst))
            ga_text_file.write(
                "\n--------------------------------------------------\n")
            for i, x in enumerate(self.chromosomes):
                ga_text_file.write("Individual #{} {} |Fitness: {}\n".format(
                    i, x.genes, round(x.fitness, 2)))
            ga_text_file.close()

        elif mode == 'ha':
            ha_text_file = open("ha-" + str(at_no) + ".txt", "w")
            self.best = round(self.chromosomes[0].fitness, 3)
            _total = 0
            for ind in self.chromosomes:
                _total += ind.fitness
            self.avg = round(_total / self.size, 2)
            self.worst = round(self.chromosomes[-1].fitness, 2)
            ha_text_file.write(
                "--------------------------------------------------\n")
            ha_text_file.write(
                "Population # {} | Best: {} Avg: {}, Worst: {} ".format(generation_number, self.best, self.avg,
                                                                        self.worst))
            ha_text_file.write(
                "\n--------------------------------------------------\n")
            for i, x in enumerate(self.chromosomes):
                ha_text_file.write("Individual #{} {} |Fitness: {}\n".format(
                    i, x.genes, round(x.fitness, 2)))
            ha_text_file.close()

    def reap_pop(self):
        self.chromosomes = self.chromosomes[:self.size]

    def __str__(self):
        return self.chromosomes.__str__()

    def __repr__(self):
        return self.chromosomes.__str__()


class Swarm:
    w = 2  # constant inertia weight
    c_1 = 10  # cognitive parameters
    c_2 = 15  # social parameters

    def __init__(self, size, generation):
        self.size = size
        self.generation = generation
        self.particles = []  # particle sınıfındaki particle lar
        self.gbest = float('inf')
        self.gbest_particle = None


def trace_schedule(chromosome):
    chromosome.generate_data()
    chromosome.due_date_assignment()
    chromosome.run_simulation(yazdir=True)
    chromosome.calculate_fitness()


def print_schedule(chromosome):
    print("--------------------------------------------------")
    print("Chromosome",
          chromosome.chr_id, "Fitness", chromosome.fitness, "Route", chromosome.routes)
    print("Due Date Rule", chromosome.ddrule,
          "Dispatchine Rule", chromosome.dsprule)
    print("--------------------------------------------------")
    for i, job in enumerate(chromosome.env.joblist):
        print("job", i, " arrival_time= ", job.arrival_time, "departure_time=", job.departure_time, "due-date=",
              job.due_date_time, "weight= ", job.weight)
        for k, operation in enumerate(job.operation_list):
            print(operation.name(), " start_time=",
                  operation.start_time, " finish_time=",
                  operation.finish_time)
        print("--------------------------------------------------")


def gantt_cizdir(chromosome):
    fig, ax = plt.subplots()
    cmap = cm.ScalarMappable(col.Normalize(0, 99), cm.hsv)
    # colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    job_list = chromosome.env.joblist
    machine_list = chromosome.env.machinelist
    for job in job_list:
        job.color = cmap.to_rgba(np.random.randint(99))

    departures = []
    # add tasks to the machines
    for job in job_list:
        departures.append(job.departure_time)
        for op in job.operation_list:
            op.machine.op_list.append(op)

    departures.sort()

    ax.set_xlabel('seconds')
    ax.set_ylabel('Machines')
    ax.set_yticks(list(range(10, 15 * len(machine_list) + 10, 15)))
    ax.set_yticklabels(['m' + str(i + 1) for i in range(len(machine_list))])
    ax.set_xticks(
        list(range(0, departures[-1] + 20, int(departures[-1] / 10))))
    ax.grid(color='lightblue', linestyle='-.', linewidth=0.1)

    for i, machine in enumerate(machine_list):
        xranges = [(op.start_time, op.duration) for op in machine.op_list]
        yrange = ((i * 15 + 5), 10)
        colors = ['b', 'yellow', 'r', 'm']
        ax.broken_barh(xranges, yrange, facecolors=colors,
                       edgecolor="black", linewidth=0.25)
    for j, job in enumerate(job_list):
        ax.annotate('job arrival', (job.arrival_time, (job.operation_list[0].machine.mc_id) * 15 + 5),
                    xytext=(0.5, 0.5), textcoords='axes fraction',
                    arrowprops={'facecolor': 'red', 'shrink': 0.002, 'width': 0.1,
                                'headwidth': 5, 'headlength': 10, 'alpha': 0.4},
                    fontsize=10,
                    horizontalalignment='right', verticalalignment='bottom')
    # for job in job_list:
    #   ax.annotate('job'+ 'arrives', (job.arrival_time, (job.operation_list[0].machine.mc_id)*15+5),
    #           xytext=(0.89, 0.97), textcoords='axes fraction',
    #               arrowprops={'facecolor': 'black', 'shrink': 0.002, 'width': 0.1,'headwidth':5, 'headlength': 10},
    #           fontsize=10,
    #           horizontalalignment='right', verticalalignment='top')
    legend_elements = [plt.Line2D(
        [0], [0], color=colors[j], label=job) for j, job in enumerate(job_list)]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=5, edgecolor='b')
    plt.tight_layout()
    plt.title(str(ps.job_size) + ' jobs, ' + str(ps.operation_size) + ' operations, ' + str(
        ps.machine_size) + ' machines, ' + str(ps.route_size) + ' routes')
    plt.savefig(fname="gantt_job_" + str(ps.job_size) + "_" + str(ps.machine_size) + "_machines.svg", dpi=300,
                facecolor='w', edgecolor='b', orientation='portrait', papertype=None,
                format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
    plt.show()
