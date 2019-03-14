import codecs  # dosyaları utf8 kodlamasında kaydetmek için

import numpy as np
import salabim as sim
from scipy.stats import truncnorm
import time
from copy import deepcopy
from math import ceil, log, cos, exp, fabs
import matplotlib.pyplot as plt

# TODO

# SF Conf
ddrule_size = 36
dsprule_size = 23
sum_to = 1
pop_size = 10
at_no = int(input("Enter shop floor number"))


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


ps = ProblemSet(at_no)


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


# mean ortalama, sd standart sapma, low upp alt ve üst limit
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    """https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy"""
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


machine_numbers = np.loadtxt(fname='inputs\\machine_numbers_' + str(
    at_no) + '.txt', dtype=int).reshape(ps.job_size, ps.route_size, ps.operation_size)

operation_durations = np.loadtxt(fname='inputs\\operation_durations_' + str(
    at_no) + '.txt', dtype=int).reshape(ps.job_size, ps.route_size, ps.operation_size)

weights = np.loadtxt('inputs/weights_' + str(at_no) + '.txt', dtype=float)

arrival_times = np.loadtxt('inputs/arrivals_' + str(at_no) + '.txt', dtype=int)


class Chromosome:
    ddrule_values = list(range(0, ddrule_size))
    dsprule_values = list(range(0, dsprule_size))
    routes_values = []
    for i in range(ps.job_size):
        routes_values.append(list(range(0, ps.route_size)))

    def __init__(self, chr_id=None, particle_id=None):
        self.chr_id = chr_id
        self.ddrule = -1
        self.dsprule = -1
        self.routes = []
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
        self.ddrule_probs = []
        self.dsprule_probs = []
        self.routes_probs = []
        self.position_probs = self.ddrule_probs + self.dsprule_probs + self.routes_probs
        self.pbest = float('inf')  # Personal Best Fitness değeri
        self.pbest_ddrule_values = []
        self.pbest_dsprule_values = []
        self.pbest_routes_values = []
        self.pbest_ddrule_value = -1
        self.pbest_dsprule_value = -1
        self.pbest_routes_value = []

    def set_positions(self):
        self.position_values = []
        self.position_values.append(self.ddrule)
        self.position_values.append(self.dsprule)
        self.position_values += self.routes

    def __str__(self):  # real signature unknown
        """ Return str(self). """
        pass

    def generate_genes(self):
        # Rastgele kromozoma gen oluşturur.
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
        # WSLK # Ağırlık grubu sayısına göre bölündü Test edildi #TODO k parametresi ile optimize edilecek.
        elif 3 <= self.ddrule <= 5:
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
                penalty_t = job.weight * (10 + 12 * (tardiness / 480))
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

        ga_text_file = codecs.open("results\\ga-" + str(at_no) + ".txt", "a+")
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

    def reap_pop(self):
        self.chromosomes = self.chromosomes[:self.size]

    def __str__(self):
        return self.chromosomes.__str__()

    def __repr__(self):
        return self.chromosomes.__str__()


all_gbests = []


class Swarm:
    w = 2  # constant inertia weight
    c_1 = 10  # cognitive parameters
    c_2 = 15  # social parameters

    def __init__(self, size, generation):
        self.size = size
        self.generation = generation
        self.best = -1
        self.avg = -1
        self.worst = -1
        self.particles = []  # particle sınıfındaki particle lar
        self.gbest = float('inf')
        self.gbest_particle = None

    def initialize_swarm(self):
        print("-" * 100)
        print("Initial Swarm", "w=", Swarm.w, "c1=", Swarm.c_1, "c2=", Swarm.c_2)
        print("-" * 100)
        print("{0:<5} {1:<5} {2:<5} {3:<30} {4:<6} {5:<6} {6:<10}".format("id", "dd", "dsp", "routes",
                                                                          "fitness", "pbest", "swarm: 0"))
        print("-" * 100)
        for i in range(self.size):
            self.particles.append(Chromosome(particle_id=i))
        for particle in self.particles:
            # Randomly generate first swarm
            #particle.ddrule_probs = sum_to_x(ddrule_size, sum_to)  # 0 ile 1 arasında ddrule size kadar rastgele.
            particle.ddrule_probs = [(1/ddrule_size) for i in range(ddrule_size)]
            #particle.dsprule_probs = sum_to_x(dsprule_size, sum_to)
            particle.dsprule_probs =[(1/dsprule_size) for i in range(dsprule_size)]
            particle.routes_probs = []
            for i in range(ps.job_size):
                particle.routes_probs.append(sum_to_x(ps.route_size, sum_to))

            # Make sample -- Olasılıklardan değer seç
            particle.ddrule = np.random.choice(Chromosome.ddrule_values, 1, p=particle.ddrule_probs)[0]
            particle.dsprule = np.random.choice(Chromosome.dsprule_values, 1, p=particle.dsprule_probs)[0]
            particle.routes = []
            for i in range(ps.job_size):
                prob = particle.routes_probs[i]
                value = Chromosome.routes_values[i]
                route_value = np.random.choice(value, size=1, p=prob)[0]
                particle.routes.append(route_value)

            # Fitness değerlerini hesapla
            particle.set_positions()
            particle.change_genes(particle.position_values)
            particle.generate_data()
            particle.due_date_assignment()
            particle.run_simulation(yazdir=False)
            particle.calculate_fitness()
            particle.pbest = particle.fitness
            particle.pbest_ddrule_probs = particle.ddrule_probs
            particle.pbest_dsprule_probs = particle.dsprule_probs
            particle.pbest_routes_probs = particle.routes_probs
            particle.pbest_ddrule_value = particle.ddrule
            particle.pbest_dsprule_value = particle.dsprule
            particle.pbest_routes_values = particle.routes
            particle.velocity_ddrule = [0] * ddrule_size
            particle.velocity_dsprule = [0] * dsprule_size
            particle.velocity_routes = []
            for i in range(ps.job_size):
                particle.velocity_routes.append([0] * ps.route_size)

            # eğer sürüdeki bir parçacık gbest'ten daha iyiyse gbest i değiştir.
            if particle.fitness <= self.gbest:
                self.gbest = particle.fitness
                self.gbest_particle = particle

            # Başlangıç Sürüsünü Yazdır
            print("{0:<5} {1:<5} {2:<5} {3:<30} {4:<6} {5:<6}".format(particle.particle_id, particle.ddrule,
                                                                      particle.dsprule, str(particle.routes),
                                                                      round(particle.fitness, 2),
                                                                      round(particle.pbest, 2)))
            # Olasılıkları test et.
            # print(
            #    "dd: {}, dsp: {}, rt: {}".format(particle.ddrule_probs, particle.dsprule_probs, particle.routes_probs))

        # Gbest'i yazdır
        print("-" * 100)
        print("gbest: \t{}\tgbest particle:\t{}\n\n".format(round(self.gbest, 2), self.gbest_particle.particle_id))

    def run(self, iter_size):
        for i in range(iter_size):
            self.generation = i
            print("{0:<5} {1:<5} {2:<5} {3:<30} {4:<6} {5:<6} swarm: {6:<4}".format("id", "dd", "dsp",
                                                                                    "routes",
                                                                                    "fitness", "pbest",
                                                                                    str(self.generation + 1)))
            print("-" * 100)
            total_fitness = 0
            for p, particle in enumerate(self.particles):
                # random variables
                r1 = np.random.random()
                r2 = np.random.random()
                # new ddrule
                inertia_ddrule = [x * Swarm.w for x in particle.velocity_ddrule]
                cognitive_ddrule = [r1 * Swarm.c_1 * (a + b) for a, b in
                                    zip(particle.pbest_ddrule_probs, particle.ddrule_probs)]  # TODO
                social_ddrule = [r2 * Swarm.c_2 * (a + b) for a, b in
                                 zip(self.gbest_particle.ddrule_probs, particle.ddrule_probs)]
                ddrule_velocity = [a + b + c for a, b, c in zip(inertia_ddrule, cognitive_ddrule, social_ddrule)]
                new_ddrule_probs = [a + b for a, b in zip(particle.ddrule_probs, ddrule_velocity)]
                normalized_ddrule_probs = [x / sum(new_ddrule_probs) for x in new_ddrule_probs]
                print("normalized_ddrule_probs", normalized_ddrule_probs)

                # new dsprule
                inertia_dsprule = [x * Swarm.w for x in particle.velocity_dsprule]
                cognitive_dsprule = [r1 * Swarm.c_1 * (a + b) for a, b in
                                     zip(particle.pbest_dsprule_probs, particle.dsprule_probs)]  # TODO
                social_dsprule = [r2 * Swarm.c_2 * (a + b) for a, b in
                                  zip(self.gbest_particle.dsprule_probs, particle.dsprule_probs)]
                dsprule_velocity = [a + b + c for a, b, c in zip(inertia_dsprule, cognitive_dsprule, social_dsprule)]
                new_dsprule_probs = [a + b for a, b in zip(particle.dsprule_probs, dsprule_velocity)]
                normalized_dsprule_probs = [x / sum(new_dsprule_probs) for x in new_dsprule_probs]

                # new routes
                for i in range(ps.job_size):
                    # New routes
                    inertia_routes = [x * Swarm.w for x in particle.velocity_routes[i]]
                    cognitive_routes = [r1 * Swarm.c_1 * (a + b) for a, b in
                                        zip(particle.pbest_routes_probs[i], particle.routes_probs[i])]  # TODO
                    social_routes = [r2 * Swarm.c_2 * (a + b) for a, b in
                                     zip(self.gbest_particle.routes_probs[i], particle.routes_probs[i])]
                    routes_velocity = [a + b + c for a, b, c in zip(inertia_routes, cognitive_routes, social_routes)]
                    new_routes_probs = [a + b for a, b in zip(particle.routes_probs[i], routes_velocity)]
                    normalized_routes_probs = [x / sum(new_routes_probs) for x in new_routes_probs]

                # New ddrule value
                particle.ddrule = np.random.choice(Chromosome.ddrule_values, size=1, p=normalized_ddrule_probs)[0]
                # New dsprule value
                particle.dsprule = np.random.choice(Chromosome.dsprule_values, size=1, p=normalized_dsprule_probs)[
                    0]
                # New routes value
                for i in range(ps.job_size):
                    particle.routes[i] = \
                        np.random.choice(Chromosome.routes_values[i], size=1, p=normalized_routes_probs)[0]
                # Print new positions
                # print("particle.ddrule", particle.ddrule)
                # print("particle.dsprule", particle.dsprule)
                # print("particle.routes", particle.routes)

                # Calculate new fitness
                particle.generate_data()
                particle.due_date_assignment()
                particle.run_simulation()
                particle.calculate_fitness()
                # print(p, ". ", "particle.fitness", particle.fitness)

                # Pbest
                if particle.fitness <= particle.pbest:
                    particle.pbest = particle.fitness
                    particle.pbest_ddrule_value = particle.ddrule
                    particle.pbest_dsprule_value = particle.dsprule
                    particle.pbest_routes_value = particle.routes

                # Gbest
                if particle.fitness <= self.gbest:
                    self.gbest = particle.fitness
                    self.gbest_particle = particle
                print("{0:<5} {1:<5} {2:<5} {3:<30} {4:<6} {5:<6}".format(particle.particle_id, particle.ddrule,
                                                                          particle.dsprule, str(particle.routes),
                                                                          round(particle.fitness, 2),
                                                                          round(particle.pbest, 2)))

                # Olasılıkları test et.
                # print(
                #    "dd: {}, dsp: {}, rt: {}".format(normalized_ddrule_probs, normalized_dsprule_probs,
                #                                     normalized_routes_probs))
                # total fitness
                total_fitness += particle.fitness

            self.avg = total_fitness / self.size
            all_gbests.append(self.gbest)
            print("-" * 100)
            print(
                "gbest: {} gbest_particle: {} avg: {}\n\n".format(round(self.gbest, 2), self.gbest_particle.particle_id,
                                                                  round(self.avg, 2)))

    def run2(self, iter_size):
        for i in range(iter_size):
            self.generation = i
            print("{0:<5} {1:<5} {2:<5} {3:<30} {4:<6} {5:<6} swarm: {6:<4}".format("id", "dd", "dsp",
                                                                                    "routes",
                                                                                    "fitness", "pbest",
                                                                                    str(self.generation + 1)))
            print("-" * 100)
            total_fitness = 0
            gbest = None
            pbest = None
            for p, particle in enumerate(self.particles):

                if particle.fitness < pbest.fitness:
                    pbest=particle
                    pass
                pass
            pass
        pass





class GeneticAlgorithm:
    start_time = time.time()
    finish_time = time.time()
    pc = 0.7
    pe = 0.5
    pm = 0.3
    gene_size = ps.job_size + 2
    num_cross = 6  # number of crossover population
    num_mut = 4  # number of mutation population
    num_points_crossover = ceil(gene_size * 0.1)
    num_points_mutation = ceil(gene_size * 0.3)
    prob_chr = [0.3, 0.2, 0.15, 0.12, 0.10, 0.07,
                0.03, 0.02, 0.006, 0.004]  # chromosome prob, ranking probability
    prob_genes = [0.25, 0.25] + sum_to_x(ps.job_size, 0.5)

    def __init__(self):
        self.pop_list = []

    def initialize_population(self, random_iter_size):
        # Random iter size adet rastgele arama yapacak.
        rs_ga_solutions = []
        for i in range(random_iter_size):
            solution_rs_ga = Chromosome()
            solution_rs_ga.generate_genes()
            solution_rs_ga.generate_data()
            solution_rs_ga.due_date_assignment()
            solution_rs_ga.run_simulation()
            solution_rs_ga.calculate_fitness()
            rs_ga_solutions.append(solution_rs_ga)
        # Rastgele çözümleri sırala
        rs_ga_solutions.sort(key=lambda x: x.fitness, reverse=False)
        # En iyi 10 tanesini başlangıç çözümü için al.
        rs_ga_solutions_10 = rs_ga_solutions[:10]
        # initialize population

        pop = Population(size=pop_size)
        for rs_ga in rs_ga_solutions_10:
            pop.chromosomes.append(rs_ga)
        self.pop_list.append(pop)
        self.pop_list[0].best = round(self.pop_list[0].chromosomes[0].fitness, 3)
        _total = 0
        for ind in self.pop_list[0].chromosomes:
            _total += ind.fitness
        self.pop_list[0].avg = round(_total / self.pop_list[0].size, 2)
        self.pop_list[0].worst = round(self.pop_list[0].chromosomes[-1].fitness, 2)
        print("\nInitial Population \n")
        print("Generation: {0:<3} Best: {1:4} Avg: {2:4} Worst:{3:4}".format(0, round(self.pop_list[0].best, 2),
                                                                             round(self.pop_list[0].avg, 2),
                                                                             round(self.pop_list[0].worst, 2)))
        print("-" * 100)
        print("{0:<7} {1:<7} {2:<15} {3:<7}".format("ddrule", "dsprule", "routes", "fitness"))
        for kromozom in self.pop_list[0].chromosomes:
            print("{0:<7} {1:<7} {2:<15} {3:<7}".format(kromozom.ddrule, kromozom.dsprule, str(kromozom.routes),
                                                        round(kromozom.fitness, 2)))
        print("-" * 100)

    def run(self, iter_size):

        for i in range(1, iter_size):
            new_pop = Population(pop_size)
            new_pop.chromosomes.extend(self.pop_list[i - 1].chromosomes)
            new_pop_copy = [deepcopy(chr.genes)
                            for chr in new_pop.chromosomes]  # copy genes
            new_pop_copy = np.array(new_pop_copy)
            # np.random.seed(100+i)
            # crossover
            crossover_pop_idx = np.random.choice(
                range(len(new_pop_copy)), size=GeneticAlgorithm.num_cross, replace=False,
                p=GeneticAlgorithm.prob_chr)  # crossover için genes seç
            crossover_pop_idx.sort()
            # print("iterasyon {}, seçilen bireyler: {}".format(i, crossover_pop_idx))
            crossover_pop = new_pop_copy[crossover_pop_idx, :]
            crossover_pop = np.array(crossover_pop)
            crossover_chromosomes = []
            while len(crossover_chromosomes) < GeneticAlgorithm.num_cross:
                parents_idx = np.random.choice(
                    range(len(crossover_pop)), size=2, replace=False)
                parent1, parent2 = crossover_pop[parents_idx, :]
                parent1 = list(parent1)
                parent2 = list(parent2)
                rs = np.random.random()
                # print("random for crossover: ", rs)
                if rs <= GeneticAlgorithm.pc:
                    if GeneticAlgorithm.num_points_crossover > 5:
                        num_points_crossover = 5
                    points = np.random.choice(
                        range(GeneticAlgorithm.gene_size), size=GeneticAlgorithm.num_points_crossover, replace=False,
                        p=GeneticAlgorithm.prob_genes)
                    points.sort()
                    if GeneticAlgorithm.num_points_crossover == 1:
                        child1 = parent1[0:points[0]] + parent2[points[0]:]
                        child2 = parent2[0:points[0]] + parent1[points[0]:]
                    elif GeneticAlgorithm.num_points_crossover == 2:
                        child1 = parent1[0:points[0]] + \
                                 parent2[points[0]:points[1]] + parent1[points[1]:]
                        child2 = parent2[0:points[0]] + \
                                 parent1[points[0]:points[1]] + parent2[points[1]:]
                    elif GeneticAlgorithm.num_points_crossover == 3:
                        child1 = parent1[0:points[0]] + parent2[points[0]:points[1]] + \
                                 parent1[points[1]:points[2]] + parent2[points[2]:]
                        child2 = parent2[0:points[0]] + parent1[points[0]:points[1]] + \
                                 parent2[points[1]:points[2]] + parent1[points[2]:]
                    elif GeneticAlgorithm.num_points_crossover == 4:
                        child1 = parent1[0:points[0]] + parent2[points[0]:points[1]] + \
                                 parent1[points[1]:points[2]] + \
                                 parent2[points[2]:points[3]] + parent1[points[3]:]
                        child2 = parent2[0:points[0]] + parent1[points[0]:points[1]] + \
                                 parent2[points[1]:points[2]] + \
                                 parent1[points[2]:points[3]] + parent2[points[3]:]
                    else:
                        child1 = parent1[0:points[0]] + parent2[points[0]:points[1]] + \
                                 parent1[points[1]:points[2]] + parent2[points[2]:points[3]] + \
                                 parent1[points[3]:points[4]] + parent2[points[4]:]
                        child2 = parent2[0:points[0]] + parent1[points[0]:points[1]] + \
                                 parent2[points[1]:points[2]] + parent1[points[2]:points[3]] + \
                                 parent2[points[3]:points[4]] + parent1[points[4]:]
                    chr1 = Chromosome()
                    chr2 = Chromosome()
                    chr1.change_genes(child1)
                    chr2.change_genes(child2)
                    # print("child1", child1, "popülasyona ekleniyor")
                    crossover_chromosomes.append(chr1)
                    # print("child2", child2, "popülasyona ekleniyor")
                    crossover_chromosomes.append(chr2)
                else:
                    chr1 = Chromosome()
                    chr2 = Chromosome()
                    chr1.change_genes(parent1)
                    chr2.change_genes(parent2)
                    crossover_chromosomes.append(chr1)
                    crossover_chromosomes.append(chr2)
            new_pop.chromosomes.extend(crossover_chromosomes)

            # mutation
            new_pop_copy_mut = [deepcopy(chr.genes) for chr in new_pop.chromosomes]
            new_pop_copy_mut = np.array(new_pop_copy_mut)
            mutation_pop_idx = np.random.choice(
                range(len(new_pop_copy_mut)), size=GeneticAlgorithm.num_mut)
            mutation_pop_idx.sort()
            mutation_pop = new_pop_copy_mut[mutation_pop_idx, :]
            mutation_pop = np.array(mutation_pop)
            mutation_chromosomes = []

            while len(mutation_chromosomes) < GeneticAlgorithm.num_mut:
                child_idx = np.random.choice(
                    range(len(mutation_pop)), size=1, replace=False)
                child = mutation_pop[child_idx, :]
                rs2 = np.random.random()
                # print("random for mutation: ", rs2)
                if rs2 <= GeneticAlgorithm.pm:
                    mut_points = np.random.choice(
                        range(GeneticAlgorithm.gene_size), size=GeneticAlgorithm.num_points_mutation, replace=False,
                        p=GeneticAlgorithm.prob_genes)
                    mut_points.sort()
                    for mut_point in mut_points:
                        # print(mut_point, "mut point")
                        if mut_point == 0:  # ddrule
                            child[0][0] = np.random.randint(ddrule_size)
                        elif mut_point == 1:  # dsprule
                            child[0][1] = np.random.randint(dsprule_size)
                        else:  # routes
                            child[0][mut_point] = np.random.randint(ps.route_size)
                    chr3 = Chromosome()
                    chr3.change_genes(child[0])
                    mutation_chromosomes.append(chr3)
                else:
                    chr3 = Chromosome()
                    chr3.change_genes(child[0])
                    mutation_chromosomes.append(chr3)
            new_pop.chromosomes.extend(mutation_chromosomes)
            for ind in new_pop.chromosomes:
                ind.generate_data()
                ind.due_date_assignment()
                ind.run_simulation()
                ind.calculate_fitness()
            seen_chrs = list()
            new_chromosome_list = []
            for obj in new_pop.chromosomes:
                if obj.genes not in seen_chrs:
                    new_chromosome_list.append(obj)
                    seen_chrs.append(obj.genes)
            new_pop.chromosomes = new_chromosome_list  # remove chromosome with same genes
            while len(new_pop.chromosomes) < pop_size:
                chr4 = Chromosome()
                random_genes = []
                random_genes.append(np.random.randint(ddrule_size))
                random_genes.append(np.random.randint(dsprule_size))
                random_genes.append(np.random.randint(ps.route_size, size=ps.job_size))
                chr4.change_genes(random_genes)
                new_pop.chromosomes.append(chr4)
            new_pop.sort_pop()  # sort pop
            new_pop.chromosomes = new_pop.chromosomes[:pop_size]  # reap pop
            self.pop_list.append(new_pop)
            self.pop_list[i].best = round(self.pop_list[i].chromosomes[0].fitness, 3)
            _total = 0
            for ind in self.pop_list[i].chromosomes:
                _total += ind.fitness
            self.pop_list[i].avg = round(_total / self.pop_list[i].size, 2)
            self.pop_list[i].worst = round(self.pop_list[i].chromosomes[-1].fitness, 2)
            print("Generation: {0:<3} Best: {1:4} Avg: {2:4} Worst:{3:4}".format(i, round(self.pop_list[i].best, 2),
                                                                                 round(self.pop_list[i].avg, 2),
                                                                                 round(self.pop_list[i].worst, 2)))
            print("-" * 100)
            print("{0:<7} {1:<7} {2:<15} {3:<7}".format("ddrule", "dsprule", "routes", "fitness"))
            for kromozom in self.pop_list[i].chromosomes:
                print("{0:<7} {1:<7} {2:<15} {3:<7}".format(kromozom.ddrule, kromozom.dsprule,
                                                            str(np.array(kromozom.routes)),
                                                            round(kromozom.fitness, 2)))
            print("-" * 100)


iter_size = int(input("Enter iteration size: "))
algoritma = str(input("Input the algorithm to be applied (PSO,GA,both)"))
if algoritma == "PSO":
    bas_pop = Swarm(pop_size, 0)
    bas_pop.initialize_swarm()
    bas_pop.run(iter_size)
    plt.plot(all_gbests,"-b",label='PSO')
    plt.xlabel("iteration")
    plt.ylabel("performance")
    plt.title("PSO Performance")
    plt.legend(loc='best')
    plt.show()
elif algoritma == "GA":
    ga = GeneticAlgorithm()
    ga.initialize_population(10)
    ga.run(iter_size)
    ga.finish_time=time.time()
    ga_bests = [x.best for x in ga.pop_list]
    plt.plot(ga_bests, "-r", label='GA')
    plt.xlabel("iteration")
    plt.ylabel("performance")
    plt.title("GA Performance")
    plt.legend(loc='best')
    plt.show()
elif algoritma=="both":
    #PSO
    pso_start_time=time.time()
    bas_pop = Swarm(pop_size, 0)
    bas_pop.run(iter_size)
    pso_finish_time = time.time()
    # GA
    ga = GeneticAlgorithm()
    ga.start_time = time.time()
    ga.initialize_population(10)
    ga.run(iter_size)
    ga.finish_time = time.time()
    ga_bests = [x.best for x in ga.pop_list]
    # plot
    plt.plot(all_gbests, "-b", label='PSO')
    plt.plot(ga_bests, "-r", label='GA')
    plt.xlabel("iteration")
    plt.ylabel("performance")
    plt.title("Comparision GA and PSO Performance")
    plt.legend(loc='best')
    plt.show()
    bas_pop.initialize_swarm()

