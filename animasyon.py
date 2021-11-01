# An animation for dippsdda..

import salabim as sim

machine_numbers = [[0, 1, 1, 1], [1, 0, 1, 1],
                   [0, 0, 1, 1], [1, 0, 1, 0]]

operation_durations = [[10, 9, 23, 17], [1, 12, 45, 7],
                       [23, 17, 5, 13], [10, 8, 7, 10]]

arrival_times = [16, 20, 22, 45]

number_of_jobs = len(machine_numbers)
number_of_operations_per_job = len(machine_numbers[0])
number_of_machines = 2

# strategy = 'sptsum'
strategy = 'spt'


# strategy = 'fifo'
# strategy = 'lifo'


class Job(sim.Component):
    def setup(self, i):
        self.operations = sim.Queue(name=self.name() + '.operations')
        for j in range(number_of_operations_per_job):
            machine = env.machines[machine_numbers[i][j]]
            duration = operation_durations[i][j]
            Operation(
                name='operation.' +
                     str(i) + '.' + str(j) + ' m=' +
                     machine.name() + ' d=' + str(duration),
                job=self, machine=machine, duration=duration).enter(self.operations)

        self.arrival_time = arrival_times[i]
        self.depart_time = 0

    def process(self):
        yield self.hold(self.arrival_time)
        self.enter(env.jobs)
        # sim.AnimateText(self.operations.name(), x=500, y=lambda: (env.jobs.index(self) + 1) * 150 + 25, parent=self)
        sim.AnimateQueue(
            queue=self.operations, x=500, y=lambda: (env.jobs.index(self) + 1) * 150, direction='s', parent=self)
        while self.operations:
            next_operation = self.operations.head()
            next_operation.enter(next_operation.machine.queue)
            if next_operation.machine.ispassive():
                next_operation.machine.activate()
            yield self.passivate()
        self.leave()
        self.depart_time = env.now()


class Operation(sim.Component):
    def setup(self, job, machine, duration):
        self.job = job
        self.machine = machine
        self.duration = duration
        self.color = 'red'

    def animation_objects(self):
        ao = sim.AnimateRectangle(
            (0, 0, 250, 20), fillcolor=lambda: self.color, text=self.name(), text_anchor='w')
        return 110, 25, ao


class Machine(sim.Component):
    def setup(self):
        self.queue = sim.Queue(self.name() + '.queue')
        # sim.AnimateText(self.queue.name(), x=100, y=(self.sequence_number() + 1) * 150 + 25)
        sim.AnimateQueue(queue=self.queue, x=100, y=(self.sequence_number() + 1) * 150, direction='s')

    def process(self):
        while True:
            if self.queue:  # kuyrukta iş varsa
                if strategy == 'fifo':
                    sel_operation = self.queue.head()
                elif strategy == 'lifo':
                    sel_operation = self.queue.tail()
                elif strategy == 'spt':
                    sel_operation = self.queue.head()
                    for operation in self.queue:
                        if operation.duration < sel_operation.duration:
                            sel_operation = operation
                elif strategy == 'sptsum':
                    sel_sum = sim.inf
                    for operation in self.queue:
                        this_sum = sum(
                            op.duration for op in operation.job.operations)
                        if this_sum < sel_sum:
                            sel_sum = this_sum
                            sel_operation = operation
                sel_operation.color = 'green'
                yield self.hold(sel_operation.duration)
                sel_operation.leave()  # iş ve makine kuyruğunu birlikte terket
                sel_operation.job.activate()
            else:
                yield self.passivate()


env = sim.Environment(trace=False)
env.machines = [Machine() for _ in range(number_of_machines)]
for i in range(number_of_jobs):
    Job(i=i)
env.jobs = sim.Queue(name='jobs')
env.animation_parameters(modelname='job shop 4.3', speed=2)
env.run()
