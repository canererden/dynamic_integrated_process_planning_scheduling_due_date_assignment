import numpy as np

# inputs for the shop floor
np.random.seed(12345)
# generate arrival times
number_of_jobs = [4,25]
weight_list0 = [0.33, 0.66, 1]
weight_list1 = [0.25, 0.50, 0.75, 1]
weight_list2 = [0.2, 0.4, 0.6, 0.8, 1]
weight_list3 = [0.16, 0.32, 0.48, 0.64, 0.8, 1]

class ProblemSet:
    DATA = [[4, 25, 50],  # 0. Is Sayilari
            [4, 10, 10],  # 1. Operasyon Sayilari
            [2, 5, 10],  # 2. Makine Sayilari
            [2, 5, 5],  # 3. Rota Sayilari
            [10, 20, 30]]  # iter_size

    def __init__(self, ps_id):
        self.ps_id = ps_id
        self.job_size = ProblemSet.DATA[0][ps_id]
        self.operation_size = ProblemSet.DATA[1][ps_id]
        self.machine_size = ProblemSet.DATA[2][ps_id]
        self.route_size = ProblemSet.DATA[3][ps_id]
i=2
ps = ProblemSet(i)
machine_numbers = np.random.randint(0, ps.machine_size, size=(
    ps.job_size, ps.route_size, ps.operation_size))
operation_durations = abs(np.random.normal(
    6, 12, (ps.job_size, ps.route_size, ps.operation_size))).astype(int)+1
with open('inputs\\machine_numbers_'+str(i)+'.txt', 'w') as outfile:
    outfile.write('# Job size: {}, Route size {}, Operation size {} \n'.format(
        ps.job_size, ps.route_size, ps.operation_size))
    outfile.write('# Machine size: {}\n'.format(ps.machine_size))
    for data_slice in machine_numbers:
        np.savetxt(outfile, X=data_slice, fmt='%i')
        outfile.write('# New job\n')
with open('inputs\\operation_durations_'+str(i)+'.txt', 'w') as outfile:
    outfile.write('# Job size: {}, Route size {}, Operation size {} \n'.format(
        ps.job_size, ps.route_size, ps.operation_size))
    outfile.write('# Machine size: {}\n'.format(ps.machine_size))
    for data_slice in operation_durations:
        np.savetxt(outfile, X=data_slice, fmt='%i')
        outfile.write('# New job\n')
arrival_times = np.cumsum(
    list(map(int, np.random.exponential(20, number_of_jobs[i]))))
np.savetxt('inputs\\arrivals_'+str(i)+'.txt', X=arrival_times, fmt='%i')
if 0 <= i < 3:
    w_l = weight_list0
elif 3 <= i < 5:
    w_l = weight_list1
elif 5 <= i < 7:
    w_l = weight_list2
elif 7 <= i < 9:
    w_l = weight_list3
weights = np.random.choice(w_l, size=number_of_jobs[i], replace=True)
np.savetxt('inputs\\weights_'+str(i)+'.txt', X=weights, fmt='%1.3f')