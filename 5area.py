import pandas as pd
import os
import re
# from tqdm import tqdm
from scipy.stats import norm


BuildInPrint = print
FOLDER = 'C:/FILE/1Ds/OneDrive - Saint Louis University/SCM/SPRING 2019/OPM 6460/simulation_2'
I = 1.1 ** (1 / 365) - 1
H = 100 / 365
TRUCKLOAD = 200
DAYZERO = 669
UPPER_Q = 1000
COST_PER_DRUM = 1000
COST_PER_BATCH = 1500
HORIZON = 32
CAPACITY = {'C': 70, 'S': 35, 'T': 100}
MAP = {'C': 'Calopeia', 'S': 'Sorange', 'T': 'Tyran', 'E': 'Entworpe', 'F': 'Fardo'}
FACTORIES = 'CST'
WAREHOUSES = 'CSTEF'
K = ((15000, 150), (20000, 200), (45000, 400))
SHIPPING_TIME = ((7, 1), (7, 1), (14, 2))
SHIPPING_METHOD = ('Truck', 'Mail')
SERVICE_LEVEL = 0.9
Z = norm.ppf(SERVICE_LEVEL)
LOGFILE = 'Output.txt'


def print(*arg, **kw):
    BuildInPrint(*arg, **kw)
    with open(LOGFILE, 'a') as f:
        kw.update(dict(file=f))
        BuildInPrint(*arg, **kw)


def int_(something):
    try:
        return int(something)
    except ValueError:
        assert isinstance(something, str)
        somestr = ''
        for char in something:
            if char.isdigit():
                somestr += char
        return int(somestr)


def readTheXl(filename):
    instr = ''
    with open(os.path.join(FOLDER, filename), 'r') as f:
        instr = f.read()
    pattern = re.compile(r'>(\d*,?\d+)<')
    dflist = [[header, ] for header in 'day,Calopeia,Sorange,Tyran,Entworpe,Fardo'.split(',')]
    numberstr = pattern.findall(instr)
    for i in range(len(numberstr)):
        dflist[i % 6].append(numberstr[i])
    dfdict = dict()
    for numberlist in dflist:
        dfdict[numberlist[0]] = numberlist[1:]
    return pd.DataFrame(dfdict)


def ols1(y, x):
    # print(x)
    # print(y)
    n = len(x)
    assert len(y) == n
    xb = sum(x) / n
    yb = sum(y) / n
    sp = sum([x[i] * y[i] for i in range(n)])
    ss = sum([x[i] ** 2 for i in range(n)])
    b1 = (sp - n * xb * yb) / (ss - n * xb ** 2)
    b0 = yb - b1 * xb

    yn = b0 + b1 * x[-1]
    se = [(b0 + b1 * x[i] - y[i]) ** 2 for i in range(n)]
    stdev = (sum(se) / (n - 2)) ** 0.5

    return b0, b1, yn, stdev


def main():
    with open(os.path.join(FOLDER, LOGFILE), 'w', encoding='utf-8') as f:
        f.truncate()

    # rawdata = pd.read_csv(os.path.join(FOLDER, 'Demand for each destination region.csv'))
    rawdata = readTheXl('Demand for each destination region..xls')

    for name in rawdata:
        rawdata[name] = rawdata[name].map(int_)
    # print(rawdata.head())
    # print(rawdata.tail())
    lastday = rawdata['day'].values[-1]
    print(f'Day {lastday}')

    mean = dict()
    stdev = dict()
    slope = dict()

    for w in 'TEF':
        warehouse = MAP[w]
        demands = rawdata[warehouse][DAYZERO:]
        n = len(demands)
        mean[w] = sum(demands) / n
        stdev[w] = demands.var() ** 0.5
        slope[w] = 0.0

    w = 'S'
    warehouse = MAP[w]
    demands = rawdata[warehouse][DAYZERO:].values
    days = rawdata['day'][DAYZERO:].values
    _, slope[w], mean[w], stdev[w] = ols1(demands, days)

    w = 'C'
    warehouse = MAP[w]
    rawdata['date'] = rawdata['day'] % 365
    if rawdata['date'].values[-1] <= 183:
        halfyear = rawdata[rawdata['date'] <= 183]
    else:
        halfyear = rawdata[rawdata['date'] > 183]
    demands = halfyear[warehouse].values
    dates = halfyear['date'].values
    _, slope[w], mean[w], stdev[w] = ols1(demands, dates)

    # print(mean)
    # print(stdev)
    # print(slope)
    statsdict = {'Warehouse': ['Mean', 'Stdev', 'Slope']}
    for w in WAREHOUSES:
        statsdict[w] = [mean[w], stdev[w], slope[w]]
    stats = pd.DataFrame(statsdict)

    dfdict = dict()
    for w in WAREHOUSES:
        dfdict[w] = dict()
        warehouse = MAP[w]
        for f in FACTORIES:
            dfdict[w][f] = list()
            if w == f:
                    zone = 0
            elif 'F' not in f + w:
                zone = 1
            else:
                zone = 2
            for ship in (0, 1):
                cost = pd.DataFrame({'Q': list(range(1, UPPER_Q + 1))})
                if ship == 0:
                    cost['Shipping'] = ((cost['Q'] - 1) // 200 + 1) * K[zone][ship] / cost['Q']
                else:
                    cost['Shipping'] = K[zone][ship]
                cost['Holding'] = 100
                cost['Material'] = COST_PER_DRUM + COST_PER_BATCH / cost['Q']
                cost['Interest'] = (1 + I) ** (cost['Q'] / min((mean[w], CAPACITY[f])))
                cost['Total'] = (cost['Shipping'] + cost['Holding'] + cost['Material']) * cost['Interest']
                dfdict[w][f].append(cost)

    eoq = dict()
    for w in WAREHOUSES:
        eoqdict = {f'to {w}': ['Shipping_method', 'EOQ', 'Cost', 'Shipping_time', 'Leadtime', '+', 'ROP']}
        for f in FACTORIES:
            pair = list()
            for ship in (0, 1):
                cost = dfdict[w][f][ship]
                i = cost['Total'].idxmin()
                pair.append((cost['Q'].values[i], cost['Total'].values[i]))
            if pair[0][1] < pair[1][1]:
                j = 0
            else:
                j = 1
            quantity, cost = pair[j]
            eoqdict[f] = [SHIPPING_METHOD[j], quantity, cost]
            if w == f:
                    zone = 0
            elif 'F' not in f + w:
                zone = 1
            else:
                zone = 2
            shipping_time = SHIPPING_TIME[zone][j]
            leadtime = quantity / CAPACITY[f] + shipping_time
            addition = slope[w] * HORIZON / 2
            rop = mean[w] * leadtime + Z * stdev[w] * leadtime ** 0.5 + addition
            eoqdict[f] += [shipping_time, leadtime, addition, rop]

        eoq[w] = pd.DataFrame(eoqdict)

    d = sum(mean.values())
    cap = sum(CAPACITY.values())
    print(f'\n{stats}\n\nTotal Demand: {d}  /  Total Cap: {cap}  ==  {d / cap}')
    for w in eoq:
        print(f'\n{eoq[w]}')


if __name__ == '__main__':
    main()
