import M1_Data
import M2_DGPRN
import M3_Train
import M4_Plot

if __name__ == '__main__':
    group = 10
    num = 3
    domain = '{0}'.format('0,16,24', '15,31,24', '30,46,32', '45,61,40', '60,84,24', '-16,0,24', '-31,-15,24', '-46,-30,32', '-61,-45,40', '-84,-60,24')
    epoch = 1
    batch_size = 40 #38
    lr = 1e-4
    load = '{1}'.format('False', 'True')
    device = '{3}'.format('cpu', 'cuda:0', 'cuda:1', 'cuda')
    # M3_Train.train(M1_Data, M2_DGPRN, group, num, domain, epoch, batch_size, lr, load, device)
    M3_Train.concat(M1_Data, M2_DGPRN, device)
    # M4_Plot.plot(group, num, domain, idx = 60)
    # M4_Plot.domain()
    # M4_Plot.flow()
    # M4_Plot.edge(domain)
    # M4_Plot.conclusion()