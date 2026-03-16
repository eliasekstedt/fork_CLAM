
def file_it(file_name, message, to_terminal=False):
    if to_terminal:
        print(message)
    with open(file_name, 'a') as file:
        file.write(f'{message}\n')

def plot_performance(protocol, dpath_run):
    epochs = range(1, len(protocol.valcost) + 1)
    import matplotlib.pyplot as plt
    for train_item, val_item, name in zip(
        [protocol.traincost, protocol.trainperformance, protocol.train_tp, protocol.train_pp],
        [protocol.valcost, protocol.valperformance, protocol.val_tp, protocol.val_pp],
        ['cost', 'accuracy', 'true_positives', 'predicted_positives'],
    ):
        plt.plot(epochs, train_item, 'tab:blue', label='train_0')
        plt.plot(epochs, val_item, 'tab:red', label='val_0')
        plt.ylim([0, max(train_item + val_item + [1])]) # crude
        plt.legend()
        plt.ylabel(name)
        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(dpath_run / f'{name}.png')
        plt.figure()
        plt.close('all')