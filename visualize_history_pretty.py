import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import sys

def nice_plot_history(history):
    fig, ax = plt.subplots()

    loss = history.history["loss"]
    val_loss = history.history["val_loss"] #[0.5286757326126099, 0.2425833038489024, 0.26228026111920677, 0.2598864257335663, 0.2978562418619792, 0.508716340859731, 0.5084260400136312, 0.4114651809136073, 0.5096775682767233, 0.5096775682767233, 0.5096775682767233, 0.5096775682767233, 0.5095644116401672, 0.5096777081489563, 0.5096776700019836, 0.5096776700019836, 0.5096777208646138, 0.5096673321723938, 0.5096775301297506, 0.509676551024119, 0.5096763094266256, 0.5096761695543925, 0.5096760042508444, 0.5096759406725565, 0.5096746818224589, 0.5096738680203756, 0.5096733212471009, 0.509672888914744, 0.5096722912788391, 0.5096718335151672]
    accuracy = history.history["accuracy"] #[0.49428571445601327, 0.5085714275496347, 0.5071428557804653, 0.5371428557804653, 0.5228571430274418, 0.45571428520338875, 0.49999999863760813, 0.5228571430274418, 0.4742857132639204, 0.5057142857142857, 0.5057142840112958, 0.5057142858845847, 0.5057142843518938, 0.49285714183534896, 0.5042857142857143, 0.5057142853736878, 0.5057142843518938, 0.5057142843518938, 0.5114285714285715, 0.5057142850330898, 0.5057142846924918, 0.5057142843518938, 0.5057142845221928, 0.5057142843518938, 0.5057142858845847, 0.5057142843518938, 0.5057142857142857, 0.5057142846924918, 0.5057142880984715, 0.5057142845221928]
    val_accuracy = history.history["val_accuracy"] #[0.6033333365122477, 0.6499999992052714, 0.6266666666666667, 0.6333333333333333, 0.5833333349227905, 0.49000000158945717, 0.49000000158945717, 0.6166666674613953, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717, 0.49000000158945717]

    data=[loss, val_loss, accuracy, val_accuracy]
    columns=['loss', 'val_loss', 'accuracy', 'val_accuracy']

    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = columns

    print(df)

    accuracies = []
    losses = []

    def plot_item(name, color, max_wanted = True):
        line_item = sns.lineplot(y=name, x=df.index, data=df, label=name)
        if max_wanted:
            max_y = df[name].max()
            max_idx = df[name].idxmax()
        else:
            max_y = df[name].min()
            max_idx = df[name].idxmin()

        text_item = plt.text(max_idx + 0.2, max_y + 0.05, str(round(max_y,2)), horizontalalignment='left', size='medium', color=color, weight='semibold')
        return [line_item, text_item]

    accuracies += plot_item("accuracy", "blue")
    accuracies += plot_item("val_accuracy", "orange")

    losses += plot_item("loss", "green", max_wanted=False)
    losses += plot_item("val_loss", "brown", max_wanted=False)

    max_val = max(df["loss"].max(), df["val_loss"].max())

    plt.ylim(0, 1)
    plt.ylabel("Accuracy")

    plt.legend(loc='lower right') #best

    def press(event):
        sys.stdout.flush()
        if event.key == '+':
            # zoom to 0-1 accuracy
            plt.ylim(0, 1)
        elif event.key == '-':
            plt.ylim(0, max_val)
        elif event.key == 'b':
            plt.legend(loc='best')
        elif event.key == 'a':
            plt.legend(loc='lower right')
        else:
            print('press', event.key)

        fig.canvas.draw()


    fig.canvas.mpl_connect('key_press_event', press)

    plt.show()
