import pickle
import matplotlib.pyplot as plt

def plot_history(history, savefilename):
    plt.figure(figsize=(12,6), dpi=300.0)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc="upper left")

    plt.tight_layout()
    plt.grid()
    plt.savefig(savefilename)


with open(f"./history_inceptionv3_GRU_Nadam_optimizer2epochs.pkl", "rb") as infile:
    history = pickle.load(infile)


print(history)
# plot_history(history)