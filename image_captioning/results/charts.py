from matplotlib import pyplot as plt
import numpy as np
data = np.load("stats(1).npy")
#data2 = np.load("stats.npy")

test = "More Epochs"
test2 = "GRU"
def save_fig(data,test):
    
    plt.plot(data[1:,2],label='Loss')
    #plt.plot(data[:,0]+1*data[:,1],data[:,3])
    
    
    plt.title("Hyperparameter Tuning: " + test)
    plt.xlabel("Iteration Number (*10)")
    plt.ylabel("Loss/Perplexity")
    
    plt.plot(data[20:,3],label='Perplexity')
    #plt.plot(data[:,0]+1*data[:,1],data[:,3])
    plt.legend()
    plt.savefig(test+".png")
    plt.close()
save_fig(data,test)
#save_fig(data2,test2)