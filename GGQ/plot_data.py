import numpy as np
import matplotlib.pyplot as plt
import json

with open("loss_tracker.txt") as infile:
    loss_json=infile.readlines()[0]
    loss_list=json.loads(loss_json)
    plt.plot(range(len(loss_list)),loss_list)
    plt.savefig("./graphs/loss.png")
    plt.clf()

with open("Q_tracker.txt") as infile:
    Q_json=infile.readlines()[0]
    Q_dict=json.loads(Q_json)
    for state in Q_dict:
        plt.plot(range(len(Q_dict[state])),Q_dict[state])
        plt.savefig("./graphs/"+state+".png")
        plt.clf()

with open("reward_breakdown.txt") as infile:
    reward_json=infile.readlines()[0]
    reward_dict=json.loads(reward_json)
    objects = ("0","1","-10","-2")
    y_pos = np.arange(len(objects))
    plt.bar(y_pos,[32116,45037,1313,2366])
    plt.xticks(y_pos,objects)
    plt.savefig("./graphs/reward_breakdown.png")
    plt.clf()
