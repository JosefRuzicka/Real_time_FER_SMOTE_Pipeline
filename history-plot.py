import matplotlib.pyplot as plt

import json

colors = {"vgg" : ["#08306a", "#63a8d2"],
          "resNet" : ["#58b668", "#97d594"],
          "mobNet" : ["#3d017c", "#9894c7"]}
line = {"vgg" : "-",
          "resNet" : "-.",
          "mobNet" : "--"}

metrics = {"acc": "Accuracy", "loss": "Loss"}
pos = {"acc": "upper", "loss": "lower"}
models = ["vgg", "resNet", "mobNet"]
 
f = open('results.json')
res = json.load(f)

factor = 2

for metric in metrics.keys():
  fig, ax = plt.subplots()
  fig.set_size_inches(16, 8)
  plots = []

  for model in models:

    train = res[model]["train"][metric] + res[model]["train_ft"][metric]

    x = [i for i in range(len(train))]
    l, = ax.plot(x, train, color=colors[model][0], linewidth = 2 * factor, linestyle=line[model])
    plots.append(l)

    test = res[model]["test"][metric] + res[model]["test_ft"][metric]

    x = [i for i in range(len(test))]
    l, = ax.plot(x, test, color=colors[model][1], linewidth = 2 * factor, linestyle=line[model])
    plots.append(l)

    plt.axvline(x=len(res[model]["test"][metric]), color = colors[model][0], linewidth = 1.5 * factor, alpha = 0.4, linestyle = ":")

  ax.text(len(res["vgg"]["train"][metric]) + 0.5, res["vgg"]["train_ft"][metric][0] - 0.03, 'fine-tuning', fontsize = 9 * factor, color=colors["vgg"][0])
  ax.text(len(res["resNet"]["train"][metric]) + 0.5, res["resNet"]["train_ft"][metric][0], 'fine-tuning', fontsize = 9 * factor, color=colors["resNet"][0])
  ax.text(len(res["mobNet"]["train"][metric]) + 0.5, res["mobNet"]["train_ft"][metric][0], 'fine-tuning', fontsize = 9 * factor, color=colors["mobNet"][0])

  ax.legend(plots, ('VGG19 - train','VGG19 - test', 'ResNet50 - train','ResNet50 - test', 'MobileNetV2 - train','MobileNetV2 - test'), loc=f'{pos[metric]} left', shadow=True, ncol=3, fontsize = 10 * factor)
  ax.set_xlabel('Epoch', fontsize = 10 * factor)
  ax.set_ylabel(metrics[metric], fontsize = 10 * factor)
  ax.tick_params(labelsize = 10 * factor)
  ax.grid(True, axis='y')

  plt.savefig(f'{metric}5_plot.svg', format='svg', bbox_inches='tight')

  plt.show()
  ax.clear()