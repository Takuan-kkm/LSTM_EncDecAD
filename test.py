import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import ArtistAnimation
import pickle
from PIL import Image

log = [
    {
        "main/loss": 309.08306884765625,
        "validation/main/loss": 55.0660285949707,
        "epoch": 1,
        "iteration": 23,
        "elapsed_time": 1526.5734852780001
    },
    {
        "main/loss": 43.35818862915039,
        "validation/main/loss": 48.24916458129883,
        "epoch": 2,
        "iteration": 45,
        "elapsed_time": 2965.421696723
    },
    {
        "main/loss": 39.60798645019531,
        "validation/main/loss": 46.20176696777344,
        "epoch": 3,
        "iteration": 68,
        "elapsed_time": 4464.678672573
    },
    {
        "main/loss": 37.974517822265625,
        "validation/main/loss": 40.81900405883789,
        "epoch": 4,
        "iteration": 90,
        "elapsed_time": 5922.0092798020005
    },
    {
        "main/loss": 36.881107330322266,
        "validation/main/loss": 43.3662223815918,
        "epoch": 5,
        "iteration": 112,
        "elapsed_time": 7378.245238298
    },
    {
        "main/loss": 36.817466735839844,
        "validation/main/loss": 43.72343826293945,
        "epoch": 6,
        "iteration": 135,
        "elapsed_time": 8897.552051657
    },
    {
        "main/loss": 36.85164260864258,
        "validation/main/loss": 40.62918472290039,
        "epoch": 7,
        "iteration": 157,
        "elapsed_time": 10354.891792588
    },
    {
        "main/loss": 36.336082458496094,
        "validation/main/loss": 39.405914306640625,
        "epoch": 8,
        "iteration": 179,
        "elapsed_time": 11809.059719523
    },
    {
        "main/loss": 35.61872482299805,
        "validation/main/loss": 40.56131362915039,
        "epoch": 9,
        "iteration": 202,
        "elapsed_time": 13309.101205149
    },
    {
        "main/loss": 35.05540466308594,
        "validation/main/loss": 39.85761642456055,
        "epoch": 10,
        "iteration": 224,
        "elapsed_time": 14748.947492231
    },
    {
        "main/loss": 34.179683685302734,
        "validation/main/loss": 38.513031005859375,
        "epoch": 11,
        "iteration": 247,
        "elapsed_time": 16247.310247628
    },
    {
        "main/loss": 33.887882232666016,
        "validation/main/loss": 41.7261848449707,
        "epoch": 12,
        "iteration": 269,
        "elapsed_time": 17686.804200754
    },
    {
        "main/loss": 33.456787109375,
        "validation/main/loss": 40.650760650634766,
        "epoch": 13,
        "iteration": 291,
        "elapsed_time": 19125.068074423998
    },
    {
        "main/loss": 33.10764694213867,
        "validation/main/loss": 39.8390007019043,
        "epoch": 14,
        "iteration": 314,
        "elapsed_time": 20630.505792706997
    },
    {
        "main/loss": 32.62324523925781,
        "validation/main/loss": 36.76996612548828,
        "epoch": 15,
        "iteration": 336,
        "elapsed_time": 22073.483083422998
    },
    {
        "main/loss": 32.76946258544922,
        "validation/main/loss": 39.56733703613281,
        "epoch": 16,
        "iteration": 358,
        "elapsed_time": 23512.56142413
    },
    {
        "main/loss": 32.02280044555664,
        "validation/main/loss": 38.68796157836914,
        "epoch": 17,
        "iteration": 381,
        "elapsed_time": 25029.949634839
    },
    {
        "main/loss": 32.046016693115234,
        "validation/main/loss": 40.07285690307617,
        "epoch": 18,
        "iteration": 403,
        "elapsed_time": 26467.711868802
    },
    {
        "main/loss": 31.09001350402832,
        "validation/main/loss": 36.12235641479492,
        "epoch": 19,
        "iteration": 426,
        "elapsed_time": 28001.753642402
    },
    {
        "main/loss": 31.077430725097656,
        "validation/main/loss": 36.99208068847656,
        "epoch": 20,
        "iteration": 448,
        "elapsed_time": 29438.026866209
    },
    {
        "main/loss": 30.753093719482422,
        "validation/main/loss": 38.15940856933594,
        "epoch": 21,
        "iteration": 470,
        "elapsed_time": 30874.338721849
    },
    {
        "main/loss": 30.307161331176758,
        "validation/main/loss": 36.603607177734375,
        "epoch": 22,
        "iteration": 493,
        "elapsed_time": 32370.205786005998
    },
    {
        "main/loss": 30.59330940246582,
        "validation/main/loss": 37.105594635009766,
        "epoch": 23,
        "iteration": 515,
        "elapsed_time": 33813.032963492995
    },
    {
        "main/loss": 30.27446937561035,
        "validation/main/loss": 35.72255325317383,
        "epoch": 24,
        "iteration": 537,
        "elapsed_time": 35288.715991313
    },
    {
        "main/loss": 30.151264190673828,
        "validation/main/loss": 36.41862869262695,
        "epoch": 25,
        "iteration": 560,
        "elapsed_time": 36786.500013282
    },
    {
        "main/loss": 29.47521209716797,
        "validation/main/loss": 36.37636184692383,
        "epoch": 26,
        "iteration": 582,
        "elapsed_time": 38222.206470927995
    },
    {
        "main/loss": 29.4910831451416,
        "validation/main/loss": 35.80398941040039,
        "epoch": 27,
        "iteration": 605,
        "elapsed_time": 39718.820271518
    },
    {
        "main/loss": 29.025218963623047,
        "validation/main/loss": 38.9838752746582,
        "epoch": 28,
        "iteration": 627,
        "elapsed_time": 41154.827504402
    },
    {
        "main/loss": 28.540803909301758,
        "validation/main/loss": 35.91774368286133,
        "epoch": 29,
        "iteration": 649,
        "elapsed_time": 42595.043810254996
    },
    {
        "main/loss": 28.57017707824707,
        "validation/main/loss": 36.373291015625,
        "epoch": 30,
        "iteration": 672,
        "elapsed_time": 44090.887716863996
    },
    {
        "main/loss": 28.31005859375,
        "validation/main/loss": 35.62309646606445,
        "epoch": 31,
        "iteration": 694,
        "elapsed_time": 45526.768883049
    },
    {
        "main/loss": 27.982988357543945,
        "validation/main/loss": 40.36832809448242,
        "epoch": 32,
        "iteration": 716,
        "elapsed_time": 46963.61090816
    },
    {
        "main/loss": 27.69352149963379,
        "validation/main/loss": 37.733036041259766,
        "epoch": 33,
        "iteration": 739,
        "elapsed_time": 48459.260907586
    },
    {
        "main/loss": 28.128908157348633,
        "validation/main/loss": 37.34853744506836,
        "epoch": 34,
        "iteration": 761,
        "elapsed_time": 49894.523755835995
    },
    {
        "main/loss": 26.895465850830078,
        "validation/main/loss": 39.753150939941406,
        "epoch": 35,
        "iteration": 784,
        "elapsed_time": 51388.796647293
    },
    {
        "main/loss": 27.60863494873047,
        "validation/main/loss": 37.26274490356445,
        "epoch": 36,
        "iteration": 806,
        "elapsed_time": 52823.168985561
    },
    {
        "main/loss": 26.661705017089844,
        "validation/main/loss": 37.32076644897461,
        "epoch": 37,
        "iteration": 828,
        "elapsed_time": 54258.117151545
    },
    {
        "main/loss": 26.164566040039062,
        "validation/main/loss": 36.26625061035156,
        "epoch": 38,
        "iteration": 851,
        "elapsed_time": 55751.088324751
    },
    {
        "main/loss": 26.075428009033203,
        "validation/main/loss": 37.5301399230957,
        "epoch": 39,
        "iteration": 873,
        "elapsed_time": 57182.276494467
    },
    {
        "main/loss": 25.433504104614258,
        "validation/main/loss": 36.65082931518555,
        "epoch": 40,
        "iteration": 895,
        "elapsed_time": 58618.057015756
    },
    {
        "main/loss": 25.42970848083496,
        "validation/main/loss": 35.66102600097656,
        "epoch": 41,
        "iteration": 918,
        "elapsed_time": 60136.260797407
    },
    {
        "main/loss": 25.534120559692383,
        "validation/main/loss": 39.847015380859375,
        "epoch": 42,
        "iteration": 940,
        "elapsed_time": 61578.468839821995
    },
    {
        "main/loss": 24.76921844482422,
        "validation/main/loss": 37.53437805175781,
        "epoch": 43,
        "iteration": 963,
        "elapsed_time": 63076.098307586995
    },
    {
        "main/loss": 24.222938537597656,
        "validation/main/loss": 38.27592849731445,
        "epoch": 44,
        "iteration": 985,
        "elapsed_time": 64512.46275914
    },
    {
        "main/loss": 23.5605525970459,
        "validation/main/loss": 37.53805923461914,
        "epoch": 45,
        "iteration": 1007,
        "elapsed_time": 65945.198503918
    },
    {
        "main/loss": 23.5427303314209,
        "validation/main/loss": 37.163143157958984,
        "epoch": 46,
        "iteration": 1030,
        "elapsed_time": 67439.939771238
    },
    {
        "main/loss": 23.17066764831543,
        "validation/main/loss": 41.32304000854492,
        "epoch": 47,
        "iteration": 1052,
        "elapsed_time": 68874.13750722201
    },
    {
        "main/loss": 22.829599380493164,
        "validation/main/loss": 40.03902053833008,
        "epoch": 48,
        "iteration": 1074,
        "elapsed_time": 70309.34692461301
    },
    {
        "main/loss": 22.286989212036133,
        "validation/main/loss": 41.890384674072266,
        "epoch": 49,
        "iteration": 1097,
        "elapsed_time": 71805.608371926
    },
    {
        "main/loss": 23.363439559936523,
        "validation/main/loss": 39.887332916259766,
        "epoch": 50,
        "iteration": 1119,
        "elapsed_time": 73240.72793141501
    },
    {
        "main/loss": 22.36461067199707,
        "validation/main/loss": 40.884315490722656,
        "epoch": 51,
        "iteration": 1142,
        "elapsed_time": 74755.37678678901
    },
    {
        "main/loss": 21.743364334106445,
        "validation/main/loss": 38.93526077270508,
        "epoch": 52,
        "iteration": 1164,
        "elapsed_time": 76194.19318422601
    },
    {
        "main/loss": 21.02189826965332,
        "validation/main/loss": 40.33576583862305,
        "epoch": 53,
        "iteration": 1186,
        "elapsed_time": 77638.794792963
    },
    {
        "main/loss": 20.69464683532715,
        "validation/main/loss": 37.587974548339844,
        "epoch": 54,
        "iteration": 1209,
        "elapsed_time": 79163.095673249
    },
    {
        "main/loss": 19.722301483154297,
        "validation/main/loss": 39.1914176940918,
        "epoch": 55,
        "iteration": 1231,
        "elapsed_time": 80618.65342132
    },
    {
        "main/loss": 19.430070877075195,
        "validation/main/loss": 40.299312591552734,
        "epoch": 56,
        "iteration": 1253,
        "elapsed_time": 82078.26046536
    },
    {
        "main/loss": 19.1760196685791,
        "validation/main/loss": 41.26410675048828,
        "epoch": 57,
        "iteration": 1276,
        "elapsed_time": 83594.09737514
    },
    {
        "main/loss": 18.467679977416992,
        "validation/main/loss": 40.829017639160156,
        "epoch": 58,
        "iteration": 1298,
        "elapsed_time": 85059.84887161401
    },
    {
        "main/loss": 18.32601547241211,
        "validation/main/loss": 46.54167556762695,
        "epoch": 59,
        "iteration": 1321,
        "elapsed_time": 86597.16690694801
    },
    {
        "main/loss": 17.562780380249023,
        "validation/main/loss": 39.96908950805664,
        "epoch": 60,
        "iteration": 1343,
        "elapsed_time": 88043.971549626
    },
    {
        "main/loss": 17.16754913330078,
        "validation/main/loss": 41.290645599365234,
        "epoch": 61,
        "iteration": 1365,
        "elapsed_time": 89508.530064121
    },
    {
        "main/loss": 17.277416229248047,
        "validation/main/loss": 41.66575241088867,
        "epoch": 62,
        "iteration": 1388,
        "elapsed_time": 91037.621563844
    },
    {
        "main/loss": 17.520301818847656,
        "validation/main/loss": 44.67304992675781,
        "epoch": 63,
        "iteration": 1410,
        "elapsed_time": 92502.14045006201
    },
    {
        "main/loss": 16.77066993713379,
        "validation/main/loss": 41.157188415527344,
        "epoch": 64,
        "iteration": 1432,
        "elapsed_time": 93963.428427613
    },
    {
        "main/loss": 16.40085220336914,
        "validation/main/loss": 41.747169494628906,
        "epoch": 65,
        "iteration": 1455,
        "elapsed_time": 95463.97890476501
    },
    {
        "main/loss": 16.044147491455078,
        "validation/main/loss": 41.79047393798828,
        "epoch": 66,
        "iteration": 1477,
        "elapsed_time": 96906.899469208
    },
    {
        "main/loss": 15.45360279083252,
        "validation/main/loss": 41.93582534790039,
        "epoch": 67,
        "iteration": 1500,
        "elapsed_time": 98408.999387524
    },
    {
        "main/loss": 15.20047378540039,
        "validation/main/loss": 42.71977615356445,
        "epoch": 68,
        "iteration": 1522,
        "elapsed_time": 99862.875680168
    },
    {
        "main/loss": 14.734149932861328,
        "validation/main/loss": 41.93739700317383,
        "epoch": 69,
        "iteration": 1544,
        "elapsed_time": 101331.95299334101
    },
    {
        "main/loss": 14.39233684539795,
        "validation/main/loss": 44.40830993652344,
        "epoch": 70,
        "iteration": 1567,
        "elapsed_time": 102860.949359844
    }
]

loss = [log[i]["main/loss"] for i in range(len(log))]
lossval = [log[i]["validation/main/loss"] for i in range(len(log))]

print(loss)

fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
ax.plot(loss, label="main/loss")
ax.plot(lossval, label="validation/main/loss")
ax.set_ylim([0, 50])
ax.set_xlim([0, 75])
ax.set_xlabel("epoch")
ax.legend()
plt.show()
