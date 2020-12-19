m = {"TP": 3000, "FN": 7, "FP": 10, "TN": 10}
pr_1 = (m["TP"] + m["FP"]) / (m["TP"] + m["FN"] + m["FP"] + m["TN"])
recall = m["TP"]/(m["TP"]+m["FN"])
precision = m["TP"]/(m["TP"]+m["FP"])

llm = (m["TP"] * (m["TP"] + m["FN"] + m["FP"] + m["TN"])) / ((m["TP"] + m["FP"]) * (m["TP"] + m["FN"]) ** 2)
llm_2 = (recall**2)/pr_1
F1_score = recall*precision/(recall+precision)
print(llm)
print(llm_2)
print(F1_score)
